import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_valid import SliceDataDev
from models import UnetKernelModulationNetworkEncDecSmallest,UnetMACReconNetEncDecKM, UnetModel
import h5py
from tqdm import tqdm

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type, args.mask_type,args.usmask_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader

def load_model(checkpoint_file,args):

    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']

    basemodel = UnetMACReconNetEncDecKM(args, in_chans=1, out_chans=1, chans=32, num_pool_layers=3, drop_prob=0.0).to(args.device).double()
    basemodel.load_state_dict(checkpoint['model'])

    km_model = UnetKernelModulationNetworkEncDecSmallest(contextvectorsize=256,in_chans=1, out_chans=1, chans=32,num_pool_layers=3).to(args.device).double() # double to make the weights in double since input type is double 
    km_model.load_state_dict(checkpoint['km_model']) 
    task_enc_checkpointfile = args.disentangle_model_path
    task_enc_checkpoint = torch.load(task_enc_checkpointfile)
    task_encoder = UnetModel(in_chans=1, out_chans=1, chans=32,num_pool_layers=3,drop_prob=0.0).to(args.device).double()
    task_encoder.load_state_dict(task_enc_checkpoint['model'])
    for param in task_encoder.parameters():
            param.requires_grad = False
    print(task_encoder) 
    return basemodel,km_model,task_encoder


def run_unet(args, model, data_loader,km_model,task_encoder):
    model.eval()
    km_model.eval()
    task_encoder.eval()
    base_weights = list(model.parameters())
    clone_weights = [w.clone() for w in base_weights]
 
    reconstructions = defaultdict(list)
    #val_task_id = torch.zeros(128).to(args.device).double()
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):
            input_img, input_kspace,target,fnames,slices,gamma_val,acc_factor_string,mask_string,dataset_string,mask = data
            input_img = input_img.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.to(args.device)
            gamma_val = gamma_val.to(args.device)
            input_img = input_img.float()
            _,val_task_id = task_encoder(input_img.double())
            val_task_id = val_task_id.unsqueeze(0)
 
            down_modulation_weights, latent_modulation_weights,up_modulation_weights = km_model(val_task_id)

            task_meta_initialization_weights = model.modulation(clone_weights,down_modulation_weights, latent_modulation_weights,up_modulation_weights)
            recons = model.adaptation(input_img.double(), task_meta_initialization_weights,input_kspace, mask)  
            recons = recons.to('cpu').squeeze(1)
            
            for i in range(recons.shape[0]):
                recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    
    data_loader = create_data_loaders(args)
    model,km_model,task_encoder = load_model(args.checkpoint,args)
    reconstructions = run_unet(args, model, data_loader,km_model, task_encoder)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='undersampling mask path')
    parser.add_argument('--mask_type',type=str,help='mask type - cartesian, gaussian')
    parser.add_argument('--disentangle-model-path', type=str,
                        help='Path to taskencoder checkpoint file.')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
