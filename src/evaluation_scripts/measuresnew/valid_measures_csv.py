import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
#from torch.utils.data import DataLoader
#from dataset_valid import SliceDataDev
#from models import UnetModel 
import h5py
from tqdm import tqdm
from evaluate import Metrics,hfn,mse,nmse,psnr,ssim
import pandas as pd 

#import sys
#currentdir = os.path.dirname(os.path.realpath(__file__))
#parentdir = os.path.dirname(currentdir)
#sys.path.append(parentdir) # this is done to import dataset_valid and models.py

METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfn
)
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

def metrics_reconstructions(predictions,targets,metrics_info,args):
    #out_dir.mkdir(exist_ok=True)
    metrics = Metrics(METRIC_FUNCS)
    #print("recons items: ", reconstructions.items())
    #print("recons items: ", reconstructions.keys())
    #for fname, [recons,target] in reconstructions.items():
    for fname in predictions.keys():
        recons = predictions[fname]
        target = targets[fname]
        
        #recons = np.transpose(recons,[1,2,0])
        #target = np.transpose(target,[1,2,0])
        no_slices = target.shape[-1]

        for index in range(no_slices):
            #print(args.dataset_type,args.mask_type,args.acceleration_factor)
            target_slice = target[:,:,index]
            recons_slice = recons[:,:,index]
            print('target_slice: ',target_slice.shape, 'recrecons_slice: ',recons_slice.shape)
            mse_slice  = round(mse(target_slice,recons_slice),5)
            nmse_slice = round(nmse(target_slice,recons_slice),5)
            psnr_slice = round(psnr(target_slice,recons_slice),2)
            ssim_slice = round(ssim(target_slice,recons_slice),4)

            metrics_info['MSE'].append(mse_slice)
            metrics_info['NMSE'].append(nmse_slice)
            metrics_info['PSNR'].append(psnr_slice)
            metrics_info['SSIM'].append(ssim_slice)
            metrics_info['VOLUME'].append(fname)
            metrics_info['SLICE'].append(index)
 
        #print (recons.shape,target.shape)
        #print (recons)
        #break
        metrics.push(target, recons)
    print ('end of metrics_reconstructions')
    print(metrics, metrics_info)
    return metrics, metrics_info 
 

def create_data_loaders(args):

    #data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type,args.usmask_path)
    #data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type)
    data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type, args.mask_type, args.usmask_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = UnetModel(args, in_chans=1, out_chans=1, chans=32, num_pool_layers=3, drop_prob=0.0).to(args.device) 
    #print(model)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model



def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input, input_kspace,target,mask, fnames,slices = data
            input = input.unsqueeze(1).to(args.device).float()
            input_kspace = input_kspace.to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.to(args.device)

            input = input.float()

            #recons = model(input,input_kspace).to('cpu').squeeze(1)
            #print (input.shape,acc_val.shape)
            recons = model(input,input_kspace, mask).to('cpu').squeeze(1)

            #if args.dataset_type == 'cardiac':
            #    recons = recons[:,5:155,5:155]
            target = target.to('cpu').squeeze(1)

            
            for i in range(recons.shape[0]):
                #recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy(),target[i].numpy()))

#    reconstructions = {
#        fname: np.stack([pred for _, pred in sorted(slice_preds)])
#        for fname, slice_preds in reconstructions.items()
#    }

    predictions = {

        fname: np.stack([pred for _, pred,_ in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    targets = {

        fname: np.stack([targ for _,_,targ in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
 
    return predictions, targets

def get_volume_dicts(args,recons_key):

    predictions = {}
    targets = {}

    for tgt_file in args.target_path.iterdir():
        print(tgt_file)
        with h5py.File(tgt_file) as target, h5py.File(args.predictions_path / tgt_file.name) as recons:
            target = target[recons_key]
            target = np.array(target)
            recons = np.array(recons['reconstruction'])
            recons = np.transpose(recons,[1,2,0])
            predictions[tgt_file] = recons
            targets[tgt_file] = target
            #print(tgt_file,target.shape,recons.shape)

    return predictions,targets



def main(args):
    
    #data_loader = create_data_loaders(args)
    #model = load_model(args.checkpoint)
    #predictions, targets = run_unet(args, model, data_loader)
    recons_key = 'volfs'
    predictions, targets = get_volume_dicts(args,recons_key)
    metrics_info = {'VOLUME':[],'SLICE':[],'MSE':[],'NMSE':[],'PSNR':[],'SSIM':[]}
    metrics, metrics_info = metrics_reconstructions(predictions,targets,metrics_info, args)

 
    individual_contrast_report_path = args.report_path / args.dataset_type
    #print ('exited')
    #save_reconstructions(reconstructions, args.out_dir)
    #metrics_report = metrics.get_report()
    #with open(args.report_path / 'report_{}.txt'.format(args.acceleration_factor),'w') as f:
    #with open(args.report_path / 'report_{}_{}_{}.txt'.format(args.dataset_type,args.mask_type,args.acceleration_factor),'w') as f:
    #    f.write(metrics_report)
    csv_path     = args.report_path / 'metrics_{}_{}_{}.csv'.format(args.dataset_type,args.mask_type,args.acceleration_factor)
    individual_path = individual_contrast_report_path / 'metrics_{}_{}_{}.csv'.format(args.dataset_type,args.mask_type,args.acceleration_factor)
    print(csv_path, individual_path)
    df = pd.DataFrame(metrics_info)
    #print(len(df))
    df.to_csv(csv_path)
    df.to_csv(individual_path)

def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--mask_type',type=str,help='mask type - cartesian, gaussian')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
 
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
