import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
import h5py
from tqdm import tqdm
import pandas as pd 
import os
import sys

from evaluate import hfn,mse,nmse,psnr,ssim,ssim_slicewise

def metrics_reconstructions(predictions,targets,metrics_info,args):

    for fname in predictions.keys():
        recons = predictions[fname]
        target = targets[fname]

        no_slices = target.shape[2]
        no_frames = target.shape[3]

        for frame_index in range(no_frames):
            for slice_index in range(no_slices):

                if np.max(target[:,:,slice_index,frame_index]) != 0:

                    target_slice = target[:,:,slice_index,frame_index]
                    recons_slice = recons[:,:,slice_index,frame_index]
                    mse_slice  = round(mse(target_slice,recons_slice),5)
                    nmse_slice = round(nmse(target_slice,recons_slice),5)
                    psnr_slice = round(psnr(target_slice,recons_slice),2)
                    ssim_slice = round(ssim_slicewise(target_slice,recons_slice),4)

                    metrics_info['MSE'].append(mse_slice)
                    metrics_info['NMSE'].append(nmse_slice)
                    metrics_info['PSNR'].append(psnr_slice)
                    metrics_info['SSIM'].append(ssim_slice)
                    metrics_info['VOLUME'].append(fname)
                    metrics_info['SLICE'].append(slice_index)
                    metrics_info['FRAME'].append(frame_index)

    return metrics_info


def get_volume_dicts(args,recons_key):

    predictions = {}
    targets = {}

    for tgt_file in pathlib.Path(args.target_path).iterdir():

        with h5py.File(tgt_file) as target, h5py.File(args.prediction_path+"/"+str(tgt_file.name)) as recons:

            target = target[recons_key]
            target = np.array(target)

            recons = np.array(recons['reconstruction'])

            predictions[tgt_file] = recons
            targets[tgt_file] = target

    return predictions,targets

def main(args):

    recons_key = 'volfs'
    predictions, targets = get_volume_dicts(args,recons_key)

    metrics_info = {'VOLUME':[],'SLICE':[],'MSE':[],'NMSE':[],'PSNR':[],'SSIM':[],'FRAME':[]}
    metrics_info = metrics_reconstructions(predictions,targets,metrics_info, args)

    individual_artifact_report_path = args.report_path / args.degradation_name

    if not os.path.exists(args.report_path):
        os.makedirs(args.report_path)

    if not os.path.exists(individual_artifact_report_path):
        os.makedirs(individual_artifact_report_path)

    csv_path = args.report_path / 'metrics_{}_{}.csv'.format(args.degradation_name,args.degradation_amount)
    individual_path = individual_artifact_report_path / 'metrics_{}_{}.csv'.format(args.degradation_name,args.degradation_amount)

    df = pd.DataFrame(metrics_info)
    df.to_csv(csv_path)
    df.to_csv(individual_path)



def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")

    parser.add_argument('--target_path',type=str,help='path to validation dataset')
    parser.add_argument('--partial_prediction_path',type=str,help='path to predicted dataset by the model')

    parser.add_argument('--report-path', type=pathlib.Path, required=True,help='Path to save metrics')

    parser.add_argument('--data_flag',type=str,help='TRAIN/VALID SUPPORT/QUERY')

    parser.add_argument('--one_task',type=str,help='Name of one task')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])

    args.degradation_name,args.degradation_amount = args.one_task.split("/")

    args.prediction_path = args.partial_prediction_path+args.degradation_name+"/amount_"+args.degradation_amount

    main(args)
