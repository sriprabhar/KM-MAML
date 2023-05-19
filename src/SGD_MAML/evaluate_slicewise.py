import argparse
import pathlib
from argparse import ArgumentParser
import os
import h5py
import numpy as np
from runstats import Statistics
#from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import laplace
from tqdm import tqdm

# adding hfn metric 
def hfn(gt,pred):

    hfn_total = []

    for ii in range(gt.shape[-1]):
        gt_slice = gt[:,:,ii]
        pred_slice = pred[:,:,ii]

        pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
        pred_slice[pred_slice>1] = 1

        gt_slice_laplace = laplace(gt_slice)        
        pred_slice_laplace = laplace(pred_slice)

        hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
        hfn_total.append(hfn_slice)

    return np.mean(hfn_total)

def hfnSlicewise(gt,pred):

    hfn_total = []

    for ii in range(1):
        gt_slice = gt
        pred_slice = pred

        pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
        pred_slice[pred_slice>1] = 1

        gt_slice_laplace = laplace(gt_slice)        
        pred_slice_laplace = laplace(pred_slice)

        hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
        hfn_total.append(hfn_slice)

    return np.mean(hfn_total)



def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    #return compare_ssim(
    #    gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    #)
    return structural_similarity(gt,pred,multichannel=True, data_range=gt.max())

METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfnSlicewise
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            metricval = func(target, recons)
            #print("metric, func, metric val: ", metric, func, metricval)         
            self.metrics[metric].push(func(target, recons))#here target and recons are not slices they are volumes

    def means(self):
        #print("self.metrics.items(): ",self.metrics.items())
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        #print("inside stddevs: ")
        #for metric, stat in self.metrics.items():
        #    print("stat: ", stat.mean(),stat.stddev())
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }


    '''
    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )
    '''

    def get_report(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )

def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)
    print("metrics done")
    print("args.target_path: ", args.target_path)
    for tgt_file in args.target_path.iterdir():
        print ("tgt file: ", tgt_file)
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
            target = target[recons_key].value
            recons = recons['reconstruction'].value
            recons = np.transpose(recons,[1,2,0])
            print(tgt_file)
            print (target.shape,recons.shape)

            metrics.push(target, recons)
            
    return metrics

def evaluateSlicewise(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)
    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
            target = target[recons_key]
            recons = recons['reconstruction']
            recons = np.transpose(recons,[1,2,0])
            for ii in range(recons.shape[2]):
                target_slice = target[:,:,ii]
                recons_slice = recons[:,:,ii]
            #print(tgt_file)
            #print (target.shape,recons.shape)

                metrics.push(target_slice, recons_slice)
            
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--base-target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--base-predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')
    parser.add_argument('--evaluate_task_strings', type=str, default='',help='Put all the tasks for evaluation')

    args = parser.parse_args()

    all_dataset_types = []

    all_mask_types = []
    all_acc_types = []

    full_tasks = args.evaluate_task_strings.split(",")
    for one_task in full_tasks:
        types = one_task.split('_')
        all_dataset_types.append(types[0] + "_" + types[1] + "_" + types[2])
        all_mask_types.append(types[3])
        all_acc_types.append(types[4])

    dataset_types = np.unique(all_dataset_types)
    mask_types = np.unique(all_mask_types)
    acc_factors = np.unique(all_acc_types)

    recons_key = 'volfs'

    for dataset_type in dataset_types:
        args.dataset_type = dataset_type
        for mask_type in mask_types:
            args.mask_type = mask_type
            for acc_factor in acc_factors:
                args.acc_factor = acc_factor

                args.target_path = pathlib.Path(os.path.join(str(args.base_target_path), args.dataset_type, args.mask_type,"valid_query","acc_" + args.acc_factor))
                
                args.predictions_path= pathlib.Path(os.path.join(str(args.base_predictions_path),args.dataset_type,args.mask_type,"acc_"+args.acc_factor))

                metrics = evaluateSlicewise(args, recons_key)
                print("evaluate done for the task:",dataset_type,mask_type,acc_factor)
                metrics_report = metrics.get_report()

                with open(args.report_path / 'report_{}_{}_{}.txt'.format(args.dataset_type,args.mask_type,args.acc_factor),'w') as f:
                    f.write(metrics_report)

                print(metrics)
