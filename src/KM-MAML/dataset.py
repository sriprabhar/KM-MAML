import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch,CreateZeroFilledImageFn



class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,mask_type,train_valid_support_or_query): # acc_factor can be passed here and saved as self variable
        self.examples = []
        self.root = root
        newroot = os.path.join(root,"datasets",dataset_type,mask_type,train_valid_support_or_query,'acc_{}'.format(acc_factor))
        #print(newroot)
        files = list(pathlib.Path(newroot).iterdir())
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                #acc_factor = float(acc_factor[:-1].replace("_","."))
                self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice,acc_factor,mask_type,dataset_type = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:
            key_img = 'img_volus_{}'.format(acc_factor)
            key_kspace = 'kspace_volus_{}'.format(acc_factor)

            input_img  = data[key_img][:,:,slice]
            #print(key_img)
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)

            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double
            
            mask_path = os.path.join(self.root,"usmasks",dataset_type,mask_type,"mask_{}.npy".format(acc_factor))

            #us_mask = torch.from_numpy(np.load(mask_path)).unsqueeze(2).unsqueeze(0)
            us_mask = torch.from_numpy(np.load(mask_path))
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), acc_factor, mask_type,dataset_type,us_mask,str(fname)

class SliceDataOntheflyMask(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,mask_type,train_valid_support_or_query): # acc_factor can be passed here and saved as self variable
        self.examples = []
        self.root = root
        newroot = os.path.join(root,"datasets",dataset_type,mask_type,train_valid_support_or_query,'acc_{}'.format(acc_factor))
        #print(newroot)
        files = list(pathlib.Path(newroot).iterdir())
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                #acc_factor = float(acc_factor[:-1].replace("_","."))
                self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice,acc_factor,mask_type,dataset_type = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:

            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double
            
            acc_val = float(acc_factor[:-1].replace("_","."))
            input_img,mask,input_kspace = CreateZeroFilledImageFn(target,acc_val,mask_type)
            input_kspace = npComplexToTorch(input_kspace)

            us_mask = torch.from_numpy(mask)
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), acc_factor, mask_type,dataset_type,us_mask,str(fname)

#datasetdict = {'mrbrain_t1_few':0,'mrbrain_flair_few':1,'mrbrain_ir_few':2,'ixi_pd_few':3,'ixi_t2_few':4}
datasetdict = {'mrbrain_t1_few':0,'mrbrain_flair_few':1,'mrbrain_ir_few':2,'ixi_pd_few':3,'ixi_t2_few':4, 'sri24_t2_few':4,'sri24_pd_few':3, 'sri24_t1_few':0}
maskdict={'cartesian':0,'gaussian':1}

def GetTaskIDFromTaskString(task_string):
    for dataset_key in datasetdict.keys():
        if dataset_key in task_string:
            dataset_val = datasetdict[dataset_key]

    for masktype_key in maskdict.keys():
        if masktype_key in task_string:
            mask_val = maskdict[masktype_key]
    acc_val = float(task_string[len(task_string)-2])
    task_id = np.array([acc_val, mask_val,dataset_val])
    #print("task id for vis:",task_id)
    return torch.from_numpy(task_id)

    
class taskdataset(Dataset):
    """
    A PyTorch Dataset that provides different tasks.
    """

    def __init__(self,task_list):

        self.task_list = task_list

    def __len__(self):
        
        return len(self.task_list)

    def __getitem__(self,index):
         
        task_mini_batch = self.task_list[index]
        
        for dataset_key in datasetdict.keys():
            if dataset_key in task_mini_batch:
                dataset_val = datasetdict[dataset_key]

        for masktype_key in maskdict.keys():
            if masktype_key in task_mini_batch:
                mask_val = maskdict[masktype_key]
        acc_val = float(task_mini_batch[len(task_mini_batch)-2])
        task_id = np.array([acc_val, mask_val,dataset_val])
        #print("task id:",task_id)
        return task_mini_batch,torch.from_numpy(task_id)


class slice_indices(Dataset):
    """
    A PyTorch Dataset that provides indices for a task based on the total amount of data.
    """

    def __init__(self,datapoints):

        self.datapoints = datapoints

    def __len__(self):

        return len(self.datapoints)

    def __getitem__(self,index):

        datapoints_mini_batch = self.datapoints[index]

        return datapoints_mini_batch

