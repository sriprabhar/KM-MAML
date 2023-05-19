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

    def __init__(self, root, acc_factors,dataset_types,mask_types,train_or_valid,mask_path): # acc_factor can be passed here and saved as self variable
        #files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.datasetdict = {'mrbrain_t1320x320':1,'kneeMRI320x320':2}
        self.maskdict={'cartesian':1,'gaussian':2}
 
        self.mask_path = mask_path 
        for dataset_type in dataset_types:
            dataroot = os.path.join(root, dataset_type)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot, mask_type,train_or_valid)
                for acc_factor in acc_factors:
                    #print("acc_factor: ", acc_factor)
                    files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor))).iterdir())
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice,acc_factor,mask_type, dataset_type = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:
            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double
            
            acc_val = float(acc_factor[:-1].replace("_","."))
            mask_val = self.maskdict[mask_type] 
            dataset_val = self.datasetdict[dataset_type]

            gamma_input = np.array([acc_val, mask_val,dataset_val])

            input_img, mask,input_kspace =CreateZeroFilledImageFn(target,acc_val,mask_type) 
            input_kspace = npComplexToTorch(input_kspace)

            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(gamma_input), acc_factor, mask_type,dataset_type,torch.from_numpy(mask)
            #return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),torch.from_numpy(mask)

            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,acc_factor,dataset_type,mask_type,mask_path):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.mask_path = mask_path
        self.datasetdict = {'mrbrain_t1320x320':1,'kneeMRI320x320':2}
        self.maskdict={'cartesian':1,'gaussian':2}
 
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]

          

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice, acc_factor,mask_type, dataset_type = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor)
            key_kspace = 'kspace_volus_{}'.format(acc_factor)
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]

            acc_val = float(acc_factor[:-1].replace("_","."))
            mask_val = self.maskdict[mask_type] 
            dataset_val = self.datasetdict[dataset_type]
 
            gamma_input = np.array([acc_val, mask_val,dataset_val])
 
            mask = np.load(os.path.join(self.mask_path,dataset_type, mask_type,'mask_{}.npy'.format(acc_factor)))
            #return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(mask), str(fname.name),slice

            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),str(fname.name),slice, torch.from_numpy(gamma_input), acc_factor, mask_type, dataset_type, torch.from_numpy(mask)

class SliceDisplayDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root,acc_factor,dataset_type,mask_path):
    def __init__(self, root,dataset_type,mask_type,acc_factor,mask_path):

        # List the h5 files in root 
        newroot = os.path.join(root, dataset_type,mask_type,'validation','acc_{}'.format(acc_factor))
        files = list(pathlib.Path(newroot).iterdir())
        self.examples = []
        self.datasetdict = {'mrbrain_t1320x320':1,'kneeMRI320x320':2}
        self.maskdict={'cartesian':1,'gaussian':2}
 
        #self.acc_factor = acc_factor
        #self.dataset_type = dataset_type

        #self.key_img = 'img_volus_{}'.format(self.acc_factor)
        #self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        #for acc_factor in acc_factors:

        #files = list(pathlib.Path(os.path.join(root,'acc_{}'.format(acc_factor))).iterdir())
        #print(files)
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                #acc_factor = float(acc_factor[:-1].replace("_","."))
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice, acc_factor,mask_type,dataset_type = self.examples[i]
        # Print statements 
        #print (type(fname),slice)
    
        with h5py.File(fname, 'r') as data:

            #key_img = 'img_volus_{}'.format(acc_factor)
            #key_kspace = 'kspace_volus_{}'.format(acc_factor)
            #input_img  = data[key_img][:,:,slice]
            #input_kspace  = data[key_kspace][:,:,slice]
            #input_kspace = npComplexToTorch(input_kspace)
            #target = data['volfs'][:,:,slice]
            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
 

            #if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                #input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                #target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            acc_val = float(acc_factor[:-1].replace("_","."))

            input_img, mask,input_kspace =CreateZeroFilledImageFn(target,acc_val,mask_type) 
            input_kspace = npComplexToTorch(input_kspace)
            mask_val = self.maskdict[mask_type] 
            dataset_val = self.datasetdict[dataset_type]

            maskvector = mask.flatten()
 
            gamma_input = np.array([acc_val, mask_val,dataset_val])
            #gamma_input_final = np.append(gamma_input,maskvector)
            #gamma_input_final = maskvector*dataset_val #pass only the mask as context
 
            #gamma_input_final = np.append(gamma_input,maskvector)
            #gamma_input = np.array([acc_val,dataset_val])
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),torch.from_numpy(gamma_input), acc_factor, mask_type, dataset_type, torch.from_numpy(mask)
 
class SliceDisplayDataDevold(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,dataset_type,mask_type,acc_factor,mask_path):

        newroot = os.path.join(root, dataset_type,mask_type,'validation','acc_{}'.format(acc_factor))
        # List the h5 files in root 
        files = list(pathlib.Path(newroot).iterdir())
        self.examples = []
        self.acc_factor = acc_factor
        self.dataset_type = dataset_type

        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        mask_path = os.path.join(mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor))
        self.mask_path = mask_path

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                #print(hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)
            mask = np.load(self.mask_path)
            #return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),mask
 
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),torch.from_numpy(gamma_input), acc_factor, mask_type, dataset_type, torch.from_numpy(mask)
