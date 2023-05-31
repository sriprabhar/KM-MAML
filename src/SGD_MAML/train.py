import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse
import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset

from dataset import taskdataset,SliceData,SliceDataOntheflyMask,slice_indices
from models import DataConsistencyLayer,DC_CNN_MAML,UnetModel

import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
import torchviz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def CreateLoadersForAllTasks(args,train_flag):
    support_loaderdict = {}
    query_loaderdict = {}
    
    support_task_size = {}
    query_task_size = {}

    support_dataset_dict = {}
    query_dataset_dict = {}

    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')
    train_path = args.train_path
    for dataset_type in dataset_types:
        for mask_type in mask_types:
            for acc_factor in acc_factors:
                #print(dataset_type,mask_type,acc_factor)
                if train_flag:
                    support_task_dataset = SliceDataOntheflyMask(train_path,acc_factor,dataset_type,mask_type,"support")
                    query_task_dataset = SliceDataOntheflyMask(train_path,acc_factor,dataset_type,mask_type,"query")
                    support_task_loader = DataLoader(dataset=support_task_dataset, batch_size = args.train_support_batch_size,shuffle = True)                
                    query_task_loader = DataLoader(dataset=query_task_dataset, batch_size = args.train_query_batch_size,shuffle = True)
 
                elif not train_flag:
                    support_task_dataset = SliceData(train_path,acc_factor,dataset_type,mask_type,"valid_support")
                    query_task_dataset = SliceData(train_path,acc_factor,dataset_type,mask_type,"valid_query")

                    support_task_loader = DataLoader(dataset=support_task_dataset, batch_size = args.val_support_batch_size,shuffle = True)                
                    query_task_loader = DataLoader(dataset=query_task_dataset, batch_size = args.val_query_batch_size,shuffle = True)
                
                loaderkey = dataset_type+"_"+mask_type+"_"+acc_factor

                support_task_size[loaderkey] = len(support_task_dataset)
                query_task_size[loaderkey] = len(query_task_dataset)
                
                support_dataset_dict[loaderkey] = support_task_dataset
                query_dataset_dict[loaderkey] = query_task_dataset

                support_loaderdict[loaderkey] = support_task_loader
                query_loaderdict[loaderkey] = query_task_loader

    return support_loaderdict,query_loaderdict,support_dataset_dict,query_dataset_dict,support_task_size,query_task_size


def get_indices_for_all_tasks(args,support_task_size_dict,query_task_size_dict,train_or_valid_flag):
    
    support_index_dict = {}
    query_index_dict = {}

    max_epochs = args.num_epochs

    if train_or_valid_flag:
        support_batch_size = args.train_support_batch_size
        no_of_adapt_steps = args.no_of_train_adaptation_steps
        query_batch_size = args.train_query_batch_size
    elif not train_or_valid_flag:
        support_batch_size = args.val_support_batch_size
        no_of_adapt_steps = args.no_of_val_adaptation_steps
        query_batch_size = args.val_query_batch_size
    
    for task_name in support_task_size_dict.keys():
        break_flag = False
        index_points = np.arange(0,support_task_size_dict[task_name])
        tensor_index_points = torch.from_numpy(index_points)
        index_dataset = slice_indices(tensor_index_points)
        index_loader = DataLoader(index_dataset,batch_size = support_batch_size*no_of_adapt_steps,shuffle=True) # length of this loader is len(index_dataset) / (support_batch_size * no_of_adapt_steps)
        support_index_dict[task_name] = []
        
        while True: # enumerate until max  number of epoch times. within exhaustion of the index_loader, we ensure that the mini-batches are unique
            for _,index_mini_batch in enumerate(index_loader):
                support_index_dict[task_name].append(index_mini_batch)
                if len(support_index_dict[task_name]) >= max_epochs:
                    break_flag = True
                    break
            if break_flag:
                break
    # repeat the above logic for query
    for task_name in query_task_size_dict.keys():
        break_flag = False
        index_points = np.arange(0,query_task_size_dict[task_name])
        tensor_index_points = torch.from_numpy(index_points)
        index_dataset = slice_indices(tensor_index_points)
        index_loader = DataLoader(index_dataset,batch_size = query_batch_size*1,shuffle=True) # the len of this loader is index_dataset / (query_batch_size*1)
        query_index_dict[task_name] = []
        
        while True:
            for _,index_mini_batch in enumerate(index_loader):
                query_index_dict[task_name].append(index_mini_batch)
                if len(query_index_dict[task_name]) >= max_epochs:
                    break_flag = True
                    break
            if break_flag:
                break

    return support_index_dict,query_index_dict # len (dict ) is number of  tasks. Within tasks there is list of len = max_epochs


def CreateLoadersForVisualize(args,task_string):
    task_tokens = task_string.split('_')
    dataset_type=task_tokens[0]+'_'+task_tokens[1]+'_'+task_tokens[2]
    mask_type = task_tokens[3]
    acc_factor = task_tokens[4]
    displaydataset = SliceData(args.train_path,acc_factor,dataset_type,mask_type,"valid_query")
    display1 = [displaydataset[i] for i in range(0, len(displaydataset), len(displaydataset) // 16)]
    display_loader = DataLoader(dataset=display1, batch_size = 16,shuffle = True)
    return display_loader
    



def maml_train_epoch(args,epoch,model,train_task_loader,train_support_dataset,train_query_dataset,train_support_task_index,train_query_task_index,meta_optimizer,writer):
    
    model.train()
    avg_train_meta_loss = 0 # per epoch average query loss
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_task_loader)

    save_computational_graph = True

    for iter,train_task_mini_batch in enumerate(train_task_loader):

        train_meta_loss = 0 # per task-mini-batch average query loss

        for train_task in train_task_mini_batch:

            train_task_support_subdataset = Subset(train_support_dataset[train_task],train_support_task_index[train_task][epoch]) # train_support_dataset[train_task] is the dataset of all samples of that task. train_support_task_index[train_task][epoch] is the index set sampled per train task per epoch. train_support_task_index[train_task][epoch] has args.train_support_batch_size * args.no_of_train_adaptation_steps elements in it
            
            train_task_support_subloader = DataLoader(train_task_support_subdataset,batch_size = args.train_support_batch_size,shuffle = True) # create a subloader from the sub-dataset . Thisi loader has batch size = args.train_support_batch_size, which means it has a length of number of support adaptation steps

            train_task_query_subdataset = Subset(train_query_dataset[train_task],train_query_task_index[train_task][epoch]) # train_query_dataset[train_task] will have all slices but train_query_task_index[train_task][epoch]  will have train_query_batch_size number of indices 
            
            train_task_query_subloader = DataLoader(train_task_query_subdataset,batch_size = args.train_query_batch_size,shuffle = True) # len of this subloader will be 1 since it query we want to find the query loss for just one query mini-batch because len(loader) = len(subdataset) / train_query_batch_size = 1

            base_weights = list(model.parameters())
            clone_weights = [w.clone() for w in base_weights]

            for gd_steps,spt_data_batch in enumerate(train_task_support_subloader): # this loop automatically executes only for args.no_of_train_adaptation_steps times, hence an inner if block was removed

                #print("in train if gd steps: ",gd_steps,args.no_of_train_adaptation_steps)
                spt_us_imgs = spt_data_batch[0].unsqueeze(1).to(args.device)
                spt_ksp_imgs = spt_data_batch[1].to(args.device)
                spt_fs_imgs = spt_data_batch[2].unsqueeze(1).to(args.device)
                spt_ksp_mask = spt_data_batch[6].to(args.device)
                spt_acc_string = spt_data_batch[3][0]
                spt_mask_string = spt_data_batch[4][0]
                spt_dataset_string = spt_data_batch[5][0]

                spt_fs_pred = model.adaptation(spt_us_imgs,clone_weights,spt_ksp_imgs,spt_ksp_mask)
                
                spt_loss = F.l1_loss(spt_fs_pred,spt_fs_imgs)
                
                #clone_grads = torch.autograd.grad(spt_loss, clone_weights,create_graph=True)
                clone_grads = torch.autograd.grad(spt_loss, clone_weights)
                clone_weights = [w-args.adapt_lr*g for w, g in zip(clone_weights,clone_grads)]
            # adaptation completed for the current task in hand
            for _,qry_data_batch in enumerate(train_task_query_subloader): 
                qry_us_imgs = qry_data_batch[0].unsqueeze(1).to(args.device)
                qry_ksp_imgs = qry_data_batch[1].to(args.device)
                qry_fs_imgs = qry_data_batch[2].unsqueeze(1).to(args.device)
                qry_ksp_mask = qry_data_batch[6].to(args.device)
                qry_acc_string = qry_data_batch[3][0]
                qry_mask_string = qry_data_batch[4][0]
                qry_dataset_string = qry_data_batch[5][0]
                #break # no need to break since this train_task_query_subloader sub-loader is of len 1

            qry_fs_pred = model.adaptation(qry_us_imgs,clone_weights,qry_ksp_imgs,qry_ksp_mask)
            qry_loss = F.l1_loss(qry_fs_pred,qry_fs_imgs) # F.l1_loss will give the average loss per mini-batch of query data points in that task
            if save_computational_graph:
                torchviz.make_dot(qry_loss, params=dict(model.named_parameters())).render("graph", format="png")
                save_computational_graph = False
            train_meta_loss = train_meta_loss+qry_loss # accumulate query loss per task-minibatch
            # end of one train task in train task-minibatch loop
        
        meta_grads = torch.autograd.grad(train_meta_loss/len(train_task_mini_batch),base_weights) # train_meta_loss/len(train_task_mini_batch) is the average over num of train tasks  in traintask-minibatch
        avg_train_meta_loss = avg_train_meta_loss+train_meta_loss.item()/len(train_task_mini_batch) # Averaging over number of tasks in a task minibatch, train_meta_loss wwill be reset to 0 when we sample the next mini-batch. So we are accumulating this over train_task_mini_batch which is used down for further dividing by len of loader for getting the final average query  loss over all tasks

        meta_optimizer.zero_grad()
        for w,g in zip(base_weights,meta_grads):
            w.grad = g
        meta_optimizer.step()

        per_train_task_minibatch_avg_meta_loss = train_meta_loss.item()/len(train_task_mini_batch)

        writer.add_scalar('iter_train_qry_loss',per_train_task_minibatch_avg_meta_loss,global_step+iter) # per task-minibatch loss - that is why iter
    

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(train_task_loader):4d}] '
                f'Loss = {per_train_task_minibatch_avg_meta_loss :.4g} Avg Loss = {avg_train_meta_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        # end of task-mini-batch for loop
    avg_train_meta_loss = avg_train_meta_loss/len(train_task_loader) # Averaging over the number of task minibatches which is the len of the train_task loader

    writer.add_scalar('per_epoch_avg_train_qry_loss',avg_train_meta_loss,epoch)

    return avg_train_meta_loss,time.perf_counter()-start_epoch


def maml_validate(args,epoch,model,val_task_loader,val_support_dataset,val_query_dataset,val_support_task_index,val_query_task_index,writer):
    
    #model.eval()
    avg_val_meta_loss = 0
    start = time.perf_counter()
    
    for iter,val_task_mini_batch in enumerate(val_task_loader):
        
        val_meta_loss = 0
        
        for val_task in val_task_mini_batch:
            
            val_task_support_subdataset = Subset(val_support_dataset[val_task],val_support_task_index[val_task][epoch])
            val_task_support_subloader = DataLoader(val_task_support_subdataset,batch_size = args.val_support_batch_size,shuffle = True)

            val_task_query_subdataset = Subset(val_query_dataset[val_task],val_query_task_index[val_task][epoch])
            val_task_query_subloader = DataLoader(val_task_query_subdataset,batch_size = args.val_query_batch_size,shuffle = True)

            base_weights = list(model.parameters())
            clone_weights = [w.clone() for w in base_weights]

            for gd_steps,val_spt_data_batch in enumerate(val_task_support_subloader):

                model.train()
                val_spt_us_imgs = val_spt_data_batch[0].unsqueeze(1).to(args.device)
                val_spt_ksp_imgs = val_spt_data_batch[1].to(args.device)
                val_spt_fs_imgs = val_spt_data_batch[2].unsqueeze(1).to(args.device)
                val_spt_ksp_mask = val_spt_data_batch[6].to(args.device)
                val_spt_acc_string = val_spt_data_batch[3][0]
                val_spt_mask_string = val_spt_data_batch[4][0]
                val_spt_dataset_string = val_spt_data_batch[5][0]

                val_spt_fs_pred = model.adaptation(val_spt_us_imgs,clone_weights,val_spt_ksp_imgs,val_spt_ksp_mask)
                
                val_spt_loss = F.l1_loss(val_spt_fs_pred,val_spt_fs_imgs)
                
                clone_grads = torch.autograd.grad(val_spt_loss, clone_weights)
                clone_weights = [w-args.adapt_lr*g for w, g in zip(clone_weights,clone_grads)]

            for _,val_qry_data_batch in enumerate(val_task_query_subloader):
                model.eval()
                val_qry_us_imgs = val_qry_data_batch[0].unsqueeze(1).to(args.device)
                val_qry_ksp_imgs = val_qry_data_batch[1].to(args.device)
                val_qry_fs_imgs = val_qry_data_batch[2].unsqueeze(1).to(args.device)
                val_qry_ksp_mask = val_qry_data_batch[6].to(args.device)
                val_qry_acc_string = val_qry_data_batch[3][0]
                val_qry_mask_string = val_qry_data_batch[4][0]
                val_qry_dataset_string = val_qry_data_batch[5][0]
                #break
            
            with torch.no_grad(): # this is added to ensure that gradients do not occupy the gpu mem
                val_qry_fs_pred = model.adaptation(val_qry_us_imgs,clone_weights,val_qry_ksp_imgs,val_qry_ksp_mask)
                val_qry_loss = F.l1_loss(val_qry_fs_pred,val_qry_fs_imgs)
            
            val_meta_loss = val_meta_loss+val_qry_loss
            
        avg_val_meta_loss = avg_val_meta_loss+val_meta_loss.item()/len(val_task_mini_batch)

    avg_val_meta_loss = avg_val_meta_loss/len(val_task_loader)

    writer.add_scalar('per_epoch_avg_val_qry_loss',avg_val_meta_loss,epoch)
    
    return avg_val_meta_loss,time.perf_counter()-start

def sgd_train_epoch(args,epoch,model,train_task_loader,train_support_dataset,train_query_dataset,train_support_task_index,train_query_task_index,meta_optimizer,writer):
    
    model.train()
    avg_train_meta_loss = 0 # per epoch average query loss
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_task_loader)

    save_computational_graph = True

    for iter,train_task_mini_batch in enumerate(train_task_loader):

        train_qry_loss = 0 # per task-mini-batch average query loss

        for train_task in train_task_mini_batch:

            train_task_support_subdataset = Subset(train_support_dataset[train_task],train_support_task_index[train_task][epoch]) # train_support_dataset[train_task] is the dataset of all samples of that task. train_support_task_index[train_task][epoch] is the index set sampled per train task per epoch. train_support_task_index[train_task][epoch] has args.train_support_batch_size * args.no_of_train_adaptation_steps elements in it
            
            train_task_support_subloader = DataLoader(train_task_support_subdataset,batch_size = args.train_support_batch_size,shuffle = True) # create a subloader from the sub-dataset . Thisi loader has batch size = args.train_support_batch_size, which means it has a length of number of support adaptation steps

            train_task_query_subdataset = Subset(train_query_dataset[train_task],train_query_task_index[train_task][epoch]) # train_query_dataset[train_task] will have all slices but train_query_task_index[train_task][epoch]  will have train_query_batch_size number of indices 
            
            train_task_query_subloader = DataLoader(train_task_query_subdataset,batch_size = args.train_query_batch_size,shuffle = True) # len of this subloader will be 1 since it query we want to find the query loss for just one query mini-batch because len(loader) = len(subdataset) / train_query_batch_size = 1

            base_weights = list(model.parameters())

            for gd_steps,spt_data_batch in enumerate(train_task_support_subloader): # this loop automatically executes only for args.no_of_train_adaptation_steps times, hence an inner if block was removed

                #print("in train if gd steps: ",gd_steps,args.no_of_train_adaptation_steps)
                spt_us_imgs = spt_data_batch[0].unsqueeze(1).to(args.device)
                spt_ksp_imgs = spt_data_batch[1].to(args.device)
                spt_fs_imgs = spt_data_batch[2].unsqueeze(1).to(args.device)
                spt_ksp_mask = spt_data_batch[6].to(args.device)
                spt_acc_string = spt_data_batch[3][0]
                spt_mask_string = spt_data_batch[4][0]
                spt_dataset_string = spt_data_batch[5][0]

                spt_fs_pred = model(spt_us_imgs,spt_ksp_imgs,spt_ksp_mask)
                
                spt_loss = F.l1_loss(spt_fs_pred,spt_fs_imgs)

                base_grads = torch.autograd.grad(spt_loss,base_weights,create_graph=False)
                updated_base_weights = [w-args.adapt_lr*g for w, g in zip(base_weights,base_grads)]

                model_dict = model.state_dict()
                model_keys = [k for k, v in model_dict.items()]

                for k,t in zip(model_keys,updated_base_weights):
                    # model_dict[k] = t
                    model_dict[k] = t.detach()
                model.load_state_dict(model_dict)

                base_weights = list(model.parameters())
               
            # adaptation completed for the current task in hand
            for _,qry_data_batch in enumerate(train_task_query_subloader): 
                qry_us_imgs = qry_data_batch[0].unsqueeze(1).to(args.device)
                qry_ksp_imgs = qry_data_batch[1].to(args.device)
                qry_fs_imgs = qry_data_batch[2].unsqueeze(1).to(args.device)
                qry_ksp_mask = qry_data_batch[6].to(args.device)
                qry_acc_string = qry_data_batch[3][0]
                qry_mask_string = qry_data_batch[4][0]
                qry_dataset_string = qry_data_batch[5][0]
                #break # no need to break since this train_task_query_subloader sub-loader is of len 1

            qry_fs_pred = model(qry_us_imgs,qry_ksp_imgs,qry_ksp_mask)
            qry_loss = F.l1_loss(qry_fs_pred,qry_fs_imgs) # F.l1_loss will give the average loss per mini-batch of query data points in that task
            if save_computational_graph:
                torchviz.make_dot(qry_loss, params=dict(model.named_parameters())).render("graph", format="png")
                save_computational_graph = False

            #train_meta_loss = train_meta_loss+qry_loss # This line is commented for SGD as we do not take the average query loss over task minibatch. accumulate query loss per task-minibatch
            # end of one train task in train task-minibatch loop
        
            meta_grads = torch.autograd.grad(qry_loss,base_weights) # For SGD, we directly ccompute the gradients with respect to the base weights computed for each task in traintask-minibatch
            meta_optimizer.zero_grad()
            for w,g in zip(base_weights,meta_grads):
                w.grad = g
            meta_optimizer.step()

            train_qry_loss = train_qry_loss+qry_loss # for SGD, perform accumulation of query loss with task-minibatch here

            # end of one train task in train task-minibatch loop
        avg_train_qry_loss = avg_train_qry_loss+train_qry_loss.item()/len(train_task_mini_batch) # Averaging over number of tasks in a task minibatch, train_qry_loss wwill be reset to 0 when we sample the next mini-batch. So we are accumulating this over train_task_mini_batch which is used down for further dividing by len of loader for getting the final average query  loss over all tasks
        per_train_task_minibatch_avg_qry_loss = train_qry_loss.item()/len(train_task_mini_batch)

        writer.add_scalar('iter_train_qry_loss',per_train_task_minibatch_avg_qry_loss,global_step+iter) # per task-minibatch loss - that is why iter
    

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(train_task_loader):4d}] '
                f'Loss = {per_train_task_minibatch_avg_meta_loss :.4g} Avg Loss = {avg_train_meta_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        # end of task-mini-batch for loop
    avg_train_qry_loss = avg_train_qry_loss/len(train_task_loader) # Averaging over the number of task minibatches which is the len of the train_task loader

    writer.add_scalar('per_epoch_avg_train_qry_loss',avg_train_qry_loss,epoch)

    return avg_train_meta_loss,time.perf_counter()-start_epoch

def sgd_validate_wo_adaptation(args,epoch,model,val_task_loader,val_support_dataset,val_query_dataset,val_support_task_index,val_query_task_index,writer):
    
    #model.eval()
    unadapted_avg_val_qry_loss = 0
    adapted_avg_val_qry_loss = 0.0  ## Return zero if adaptation is not performed
    global_step = epoch * len(val_task_loader)
    start = time.perf_counter()
    
    for iter,val_task_mini_batch in enumerate(val_task_loader):
        
        unadapted_val_qry_loss = 0
        
        for val_task in val_task_mini_batch:
            
            val_task_query_subdataset = Subset(val_query_dataset[val_task],val_query_task_index[val_task][epoch])
            val_task_query_subloader = DataLoader(val_task_query_subdataset,batch_size = args.val_query_batch_size,shuffle = True)

            for _,val_qry_data_batch in enumerate(val_task_query_subloader):
                val_qry_us_imgs = val_qry_data_batch[0].unsqueeze(1).to(args.device)
                val_qry_ksp_imgs = val_qry_data_batch[1].to(args.device)
                val_qry_fs_imgs = val_qry_data_batch[2].unsqueeze(1).to(args.device)
                val_qry_ksp_mask = val_qry_data_batch[6].to(args.device)
                val_qry_acc_string = val_qry_data_batch[3][0]
                val_qry_mask_string = val_qry_data_batch[4][0]
                val_qry_dataset_string = val_qry_data_batch[5][0]
                #break
            
            model.eval()
            with torch.no_grad(): # this is added to ensure that gradients do not occupy the gpu mem
                val_qry_fs_pred = model(val_qry_us_imgs,val_qry_ksp_imgs,val_qry_ksp_mask)
                unadapted_qry_loss = F.l1_loss(val_qry_fs_pred,val_qry_fs_imgs)
            
            unadapted_val_qry_loss = unadapted_val_qry_loss+unadapted_qry_loss.item()
            
        unadapted_avg_val_qry_loss = unadapted_avg_val_qry_loss + unadapted_val_qry_loss/len(val_task_mini_batch)

    unadapted_avg_val_qry_loss = unadapted_avg_val_qry_loss/len(val_task_loader)

    writer.add_scalar('per_epoch_unadapted_avg_val_qry_loss',unadapted_avg_val_qry_loss,epoch)
    
    return unadapted_avg_val_qry_loss,adapted_avg_val_qry_loss,time.perf_counter()-start

def sgd_validate_w_adaptation(args,epoch,model,val_task_loader,val_support_dataset,val_query_dataset,val_support_task_index,val_query_task_index,writer):
    
    unadapted_avg_val_qry_loss = 0
    adapted_avg_val_qry_loss = 0.0
    global_step = epoch * len(val_task_loader)
    start = time.perf_counter()


    for iter,val_task_mini_batch in enumerate(val_task_loader):

        adapted_val_qry_loss = 0
        unadapted_val_qry_loss = 0
        
        for val_task in val_task_mini_batch:
            
            val_task_support_subdataset = Subset(val_support_dataset[val_task],val_support_task_index[val_task][epoch])
            val_task_support_subloader = DataLoader(val_task_support_subdataset,batch_size = args.val_support_batch_size,shuffle = True)

            val_task_query_subdataset = Subset(val_query_dataset[val_task],val_query_task_index[val_task][epoch])
            val_task_query_subloader = DataLoader(val_task_query_subdataset,batch_size = args.val_query_batch_size,shuffle = True)

            base_weights = list(model.parameters())
            clone_weights = [w.clone() for w in base_weights]

            for gd_steps,val_spt_data_batch in enumerate(val_task_support_subloader):

                model.train()
                val_spt_us_imgs = val_spt_data_batch[0].unsqueeze(1).to(args.device)
                val_spt_ksp_imgs = val_spt_data_batch[1].to(args.device)
                val_spt_fs_imgs = val_spt_data_batch[2].unsqueeze(1).to(args.device)
                val_spt_ksp_mask = val_spt_data_batch[6].to(args.device)
                val_spt_acc_string = val_spt_data_batch[3][0]
                val_spt_mask_string = val_spt_data_batch[4][0]
                val_spt_dataset_string = val_spt_data_batch[5][0]

                val_spt_fs_pred = model.adaptation(val_spt_us_imgs,clone_weights,val_spt_ksp_imgs,val_spt_ksp_mask)
                
                val_spt_loss = F.l1_loss(val_spt_fs_pred,val_spt_fs_imgs)
                
                clone_grads = torch.autograd.grad(val_spt_loss, clone_weights)
                clone_weights = [w-args.adapt_lr*g for w, g in zip(clone_weights,clone_grads)]

            for _,val_qry_data_batch in enumerate(val_task_query_subloader):
                val_qry_us_imgs = val_qry_data_batch[0].unsqueeze(1).to(args.device)
                val_qry_ksp_imgs = val_qry_data_batch[1].to(args.device)
                val_qry_fs_imgs = val_qry_data_batch[2].unsqueeze(1).to(args.device)
                val_qry_ksp_mask = val_qry_data_batch[6].to(args.device)
                val_qry_acc_string = val_qry_data_batch[3][0]
                val_qry_mask_string = val_qry_data_batch[4][0]
                val_qry_dataset_string = val_qry_data_batch[5][0]
                #break
            
            model.eval()
            with torch.no_grad(): # this is added to ensure that gradients do not occupy the gpu mem
                val_qry_fs_pred = model.adaptation(val_qry_us_imgs,clone_weights,val_qry_ksp_imgs,val_qry_ksp_mask)
                adapted_qry_loss = F.l1_loss(val_qry_fs_pred,val_qry_fs_imgs)

                val_qry_fs_pred = model(val_qry_us_imgs,val_qry_ksp_imgs,val_qry_ksp_mask)
                unadapted_qry_loss = F.l1_loss(val_qry_fs_pred,val_qry_fs_imgs)
            
            adapted_val_qry_loss = adapted_val_qry_loss + adapted_qry_loss.item()
            unadapted_val_qry_loss = unadapted_val_qry_loss + unadapted_qry_loss.item()
            
        adapted_avg_val_qry_loss = adapted_avg_val_qry_loss + adapted_val_qry_loss.item()/len(val_task_mini_batch)
        unadapted_avg_val_qry_loss = unadapted_avg_val_qry_loss + unadapted_val_qry_loss.item()/len(val_task_mini_batch)

    adapted_avg_val_qry_loss = adapted_avg_val_qry_loss/len(val_task_loader)

    unadapted_avg_val_qry_loss = unadapted_avg_val_qry_loss/len(val_task_loader)

    writer.add_scalar('per_epoch_adapted_avg_val_qry_loss',adapted_avg_val_qry_loss,epoch)

    writer.add_scalar('per_epoch_unadapted_avg_val_qry_loss',unadapted_avg_val_qry_loss,epoch)

    return unadapted_avg_val_qry_loss,adapted_avg_val_qry_loss,time.perf_counter()-start



def visualize(args, epoch, model, data_loader, writer,task_string):
    print("in vis")
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data_batch in enumerate(tqdm(data_loader)):
            us_imgs = data_batch[0].unsqueeze(1).to(args.device)
            ksp_imgs = data_batch[1].to(args.device)
            fs_imgs = data_batch[2].unsqueeze(1).to(args.device)
            ksp_mask = data_batch[6].to(args.device)
            acc_string = data_batch[3][0]
            mask_string = data_batch[4][0]
            dataset_string = data_batch[5][0]

            fs_pred = model(us_imgs,ksp_imgs,ksp_mask)
            save_image(us_imgs, 'Input_{}'.format(task_string))
            save_image(fs_imgs, 'Target_{}'.format(task_string))
            save_image(fs_pred, 'Reconstruction_{}'.format(task_string))
            save_image(torch.abs(fs_imgs.float() - fs_pred.float()), 'Error_{}'.format(task_string))

            break



def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    
    #model = five_layerCNN_MAML(args).to(args.device).double() # double to make the weights in double since input type is double 

    #model = DC_CNN_MAML(args).to(args.device).double() # double to make the weights in double since input type is double

    model = UnetModel(args, in_chans=1, out_chans=1, chans=32, num_pool_layers=3, drop_prob=0.0).to(args.device).double() 

    return model

def build_optim(args, params):
    optimizer = torch.optim.Adam(params,args.meta_lr,weight_decay = args.weight_decay)
    return optimizer
    

def main(args):
    
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        #checkpoint, model, optimizer, disc, optimizerD = load_model(args, args.checkpoint)
        checkpoint, model, optimizer, disc, optimizerD = load_model(args.checkpoint)
        args = checkpoint['args']
        args.batch_size = 28
        best_dev_mse= checkpoint['best_dev_mse']
        best_dev_ssim = checkpoint['best_dev_mse']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        
        model = build_model(args)

        print ("Model Built")
        if args.data_parallel:
            model = torch.nn.DataParallel(model)    
        meta_optimizer = build_optim(args,model.parameters())
        print ("Optmizer initialized")

    logging.info(args)
    logging.info(model)

    train_support_loader,train_query_loader,train_support_dataset,train_query_dataset,train_support_task_size,train_query_task_size = CreateLoadersForAllTasks(args,True)
    val_support_loader,val_query_loader,val_support_dataset,val_query_dataset,val_support_task_size,val_query_task_size = CreateLoadersForAllTasks(args,False)

    print ("Train and Validation Dataloaders are initialized")

    train_support_task_index,train_query_task_index = get_indices_for_all_tasks(args,train_support_task_size,train_query_task_size,True)
    val_support_task_index,val_query_task_index = get_indices_for_all_tasks(args,val_support_task_size,val_query_task_size,False)

    print("Train and validation indices are obtained")
    
    task_strings = args.task_strings.split(",")
    display_task_string = task_strings[0]
    
    displayloader = CreateLoadersForVisualize(args,display_task_string) 
    task_dataset_instantiation = taskdataset(task_strings)

    train_task_batch_size = args.train_task_batch_size
    train_task_loader = DataLoader(dataset = task_dataset_instantiation, batch_size = train_task_batch_size,shuffle = True)

    val_task_batch_size = args.val_task_batch_size
    val_task_loader = DataLoader(dataset = task_dataset_instantiation, batch_size = val_task_batch_size,shuffle = True)
    
    print ("Train and Validation task Dataloaders are initialized")

    #scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer,args.lr_step_size,args.lr_gamma)
    
    avg_train_qry_loss = [] # this loss will give average query loss per epoch across mini-task and across dataset (two divisions are present in the avg-train and avg-valid loops)
    avg_val_qry_loss = []
    
    best_dev_loss = 1e9
    start_epoch = 0

    for current_epoch in range(start_epoch,args.num_epochs):
        
        #scheduler.step(current_epoch)
        if args.training_style == "sgd":

            per_epoch_train_meta_loss,per_epoch_train_time = sgd_train_epoch(args,current_epoch,model,train_task_loader,train_support_dataset,train_query_dataset,train_support_task_index,train_query_task_index,meta_optimizer,writer)
            avg_train_qry_loss.append(per_epoch_train_meta_loss)
            if args.adaptation_flag_for_sgd_validation_data == 'True':
                per_epoch_unadapted_val_meta_loss,per_epoch_adapted_val_meta_loss,per_epoch_val_time = sgd_validate_w_adaptation(args,current_epoch,model,val_task_loader,val_support_dataset,val_query_dataset,val_support_task_index,val_query_task_index,writer)
            elif args.adaptation_flag_for_sgd_validation_data == 'False':
                per_epoch_unadapted_val_meta_loss,per_epoch_adapted_val_meta_loss,per_epoch_val_time = sgd_validate_wo_adaptation(args,current_epoch,model,val_task_loader,val_support_dataset,val_query_dataset,val_support_task_index,val_query_task_index,writer)
            avg_val_qry_loss.append(per_epoch_unadapted_val_meta_loss)

            visualize(args,current_epoch,model,displayloader,writer,display_task_string)

            is_new_best = per_epoch_unadapted_val_meta_loss < best_dev_loss
            best_dev_loss = min(best_dev_loss,per_epoch_unadapted_val_meta_loss)

            save_model(args,args.exp_dir,current_epoch,model,meta_optimizer,best_dev_loss,is_new_best)

            logging.info(
                f'Epoch = [{current_epoch:4d}/{args.num_epochs:4d}] TrainLoss = {per_epoch_train_meta_loss:.4g} '
                f'Unadapted Validation Loss = {per_epoch_unadapted_val_meta_loss:.4g} Adapted Validation Loss = {per_epoch_adapted_val_meta_loss:.4g} TrainTime = {per_epoch_train_time:.4f}s ValTime = {per_epoch_val_time:.4f}s',
            )


        elif args.training_style == "maml":
            per_epoch_train_meta_loss,per_epoch_train_time = maml_train_epoch(args,current_epoch,model,train_task_loader,train_support_dataset,train_query_dataset,train_support_task_index,train_query_task_index,meta_optimizer,writer)
            avg_train_qry_loss.append(per_epoch_train_meta_loss)

            per_epoch_val_meta_loss,per_epoch_val_time = maml_validate(args,current_epoch,model,val_task_loader,val_support_dataset,val_query_dataset,val_support_task_index,val_query_task_index,writer)
            avg_val_qry_loss.append(per_epoch_val_meta_loss)
        
            visualize(args, current_epoch, model, displayloader, writer,display_task_string)         
            is_new_best = per_epoch_val_meta_loss < best_dev_loss
            best_dev_loss = min(best_dev_loss,per_epoch_val_meta_loss)
            save_model(args,args.exp_dir,current_epoch,model,meta_optimizer,best_dev_loss,is_new_best)
            logging.info(
                f'Epoch = [{current_epoch:4d}/{args.num_epochs:4d}] TrainLoss = {per_epoch_train_meta_loss:.4g}'
                f'Validation Loss= {per_epoch_val_meta_loss:.4g} TrainTime = {per_epoch_train_time:.4f}s ValTime = {per_epoch_val_time:.4f}s',
            )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--device', type=str, default='cuda',help='Which device to train on. Set to "cuda" to use the GPU')
    
    parser.add_argument('--train_path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')
    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--mask_type',type=str,help='mask type - cartesian, gaussian')
    parser.add_argument('--usmask_path',type=str,help='us mask path')
    parser.add_argument('--train_support_batch_size', default=20, type=int,  help='Train Support Mini batch size')
    parser.add_argument('--train_query_batch_size', default=30, type=int,  help='Train Query Mini batch size')
 
    parser.add_argument('--val_support_batch_size', default=20, type=int,  help='Validation Support Mini batch size')
    parser.add_argument('--val_query_batch_size', default=30, type=int,  help='Validation Query Mini batch size')
    parser.add_argument('--train_task_batch_size', default=3, type=int,  help='Task Mini batch size for training')
    parser.add_argument('--val_task_batch_size', default=6, type=int,  help='Task Mini batch size for validation')
    parser.add_argument('--task_strings', type=str, default='',help='Put all the tasks')
    
    parser.add_argument('--num-epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--no_of_train_adaptation_steps', type=int, default=3, help='Number of adaptation steps during meta-training stage')
    parser.add_argument('--no_of_val_adaptation_steps', type=int, default=3, help='Number of adaptation steps during meta-validation stage')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta Learning rate')
    parser.add_argument('--adapt_lr', type=float, default=0.001, help='Task-specific Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    
    parser.add_argument('--report-interval', type=int, default=1, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    
    parser.add_argument('--training_style',type=str,help='Type of training')

    parser.add_argument('--adaptation_flag_for_sgd_validation_data', type=str, default='',help='Is adaptation required during validation data of sgd not MAML')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
