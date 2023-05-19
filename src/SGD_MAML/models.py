import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 
import torch.nn.functional as F
from collections import OrderedDict


class DataConsistencyLayer(nn.Module):

    def __init__(self,device):

        super(DataConsistencyLayer,self).__init__()

        self.device = device

    def forward(self,predicted_img,us_kspace,us_mask):

#         us_mask_path = os.path.join(self.us_mask_path,dataset_string,mask_string,'mask_{}.npy'.format(acc_factor))

#         us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(self.device)
        #print(predicted_img.shape, us_kspace.shape, us_mask.shape)
        predicted_img = predicted_img[:,0,:,:]

        #print("predicted_img: ",predicted_img.shape)
        #print("us_kspace: ", us_kspace.shape, "us_mask: ",us_mask.shape)
        kspace_predicted_img = torch.fft.fft2(predicted_img,norm = "ortho")

        #print("kspace_predicted_img: ",kspace_predicted_img.shape)
        #kspace_predicted_img_real = torch.view_as_real(kspace_predicted_img)

        us_kspace_complex = us_kspace[:,:,:,0]+us_kspace[:,:,:,1]*1j

        updated_kspace1  = us_mask * us_kspace_complex

        updated_kspace2  = (1 - us_mask) * kspace_predicted_img
        #print("updated_kspace1: ", updated_kspace1.shape, "updated_kspace2: ",updated_kspace2.shape)

        updated_kspace = updated_kspace1 + updated_kspace2
        #print("updated_kspace: ", updated_kspace.shape)
        
        #updated_kspace = updated_kspace[:,:,:,0]+updated_kspace[:,:,:,1]*1j
        #print("updated_kspace: ", updated_kspace.shape)

        updated_img  = torch.fft.ifft2(updated_kspace,norm = "ortho")
        #print("updated_img: ", updated_img.shape)

        updated_img = torch.view_as_real(updated_img)
        #print("updated_img: ", updated_img.shape)
        
        update_img_abs = updated_img[:,:,:,0] 

        update_img_abs = update_img_abs.unsqueeze(1)

        return update_img_abs.float()

class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
                      nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
                      nn.InstanceNorm2d(out_chans),
                      nn.ReLU(),
                      nn.Dropout2d(drop_prob),
                    )

    def forward(self, input):
        return self.layers(input)

    def adaptation(self,x,weights1,weights2):
        #print(x.shape,weights1.shape,weights2.shape)
        x = F.conv2d(x,weights1,weights2,stride=1,padding=1)
        x = F.instance_norm(x)
        x = F.relu(x)
        return x

class UnetModel(nn.Module):

    def __init__(self, args,in_chans, out_chans, chans, num_pool_layers, drop_prob):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

        dc = DataConsistencyLayer(args.device)

        self.dc = nn.ModuleList([dc])
        

    def forward(self, us_query_input, ksp_query_imgs, ksp_mask_query):
        stack = []
        output = us_query_input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)
        output_latent = output
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        output = self.conv2(output) 
        fs_out = output + us_query_input
        output = self.dc[0](fs_out,ksp_query_imgs,ksp_mask_query)
        return output
 
    def adaptation(self, us_support_input, weights,ksp_support_imgs,ksp_mask_support):
        stack = []
        output = us_support_input
        #         ch = chans
        for i in range(self.num_pool_layers):
            output = self.down_sample_layers[i].adaptation(output,weights[2*i],weights[(2*i)+1])
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        output = self.conv.adaptation(output,weights[2*self.num_pool_layers],weights[(2*self.num_pool_layers)+1])

        for i in range(self.num_pool_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = self.up_sample_layers[i].adaptation(output,weights[2*(self.num_pool_layers+1+i)],weights[2*(self.num_pool_layers+1+i)+1])

        finalweightindex = (self.num_pool_layers*2)+1

        output = F.conv2d(output,weights[2*finalweightindex],weights[(2*finalweightindex)+1])
        output = F.conv2d(output,weights[(2*finalweightindex)+2],weights[(2*finalweightindex)+3])
        output = F.conv2d(output,weights[(2*finalweightindex)+4],weights[(2*finalweightindex)+5])
        cnn_support_output = output + us_support_input
        fs_support_output = self.dc[0](cnn_support_output,ksp_support_imgs,ksp_mask_support)

        return fs_support_output 



class DC_CNN_MAML(nn.Module):
    def __init__(self,args,n_ch=1,nc=1):
        super(DC_CNN_MAML,self).__init__()
        self.nc = nc
        conv_blocks = []
        dcs = []
        for i in range(nc):
            cnnblock = UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=3, drop_prob=0.0)
            conv_blocks.append(cnnblock)    
            dcs.append(DataConsistencyLayer(args.device))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)


    def forward(self, us_query_input,ksp_query_imgs,ksp_mask_query):
        x = us_query_input
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x.double())
            #x = x_cnn+x
            x = self.dcs[i](x_cnn, ksp_query_imgs, ksp_mask_query)
        return x

    def adaptation(self,us_support_input,weights,ksp_support_imgs,ksp_mask_support):
        x = us_support_input
        for i in range(self.nc):
            convblockweights=[]
            for j in range(20):
                convblockweights.append(weights[j + (20*i)])
            x = self.conv_blocks[i].adaptation(x.double(),convblockweights)
            x = self.dcs[i](x,ksp_support_imgs,ksp_mask_support)
        return x


