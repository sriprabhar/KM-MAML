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

class ConvBlockTaskEnc(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob, affine=False):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans,affine=affine),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, input):
        return self.layers(input)


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

    def forward(self, inp):
        return self.layers(inp)

    def adaptation(self,x,weights1,weights2):
        #print(x.shape,weights1.shape,weights2.shape)
        x = F.conv2d(x,weights1,weights2,stride=1,padding=1)
        x = F.instance_norm(x)
        x = F.relu(x)
        return x

class UnetModel(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlockTaskEnc(in_chans, chans, drop_prob,False)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlockTaskEnc(ch, ch * 2, drop_prob,False)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)
        self.convinst = ConvBlockTaskEnc(ch, ch, drop_prob, True)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlockTaskEnc(ch * 2, ch // 2, drop_prob, False)]
            ch //= 2
        self.up_sample_layers += [ConvBlockTaskEnc(ch * 2, ch, drop_prob, False)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)
        output_latent = output #b, 128,30,30
        outstyle = F.adaptive_avg_pool3d(output_latent,[output_latent.shape[1],1,1])#b,128,1,1
        stylelatentvector = torch.mean(outstyle,dim=0)#128,1,1

        output = self.convinst(output)
        output_latent2 = output
        outcontent = F.adaptive_avg_pool3d(output_latent,[output_latent2.shape[1],1,1])
        contentlatentvector = torch.mean(outcontent,dim=0)
        csdvector = torch.cat((contentlatentvector, stylelatentvector),dim=0)
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        output = self.conv2(output)
        output = output + input
        return output,csdvector[:,0,0] #256


class UnetKernelModulationNetworkEncDecSmallest(nn.Module):
    def __init__(self,contextvectorsize,in_chans, out_chans, chans, num_pool_layers):
        super(UnetKernelModulationNetworkEncDecSmallest,self).__init__()
        
        self.num_pool_layers = num_pool_layers
        
        self.alpha_down = nn.ModuleList()
        self.down_weight_size = []

        self.alpha_latent = nn.ModuleList()
        self.latent_weight_size = []

        self.alpha_up = nn.ModuleList()
        self.up_weight_size = []

        self.alpha_down += [nn.Linear(contextvectorsize,1)]
        self.down_weight_size.append([in_chans,chans])
        print([in_chans,chans])

        ch = chans

        for i in range(num_pool_layers - 1):
            self.alpha_down += [nn.Linear(contextvectorsize,1)]
            self.down_weight_size.append([ch,ch*2])
            ch *= 2
            
        self.alpha_latent += [nn.Linear(contextvectorsize,1)]
        self.latent_weight_size.append([ch,ch])
        
        for i in range(num_pool_layers - 1):
            self.alpha_up += [nn.Linear(contextvectorsize,1)]
            self.up_weight_size.append([ch*2,ch//2])
            ch //= 2
        
        self.alpha_up += [nn.Linear(contextvectorsize,1)]
        self.up_weight_size.append([ch*2,ch])
        
    def forward(self, gamma_val):
        down_mod_weights=[]
        up_mod_weights = []
#         print(gamma_val)

        for i in range(self.num_pool_layers):
            alphadown = self.alpha_down[i](gamma_val)            
#             print(self.down_weight_size[i][1],self.down_weight_size[i][0],alphadown[0][0], alphadown.shape)
            downmodulationweight = alphadown.repeat(self.down_weight_size[i][1],self.down_weight_size[i][0])#torch.full((self.down_weight_size[i][1],self.down_weight_size[i][0]),alphadown.item())
#             print(downmodulationweight.is_cuda)
            down_mod_weights.append(downmodulationweight.unsqueeze(2).unsqueeze(3))        
        
        alphalatent = self.alpha_latent[0](gamma_val)      
        latentmodulationweight = alphalatent.repeat(self.latent_weight_size[0][1],self.latent_weight_size[0][0]) #torch.full((self.latent_weight_size[0][1],self.latent_weight_size[0][0]),alphalatent.item())
        latentmodulationweight = latentmodulationweight.unsqueeze(2).unsqueeze(3)        
        
        for i in range(self.num_pool_layers):
            alphaup = self.alpha_up[i](gamma_val)            
            upmodulationweight = alphaup.repeat(self.up_weight_size[i][1],self.up_weight_size[i][0]) #torch.full((self.up_weight_size[i][1],self.up_weight_size[i][0]),alphaup.item())
            up_mod_weights.append(upmodulationweight.unsqueeze(2).unsqueeze(3))

        return down_mod_weights, latentmodulationweight,up_mod_weights
    
    def adaptation(self,gamma_val, weights):
        down_mod_weights=[]
        up_mod_weights = []
        
        alphadownlist = []

        for i in range(0,self.num_pool_layers):       
            alphadown = F.linear(gamma_val,weights[2*i],weights[(2*i)+1])
            alphadownlist.append(alphadown)
        
        for i in range(0,self.num_pool_layers):  
            downmodulationweight = alphadownlist[i].repeat(self.down_weight_size[i][1],self.down_weight_size[i][0])
            #torch.full((self.down_weight_size[i][1],self.down_weight_size[i][0]),alphadownlist[i].item())
            down_mod_weights.append(downmodulationweight.unsqueeze(2).unsqueeze(3))
        
        latentindex = self.num_pool_layers*2
        
        alphalatent = F.linear(gamma_val,weights[latentindex],weights[latentindex+1])
        latentmodulationweight = alphalatent.repeat(self.latent_weight_size[0][1],self.latent_weight_size[0][0]) #torch.full((self.latent_weight_size[0][1],self.latent_weight_size[0][0]),alphalatent.item()) #torch.matmul(torch.t(betalatent),alphalatent) 
        latentmodulationweight = latentmodulationweight.unsqueeze(2).unsqueeze(3)

        upindex = 2 + latentindex 
        alphauplist = []
        betauplist = []
        
        for i in range(0,self.num_pool_layers):       
            alphaup = F.linear(gamma_val,weights[upindex + (2*i)],weights[(upindex + (2*i))+1])
            alphauplist.append(alphaup)
        
        upindex = upindex +  (self.num_pool_layers*2)
            
        for i in range(0,self.num_pool_layers):  
            upmodulationweight = alphauplist[i].repeat(self.up_weight_size[i][1],self.up_weight_size[i][0]) #torch.full((self.up_weight_size[i][1],self.up_weight_size[i][0]),alphauplist[i].item())#torch.matmul(torch.t(betauplist[i]),alphauplist[i])
            up_mod_weights.append(upmodulationweight.unsqueeze(2).unsqueeze(3))
        
        return down_mod_weights, latentmodulationweight,up_mod_weights


class UnetMACReconNetEncDecKM(nn.Module):
    def __init__(self, args, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        super().__init__()

        super(UnetMACReconNetEncDecKM,self).__init__()
        conv_blocks = []
        dc=[]
        
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
    

    def forward(self,us_query_input,ksp_query_imgs,ksp_mask_query,down_mod_weights, latent_mod_weights, up_mod_weights):
        stack = []
        output = us_query_input
        # Apply down-sampling layers
        for i in range(self.num_pool_layers):
            self.downweights = list(self.down_sample_layers[i].parameters())
            downweights_new = self.layermodulation(self.downweights[0],down_mod_weights[i])
            output = self.parameterized_model(output, downweights_new, self.downweights[1])
            #output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)    
        
        self.latentweights = list(self.conv.parameters())
        
        latentweights_new = self.layermodulation(self.latentweights[0],latent_mod_weights) 
        output = self.parameterized_model(output,latentweights_new,self.latentweights[1])
            
        for i in range(self.num_pool_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            self.upweights = list(self.up_sample_layers[i].parameters())
            upweights_new = self.layermodulation(self.upweights[0],up_mod_weights[i])
            output = self.parameterized_model(output, upweights_new, self.upweights[1])
            #output = layer(output)
        
        output = self.conv2(output)        
        fs_out = output + us_query_input
        output = self.dc[0](fs_out,ksp_query_imgs,ksp_mask_query)
        return output
    
    def parameterized_model(self,x,weight_val,bias_val):
        x = F.conv2d(x,weight_val,bias_val,stride=1,padding=1)
        x = F.instance_norm(x)
        x = F.relu(x)
        return x
        
    def layermodulation(self, weights, modulation_weights):
        weights_new = torch.mul(modulation_weights,weights)
        return weights_new

    def modulation(self, weights, down_mod_weights, latent_mod_weights, up_mod_weights):
        weights_new=[]
        for i in range(self.num_pool_layers):
            downweights = weights[2*i]
            downweights_new = self.layermodulation(downweights,down_mod_weights[i])
            downbias = weights[(2*i)+1]
            weights_new.append(downweights_new)
            weights_new.append(downbias)

        latentweights = weights[2*self.num_pool_layers]
        latentweights_new = self.layermodulation(latentweights,latent_mod_weights)
        latentbias = weights[(2*self.num_pool_layers)+1]
        weights_new.append(latentweights_new)
        weights_new.append(latentbias)
        for i in range(self.num_pool_layers):
            upweights = weights[2*(self.num_pool_layers+1+i)]
            upweights_new = self.layermodulation(upweights,up_mod_weights[i])
            upbias = weights[2*(self.num_pool_layers+1+i)+1]
            weights_new.append(upweights_new)
            weights_new.append(upbias)

        finalweightindex = (self.num_pool_layers*2)+1
        weights_new.append(weights[2*finalweightindex])
        weights_new.append(weights[(2*finalweightindex)+1])
        weights_new.append(weights[(2*finalweightindex)+2])
        weights_new.append(weights[(2*finalweightindex)+3])
        weights_new.append(weights[(2*finalweightindex)+4])
        weights_new.append(weights[(2*finalweightindex)+5])

        return weights_new

    def adaptation(self, us_support_input,weights,ksp_support_imgs,ksp_mask_support):
        stack = []
        output = us_support_input

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
    def __init__(self,args,nc=2):
        super(DC_CNN_MAML,self).__init__()
        self.nc = nc
        conv_blocks = []
        dcs = []
        for i in range(nc):
            cnnblock = UnetMACReconNetKM(in_chans=1, out_chans=1, chans=32, num_pool_layers=3, drop_prob=0.0)
            conv_blocks.append(cnnblock)    
            dcs.append(DataConsistencyLayer(args.device))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)


    def forward(self,us_query_input,ksp_query_imgs,ksp_mask_query,latent_mod_weights, up_mod_weights):
        x = us_query_input
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x.double(), latent_mod_weights, up_mod_weights)
            #x = x_cnn+x
            x = self.dcs[i](x, ksp_query_imgs, ksp_mask_query)
        return x

    def adaptation(self,us_support_input,weights,ksp_support_imgs,ksp_mask_support):
        x = us_support_input
        for i in range(self.nc):
            convblockweights=[]
            numlayerInSingleUnet = len(weights)
            # replace 20 by numlayerInSingleUnet
            for j in range(20): 
                convblockweights.append(weights[j + (20*i)])
            x = self.conv_blocks[i].adaptation(x.double(),convblockweights)
            x = self.dcs[i](x,ksp_support_imgs,ksp_mask_support)
        return x


