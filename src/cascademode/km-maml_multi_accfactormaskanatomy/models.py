import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 
import torch.nn.functional as F

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)

class DataConsistencyLayer(nn.Module):

    def __init__(self):

        super(DataConsistencyLayer,self).__init__()

        #self.device = device

    def forward(self,predicted_img,us_kspace,us_mask):

#         us_mask_path = os.path.join(self.us_mask_path,dataset_string,mask_string,'mask_{}.npy'.format(acc_factor))

#         us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(self.device)
        #print(predicted_img.shape, us_kspace.shape, us_mask.shape)
        us_mask = us_mask.unsqueeze(0)
        us_kspace = us_kspace.unsqueeze(0)
        predicted_img = predicted_img[:,0,:,:]

        #print("predicted_img: ",predicted_img.shape)
        #print("us_kspace: ", us_kspace.shape, "us_mask: ",us_mask.shape)
        kspace_predicted_img = torch.fft.fft2(predicted_img,norm = "ortho")

        #print("kspace_predicted_img: ",kspace_predicted_img.shape)
        kspace_predicted_img_real = torch.view_as_real(kspace_predicted_img)
        #print("kspace_predicted_img_real: ",kspace_predicted_img_real.shape)

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
        
        update_img_abs = updated_img[:,:,:,0] # taking real part only, change done on Sep 18 '19 bcos taking abs till bring in the distortion due to imag part also. this was verified was done by simple experiment on FFT, mask and IFFT

        update_img_abs = update_img_abs.unsqueeze(1)
        #print("updated_img_abs out of DC: ", update_img_abs.shape)

        return update_img_abs.float()

class DecoupleModel(nn.Module):
    
    def __init__(self, args, n_ch=1):
        
        super(DecoupleModel,self).__init__()
        
        dcs = []
        cascades = []
        instance_norms = []

        self.relu = nn.ReLU() 
        self.weights = {'fc1':[32,1,3,3],
                        'fc2':[32,32,3,3],
                        'fc3':[32,32,3,3],
                        'fc4':[32,32,3,3],
                        'fc5':[1,32,3,3]}
        
        for ii in range(5):

            cascade  = nn.ModuleDict({
                'fc1':nn.Linear(1,np.prod(self.weights['fc1'])),
                'fc2':nn.Linear(1,np.prod(self.weights['fc2'])),
                'fc3':nn.Linear(1,np.prod(self.weights['fc3'])),
                'fc4':nn.Linear(1,np.prod(self.weights['fc4'])),
                'fc5':nn.Linear(1,np.prod(self.weights['fc5']))})

            instance_norm = nn.ModuleDict({
                'fc1':nn.InstanceNorm2d(32,affine=True),
                'fc2':nn.InstanceNorm2d(32,affine=True),
                'fc3':nn.InstanceNorm2d(32,affine=True),
                'fc4':nn.InstanceNorm2d(32,affine=True) })
 
            cascades.append(cascade)
            instance_norms.append(instance_norm)
            dcs.append(DataConsistencyLayer(args.usmask_path, args.device))
            
        self.cascades = nn.ModuleList(cascades)
        self.instance_norms = nn.ModuleList(instance_norms)
        self.dcs = nn.ModuleList(dcs) # module list is added
   
        '''
        for m in self.modules():
            #print("module: ",m)
            if isinstance(m, nn.Linear):
                n = m.weight.shape[0]
                print("weight size: ",m.out_channels)
        '''
        
    def forward(self,x,k,acc,acc_string):
        #print("x enter: ", x.size())
        batch_size = x.size(0)
        batch_outputs = []
        for n in range(batch_size):        
            xout = x[n]
            xout = xout.unsqueeze(0)
            #print("xout shape in: ",xout.shape)
            for cascade_no in range(5):
                xtemp = xout
                for fc_no in self.cascades[cascade_no].keys():
                    
                    conv_weight = self.cascades[cascade_no][fc_no](acc[n])
                    conv_weight = torch.reshape(conv_weight,self.weights[fc_no])

                    #conv_weight = conv_weight.double()
                    #xout = xout.double()
                    #print ("conv weight shape: ",conv_weight.shape)
                    if fc_no=='fc5':
                        xout = F.conv2d(xout, conv_weight, bias=None, stride=1,padding=1)
                    else:
                        xout = self.relu(self.instance_norms[cascade_no][fc_no](F.conv2d(xout, conv_weight, bias=None, stride=1,padding=1)))
                    #print("xout shape: ",xout.shape)
            
                xout = self.dcs[cascade_no](xout,k[n],acc_string[n])
                xout = xout + xtemp
             
            #print("xout shape out: ",xout.shape)
            batch_outputs.append(xout)
        output = torch.cat(batch_outputs, dim=0)
        return output  

class DecoupleModelOnlyWang_etal(nn.Module):
    
    def __init__(self, args,n_ch=1):
        
        super(DecoupleModelOnlyWang_etal,self).__init__()
        

        self.relu = nn.ReLU() 
        self.weights = {'fc1':[32,1,3,3],
                        'fc2':[32,32,3,3],
                        'fc3':[32,32,3,3],
                        'fc4':[32,32,3,3],
                        'fc5':[1,32,3,3]}
        

        cascade  = nn.ModuleDict({
            'fc1':nn.Linear(1,np.prod(self.weights['fc1'])),
            'fc2':nn.Linear(1,np.prod(self.weights['fc2'])),
            'fc3':nn.Linear(1,np.prod(self.weights['fc3'])),
            'fc4':nn.Linear(1,np.prod(self.weights['fc4'])),
            'fc5':nn.Linear(1,np.prod(self.weights['fc5']))})

        instance_norm = nn.ModuleDict({
            'fc1':nn.InstanceNorm2d(32,affine=True),
            'fc2':nn.InstanceNorm2d(32,affine=True),
            'fc3':nn.InstanceNorm2d(32,affine=True),
            'fc4':nn.InstanceNorm2d(32,affine=True) })
        dc = DataConsistencyLayer(args.usmask_path, args.device)
            
        self.layer = nn.ModuleList([cascade])
        #print(self.layer[0].keys())
        self.instance_norm = nn.ModuleList([instance_norm])
        
        self.dc = nn.ModuleList([dc])
         
        
    def forward(self,x, k, acc, acc_string):
    #def forward(self,x, acc):
        #print("x enter: ", x.size())
        batch_size = x.size(0)
        batch_outputs = []
        for n in range(batch_size):        
            xout = x[n]
            xout = xout.unsqueeze(0)
            #print("xout shape in: ",xout.shape)
            xtemp = xout

            for fc_no in self.layer[0].keys():
                #print(fc_no)    
                conv_weight = self.layer[0][fc_no](acc[n])
                conv_weight = torch.reshape(conv_weight,self.weights[fc_no])

                if fc_no=='fc5':
                    xout = F.conv2d(xout, conv_weight, bias=None, stride=1,padding=1)
                else:
                    xout = self.relu(self.instance_norm[0][fc_no](F.conv2d(xout, conv_weight, bias=None, stride=1,padding=1)))
                    #print("xout shape: ",xout.shape)
            
            xout = xout + xtemp
            xout = self.dc[0](xout,k[n],acc_string[n])
             
            #print("xout shape out: ",xout.shape)
            batch_outputs.append(xout)
        output = torch.cat(batch_outputs, dim=0)
        return output  

#layer = DecoupleModelOnlyWang_etal()
#print(layer)

class DC_CNN(nn.Module):
    
    def __init__(self, args, checkpoint_file, n_ch=1,nc=5):
        
        super(DC_CNN,self).__init__()
        
        cnn_blocks = []
        #dc_blocks = []
        checkpoint = torch.load(checkpoint_file)
        self.nc = nc
        
        for ii in range(self.nc): 
            
            cnn = DecoupleModelOnlyWang_etal(args)
            cnn.load_state_dict(checkpoint['model']) 
            cnn_blocks.append(cnn)
            
            #dc_blocks.append(DataConsistencyLayer(args.usmask_path, args.device))
        
        self.cnn_blocks = nn.ModuleList(cnn_blocks)
        #self.dc_blocks  = nn.ModuleList(dc_blocks)
        
    def forward(self,x,k,acc,acc_string):
        x_cnn = x
        for i in range(self.nc):
            x_cnn = self.cnn_blocks[i](x_cnn,k,acc,acc_string)
            #x = x + x_cnn
            #x = self.dc_blocks[i](x,k,acc_string)        
        return x_cnn  

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
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
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
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
        dc = DataConsistencyLayer()

        self.dc = nn.ModuleList([dc])
        

    def forward(self, us_input, ksp, mask):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = us_input
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
        fs_out = output + us_input
        output = self.dc[0](fs_out,ksp,mask)
        return output
 
class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=5, nd=5,**kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd

#        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
#        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = conv_block


        for i in range(nc):
            unetmodel = UnetModel(in_chans=1,out_chans=1,chans=32,num_pool_layers=3,drop_prob=0.0)
            conv_blocks.append(unetmodel)
            #conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            #dcs.append(DataConsistencyLayer(args.usmask_path, args.device))
            dcs.append(DataConsistencyLayer())

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    #def forward(self,x,k,acc_string):
    def forward(self,x,k,m):
        #print (x.shape)

        for i in range(self.nc):
            #print ("x: ", x.shape)
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            #x = self.dcs[i](x,k)
            x = self.dcs[i](x,k,m)        
            #print (x.shape)

        return x

class UnetKernelModulationNetworkEncDecSmallest(nn.Module):
    def __init__(self,contextvectorsize,in_chans, out_chans, chans, num_pool_layers):
        super(UnetKernelModulationNetworkEncDecSmallest,self).__init__()
        
        self.num_pool_layers = num_pool_layers
        
        self.alpha_down = nn.ModuleList()
        self.beta_down = nn.ModuleList()
        
        self.alpha_latent = nn.ModuleList()
        self.beta_latent = nn.ModuleList()

        self.alpha_up = nn.ModuleList()
        self.beta_up = nn.ModuleList()
        
        self.alpha_down += [nn.Linear(contextvectorsize,in_chans)]
        self.beta_down += [nn.Linear(contextvectorsize,chans)]
        
        ch = chans
        
        for i in range(num_pool_layers - 1):
            self.alpha_down += [nn.Linear(contextvectorsize,ch)]
            self.beta_down += [nn.Linear(contextvectorsize,ch*2)]
            ch *= 2
            
        self.alpha_latent += [nn.Linear(contextvectorsize,ch)]
        self.beta_latent += [nn.Linear(contextvectorsize,ch)]
        
        for i in range(num_pool_layers - 1):
            self.alpha_up += [nn.Linear(contextvectorsize,ch*2)]
            self.beta_up += [nn.Linear(contextvectorsize,ch//2)]
            ch //= 2
        
        self.alpha_up += [nn.Linear(contextvectorsize,ch*2)]
        self.beta_up += [nn.Linear(contextvectorsize,ch)]

        
    def forward(self, gamma_val):
        batch_size = gamma_val.size(0)

        down_batch_outputs=[]
        latent_batch_outputs=[]
        up_batch_outputs=[]

        for n in range(batch_size):
            down_mod_weights=[]
            up_mod_weights = []
#           print(gamma_val)

            for i in range(self.num_pool_layers):
                #print("gamma_val[n].shape ",gamma_val[n].shape)
                alphadown = self.alpha_down[i](gamma_val[n])            
                betadown = self.beta_down[i](gamma_val[n])
                #print('alphadown.shape = ', alphadown.shape, ' betadown.shape = ',betadown.shape)
                downmodulationweight = torch.matmul(torch.t(betadown),alphadown)
                down_mod_weights.append(downmodulationweight.unsqueeze(2).unsqueeze(3))        
        
            alphalatent = self.alpha_latent[0](gamma_val[n])      
#           print(alphalatent)
            betalatent = self.beta_latent[0](gamma_val[n])
            latentmodulationweight = torch.matmul(torch.t(betalatent),alphalatent) 
            latentmodulationweight = latentmodulationweight.unsqueeze(2).unsqueeze(3)
        
            for i in range(self.num_pool_layers):
                alphaup = self.alpha_up[i](gamma_val[n])            
                betaup = self.beta_up[i](gamma_val[n])
                upmodulationweight = torch.matmul(torch.t(betaup),alphaup)
                up_mod_weights.append(upmodulationweight.unsqueeze(2).unsqueeze(3))

            down_batch_outputs.append(down_mod_weights)
            latent_batch_outputs.append(latentmodulationweight)
            up_batch_outputs.append(up_mod_weights)

        #return down_mod_weights, latentmodulationweight,up_mod_weights
        return down_batch_outputs, latent_batch_outputs, up_batch_outputs
    

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
        
        dc = DataConsistencyLayer()#args.usmask_path, args.device)

        self.dc = nn.ModuleList([dc])
    

    def forward(self,us_query_input,ksp_query_imgs,acc_string,mask_string,dataset_type,ksp_mask_query,down_mod_weights, latent_mod_weights, up_mod_weights):
        x = us_query_input
        batch_size = x.size(0)
        batch_outputs=[]
        for n in range(batch_size):
            stack = []
            output = x[n]
            if dataset_type =='cardiac':
                output = F.pad(output,(5,5,5,5),"constant",0)
            output = output.unsqueeze(0)
            xtemp = output
            # Apply down-sampling layers
            for i in range(self.num_pool_layers):
                self.downweights = list(self.down_sample_layers[i].parameters())
                downweights_new = self.layermodulation(self.downweights[0],down_mod_weights[n][i])
                output = self.parameterized_model(output, downweights_new, self.downweights[1])
                #output = layer(output)
                stack.append(output)
                output = F.max_pool2d(output, kernel_size=2)    
        
            self.latentweights = list(self.conv.parameters())
        
            latentweights_new = self.layermodulation(self.latentweights[0],latent_mod_weights[n]) 
            output = self.parameterized_model(output,latentweights_new,self.latentweights[1])
            
            for i in range(self.num_pool_layers):
                output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
                output = torch.cat([output, stack.pop()], dim=1)
                self.upweights = list(self.up_sample_layers[i].parameters())
                upweights_new = self.layermodulation(self.upweights[0],up_mod_weights[n][i])
                output = self.parameterized_model(output, upweights_new, self.upweights[1])
                #output = layer(output)
        
            output = self.conv2(output)        
            output = output + xtemp
            if dataset_type == 'cardiac':
                output = output[:,:,5:output.shape[2]-5,5:output.shape[3]-5]
            output = self.dc[0](output,ksp_query_imgs[n],ksp_mask_query[n])
            batch_outputs.append(output)
        output = torch.cat(batch_outputs,dim=0)
        #print("output: ",output.shape)
        return output
    
    def parameterized_model(self,x,weight_val,bias_val):
        x = F.conv2d(x,weight_val,bias_val,stride=1,padding=1)
        x = F.instance_norm(x)
        x = F.relu(x)
        return x
        
    def layermodulation(self, weights, modulation_weights):
        weights_new = torch.mul(modulation_weights,weights)
        return weights_new

class DC_CNN_MAML(nn.Module):
    def __init__(self,args,nc=5):
        super(DC_CNN_MAML,self).__init__()
        self.nc = nc
        conv_blocks = []
        #dcs = []
        for i in range(nc):
            cnnblock = UnetMACReconNetEncDecKM(args, in_chans=1, out_chans=1, chans=32, num_pool_layers=3, drop_prob=0.0)
            conv_blocks.append(cnnblock)    
            #dcs.append(DataConsistencyLayer(args.device))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        #self.dcs = nn.ModuleList(dcs)

    def forward(self,us_query_input,ksp_query_imgs,acc_string,mask_string,dataset_type,ksp_mask_query, down_mod_weights, latent_mod_weights, up_mod_weights):
        x = us_query_input
        for i in range(self.nc):
            x = self.conv_blocks[i](x,ksp_query_imgs,acc_string,mask_string,dataset_type,ksp_mask_query, down_mod_weights, latent_mod_weights, up_mod_weights)
            #x = x_cnn+x
            #x = self.dcs[i](x, ksp_query_imgs, ksp_mask_query)
        return x


