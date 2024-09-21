#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/model.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#
# This script hold the model architecture
#------------------------------------------------------------------------------

# import pytorch modules
#
from __future__ import print_function
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from SOGMP_plus.CoBEVT.opv2v.opencood.models.fusion_modules.swap_fusion_modules import SwapFusionEncoder
from SOGMP_plus.CoBEVT.opv2v.opencood.models.sub_modules.fuse_utils import regroup
from SOGMP_plus.scripts.convlstm import ConvLSTMCell

from SOGMP_plus.scripts.torch_transformation_utils import get_transformation_matrix, warp_affine

from crowd_sim.envs.utils.lidar2d import Lidar2d
#from convlstm import ConvLSTMCell
def plot_ogm(ogm, filename):
    plt.figure(figsize=(6,6))
    plt.imshow(ogm[0].detach().cpu().numpy(), cmap='gray')  # Assuming the OGM is on GPU and single-channel
    plt.colorbar()
    plt.savefig(filename)  # Saves the image to a file
    print(f"Saved {filename}")
    plt.show()
    plt.close()


# import modules
#
import os
import random

# for reproducibility, we seed the rng
#
SEED1 = 1337
NEW_LINE = "\n"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------

# function: set_seed
#
# arguments: seed - the seed for all the rng
#
# returns: none
#
# this method seeds all the random number generators and makes
# the results deterministic
#
def set_seed(seed):
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
#
# end of method


# function: get_data
#
# arguments: img_path - file pointer
#            file_name - the name of data file
#
# returns: data - the signals/features
#
# this method takes in a fp and returns the data and labels
POINTS = 90   # the number of lidar points
IMG_SIZE = 32
SEQ_LEN = 10
FUTURE_STEP=4
class VaeTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        # read the names of image data:
        self.scan_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []
        self.target_file_names = []
        # open train.txt or dev.txt:
        fp_scan = open(img_path+'/scans/'+file_name+'.txt', 'r')
        fp_pos = open(img_path+'/positions/'+file_name+'.txt', 'r')
        fp_vel = open(img_path+'/velocities/'+file_name+'.txt', 'r')
        fp_target = open(img_path+'/targets/'+file_name+'.txt', 'r')
        # for each line of the file:
        for line in fp_scan.read().split(NEW_LINE):
            if('.npy' in line): 
                self.scan_file_names.append(img_path+'/'+line)
        for line in fp_pos.read().split(NEW_LINE):
            if('.npy' in line): 
                self.pos_file_names.append(img_path+'/'+line)
        for line in fp_vel.read().split(NEW_LINE):
            if('.npy' in line): 
                self.vel_file_names.append(img_path+'/'+line)
        for line in fp_target.read().split(NEW_LINE):
            if('.npy' in line): 
                self.target_file_names.append(img_path+'/'+line)
        
        fp_scan.close()
        fp_pos.close()
        fp_vel.close()
        fp_target.close()
        self.length = len(self.scan_file_names)

        print("dataset length: ", self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx): # where is idx?
        # get the index of start point:
        scans = np.zeros((SEQ_LEN, 3,POINTS,2))
        positions = np.zeros((SEQ_LEN, 3,3))
        targets=np.zeros((SEQ_LEN,3,4,POINTS,2))
        idx=idx%self.length
        
        if idx+SEQ_LEN>= self.length:
            idx=idx-SEQ_LEN
        
        for j in range(SEQ_LEN): 
            # get the scan data:
            scan_name = self.scan_file_names[idx+j]
            scan = np.load(scan_name)
            
            scans[j,:,:,:] = scan
            # get the scan_ur data:
            pos_name = self.pos_file_names[idx+j]
            pos = np.load(pos_name)
            positions[j,:,:] = pos
            
            # get the velocity data:
            target_name = self.target_file_names[idx+j]
            target = np.load(target_name)
            
            targets[j,:,:,:,:] = target
        

        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scans)
        pose_tensor = torch.FloatTensor(positions)
        target_tensor =  torch.FloatTensor(targets)
        
        data = {
                'scan': scan_tensor,
                'position': pose_tensor,
                'target': target_tensor, 
                }
        
        return data

#
# end of function


#------------------------------------------------------------------------------
#
# the model is defined here
#
#------------------------------------------------------------------------------

# define the PyTorch VAE model
#
# define a VAE
# Residual blocks: 
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# Encoder & Decoder Architecture:
# Encoder:
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=num_hiddens//2,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens//2),
                                        nn.ReLU()
                                    ])
        self._conv_2 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=num_hiddens//2,
                                                  out_channels=num_hiddens,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens)
                                        #nn.ReLU(True)
                                    ])
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x

# Decoder:
class Decoder(nn.Module):
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_2 = nn.Sequential(*[
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(in_channels=num_hiddens,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU()
                                        ])

        self._conv_trans_1 = nn.Sequential(*[
                                            nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU(),                  
                                            nn.Conv2d(in_channels=num_hiddens//2,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.Sigmoid()
                                        ])

    def forward(self, inputs):
        x = self._residual_stack(inputs)
        x = self._conv_trans_2(x)
        x = self._conv_trans_1(x)
        return x

class VAE_Encoder(nn.Module):
    def __init__(self, input_channel):
        super(VAE_Encoder, self).__init__()
        # parameters:
        self.input_channels = input_channel
        # Constants
        num_hiddens = 128 #128
        num_residual_hiddens = 64 #32
        num_residual_layers = 2
        embedding_dim = 2 #64

        # encoder:
        in_channels = input_channel
        self._encoder = Encoder(in_channels, 
                                num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)

        # z latent variable: 
        self._encoder_z_mu = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embedding_dim,
                                    kernel_size=1, 
                                    stride=1)
        self._encoder_z_log_sd = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embedding_dim,
                                    kernel_size=1, 
                                    stride=1)  
        
    def forward(self, x):
        # input reshape:
        x = x.reshape(-1, self.input_channels, IMG_SIZE, IMG_SIZE)
        # Encoder:
        encoder_out = self._encoder(x)
        # get `mu` and `log_var`:
        z_mu = self._encoder_z_mu(encoder_out)
        z_log_sd = self._encoder_z_log_sd(encoder_out)
        return z_mu, z_log_sd

# our proposed model: SOGMP++
class RVAEP(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(RVAEP, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.z_w = int(np.sqrt(latent_dim//2))
        
        # Constants
        num_hiddens = 64 #64
        num_residual_hiddens = 64 #64
        num_residual_layers = 2
        embedding_dim = 2 
        
        # prediction encoder:
        self._convlstm = ConvLSTMCell(input_dim=self.input_channels,
                                    hidden_dim=num_hiddens//4,
                                    kernel_size=(3, 3),
                                    bias=True)
        self._encoder = VAE_Encoder((num_hiddens//4 + self.input_channels),)

        # decoder:
        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=embedding_dim, 
                                    out_channels=num_hiddens,
                                    kernel_size=1, 
                                    stride=1)
        self._decoder = Decoder(self.output_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        self.conv=nn.Sequential(nn.Conv2d(in_channels=(num_hiddens//4 + self.input_channels), 
                                    out_channels=self.output_channels,
                                    kernel_size=1, 
                                    stride=1),
                                    nn.Sigmoid())
        args = {'input_dim': 17,
            'mlp_dim': 512,
            'agent_size': 3,
            'window_size': 8,
            'dim_head': 17,
            'drop_out': 0.1,
            'depth': 2,
            'mask': True,
            'resolution': 0.2,
            'downsample_rate':1
            }
        self.fusenet=SwapFusionEncoder(args)
        # self.sttf=STTF(args)
        #self.fc=nn.Sequential(nn.Linear((num_hiddens//4 + self.input_channels)*32*32, 32*32),nn.Sigmoid())
        #self.dropout = nn.Dropout(p=self.dropout_rate)

    def vae_reparameterize(self, z_mu, z_log_sd):
        """
        :param mu: mean from the encoder's latent space
        :param log_sd: log standard deviation from the encoder's latent space
        :output: reparameterized latent variable z, Monte carlo KL divergence
        """
        # reshape:
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)
        # define the z probabilities (in this case Normal for both)
        # p(z): N(z|0,I)
        pz = torch.distributions.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_log_sd))
        # q(z|x,phi): N(z|mu, z_var)
        qz_x = torch.distributions.Normal(loc=z_mu, scale=torch.exp(z_log_sd))

        # repameterization trick: z = z_mu + xi (*) z_log_var, xi~N(xi|0,I)
        z = qz_x.rsample()
        # Monte Carlo KL divergence: MCKL(p(z)||q(z|x,phi)) = log(p(z)) - log(q(z|x,phi))
        # sum over weight dim, leaves the batch dim 
        kl_divergence = (pz.log_prob(z) - qz_x.log_prob(z)).sum(dim=1)
        kl_loss = -kl_divergence.mean()

        return z, kl_loss 
    # # this forward is for middle fusion
    # def forward(self, x, x_map,pos,ego_index):
        
    #     """
    #     Forward pass `input_img` through the network
    #     """
    #     # reconstruction: 
    #     # encode:
    #     # input reshape:
        
    #     robot_num,b,seq_len,h, w = x.size()
        
    #     x= x.reshape(robot_num,b, seq_len, 1, IMG_SIZE, IMG_SIZE)
    #     x_map = x_map.reshape(robot_num,b, 1, IMG_SIZE, IMG_SIZE)
        
        
    #     #plot_ogm(x_map[0][0],'x_map.png')
    #     robot_num,b, seq_len, c, h, w = x.size()
    #     # h_enc_list=[]
    #     # x_map_list=[]
    #     enc_list=[[] for _ in range(b)]
        
        
    #     record_len=torch.zeros(b).to(x.device)
    #     for r in range(robot_num):
            
    #         h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
    #         for t in range(seq_len): 
    #             x_in = x[r][:,t]
    #             h_enc, enc_state = self._convlstm(input_tensor=x_in,
    #                                             cur_state=[h_enc, enc_state])
    #         # for tt in range(16):
    #         #     plot_ogm(h_enc[0][tt].unsqueeze(0),'h_enc.png')
    #         # exit()
    #         distance_check=torch.sqrt((pos[:,ego_index,0]-pos[:,r,0])**2+(pos[:,ego_index,1]-pos[:,r,1])**2)<10 #b
            
    #         dx = pos[:, r, 0] - pos[:, ego_index, 0]
    #         dy = (pos[:, r, 1] - pos[:, ego_index, 1])
    #         dtheta = (pos[:, r, 2] - pos[:, ego_index, 2])

    #         # Calculate rotation matrix components
    #         cos = torch.cos(dtheta)
    #         sin = torch.sin(dtheta)

    #         # Assembling the transformation matrix (2x3)
    #         rotation_matrices = torch.zeros(b, 2, 3)  # Shape: [batch_size, 2, 3]
    #         rotation_matrices[:, 0, 0] = cos
    #         rotation_matrices[:, 0, 1] = -sin
    #         rotation_matrices[:, 1, 0] = sin
    #         rotation_matrices[:, 1, 1] = cos
    #         rotation_matrices[:, 0, 2] = dx/(32*0.3125)
    #         rotation_matrices[:, 1, 2] = dy/(32*0.3125)
            
    #         T = get_transformation_matrix(rotation_matrices, (32, 32))
            
    #         h_enc_rec = warp_affine(h_enc, T, (32, 32)) #b 16 32 32
            
    #         #h_enc_rec[~distance_check]=0.
    #         #h_enc_list.append(h_enc_rec)
            
    #         x_map_rec = warp_affine(x_map[r], T, (32, 32)) #b 1 32 32
            
    #         enc_in = torch.cat([h_enc_rec, x_map_rec], dim=1)  # 64 17 32 32
            
    #         #enc_list.append(enc_in)
    #         for bz in range(b):
    #             if distance_check[bz]:
    #                 enc_list[bz].append(enc_in[bz])
    #                 record_len[bz]+=1
                    
    #     enc_list_=[]
    #     for bz in range(b):

    #         enc_list_+=enc_list[bz]
           
        
    #     enc_in_tensor=torch.stack(enc_list_,dim=0)
        
    #     # h_enc_tensor=torch.stack(h_enc_list,dim=0) # robot_num 64 16 32 32
    #     # x_map_tensor=torch.stack(x_map_list,dim=0)  # robot_num 64 1 32 32
    #     # enc_in = torch.cat([h_enc_tensor, x_map_tensor], dim=2)  #robot_num b 17 32 32
        
        
        
        
    #     #enc_in_tensor=enc_in_tensor.permute(1,0,2,3,4).reshape(b*robot_num,17,32,32)

        
    #     regroup_feature, mask = regroup(enc_in_tensor,record_len,max_len=3)
        
        
        
    #     com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    #     # com_mask = torch.repeat(com_mask,
    #     #                   'b h w c l -> b (h new_h) (w new_w) c l',
    #     #                   new_h=regroup_feature.shape[3],
    #     #                   new_w=regroup_feature.shape[4])
    #     com_mask = com_mask.repeat(1,regroup_feature.shape[3] // com_mask.shape[2], 
    #                        regroup_feature.shape[4] // com_mask.shape[3], 1,1)
        
        
    #     fused_feature = self.fusenet(regroup_feature, com_mask)

    #     #plot_ogm(fused_feature[0],'fused_feature.png')
    #     #print(h_enc.shape)

    #     z_mu, z_log_sd = self._encoder(fused_feature)

    #     # get the latent vector through reparameterization:
    #     z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)
        
    #     # decode:
    #     # reshape:
    #     z = z.reshape(-1, 2, self.z_w, self.z_w)
    #     #print(distance_check[:,robot_index].shape) #b
        
    #     x_d = self._decoder_z_mu(z)
                
    #     prediction = self._decoder(x_d)
        
        
    #     return prediction, kl_loss
    #this forward is for no fusion and early fusion
    def forward(self, x, x_map):
        
        """
        Forward pass `input_img` through the network
        """
        # reconstruction: 
        # encode:
        # input reshape:
        
        b,seq_len, h, w = x.size()
        
        x = x.reshape(b, seq_len, 1, IMG_SIZE, IMG_SIZE)
        
        x_map = x_map.reshape(b, 1, IMG_SIZE, IMG_SIZE)
        # find size of different input dimensions
        b, seq_len, c, h, w = x.size()
        
        # encode: 
        # initialize hidden states
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t in range(seq_len): 
            x_in = x[:,t]
            h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                              cur_state=[h_enc, enc_state])
        
        #print(h_enc)
        enc_in = torch.cat([h_enc, x_map], dim=1)  
        
        #prediction = self.conv(enc_in)
        #prediction = self.fc(enc_in.reshape(b,-1)).reshape(b,1,IMG_SIZE,IMG_SIZE)
    
        #return prediction,torch.tensor(0.0).to(device)
        z_mu, z_log_sd = self._encoder(enc_in)

        # get the latent vector through reparameterization:
        z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)
        
        # decode:
        # reshape:
        z = z.reshape(-1, 2, self.z_w, self.z_w)

        x_d = self._decoder_z_mu(z)
        #x_d = self.dropout(x_d)
        
        prediction = self._decoder(x_d)
        
        
        return prediction, kl_loss

#
# end of class
class RConvLSTM(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(RConvLSTM, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.z_w = int(np.sqrt(latent_dim//2))

        # Constants
        num_hiddens = 64
        
    
        # prediction encoder:
        self._convlstm = ConvLSTMCell(input_dim=self.input_channels,
                                    hidden_dim=num_hiddens//4,
                                    kernel_size=(3, 3),
                                    bias=True)
        
    def forward(self, x, x_map):
        b, seq_len, h, w = x.size()
        x = x.reshape(b, seq_len, 1, IMG_SIZE, IMG_SIZE)
        x_map = x_map.reshape(b, 1, IMG_SIZE, IMG_SIZE)
        # find size of different input dimensions
        b, seq_len, c, h, w = x.size()
        
        # encode: 
        # initialize hidden states
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t in range(seq_len): 
            x_in = x[:,t]
            h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                              cur_state=[h_enc, enc_state])
        return h_enc,x_map
class RVAE(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(RVAE, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.z_w = int(np.sqrt(latent_dim//2))

        # Constants
        num_hiddens = 64
        num_residual_hiddens = 64 
        num_residual_layers = 2
        embedding_dim = 2 
    
        # prediction encoder:
        
        self._encoder = VAE_Encoder(num_hiddens//4 + self.input_channels,)

        # decoder:
        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=embedding_dim, 
                                    out_channels=num_hiddens,
                                    kernel_size=1, 
                                    stride=1)
        self._decoder = Decoder(self.output_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        

    def vae_reparameterize(self, z_mu, z_log_sd):
        """
        :param mu: mean from the encoder's latent space
        :param log_sd: log standard deviation from the encoder's latent space
        :output: reparameterized latent variable z, Monte carlo KL divergence
        """
        # reshape:
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)
        # define the z probabilities (in this case Normal for both)
        # p(z): N(z|0,I)
        pz = torch.distributions.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_log_sd))
        # q(z|x,phi): N(z|mu, z_var)
        qz_x = torch.distributions.Normal(loc=z_mu, scale=torch.exp(z_log_sd))

        # repameterization trick: z = z_mu + xi (*) z_log_var, xi~N(xi|0,I)
        z = qz_x.rsample()
        # Monte Carlo KL divergence: MCKL(p(z)||q(z|x,phi)) = log(p(z)) - log(q(z|x,phi))
        # sum over weight dim, leaves the batch dim 
        kl_divergence = (pz.log_prob(z) - qz_x.log_prob(z)).sum(dim=1)
        kl_loss = -kl_divergence.mean()

        return z, kl_loss 
    
    
    def forward(self, h_enc, x_map):
        enc_in = torch.cat([h_enc, x_map], dim=1)  # concatenate along channel axis
        z_mu, z_log_sd = self._encoder(enc_in)

        # get the latent vector through reparameterization:
        z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)
    
        # decode:
        # reshape:
        z = z.reshape(-1, 2, self.z_w, self.z_w)
        x_d = self._decoder_z_mu(z)
        prediction = self._decoder(x_d)

        return prediction, kl_loss

    
# end of file
class RVAEP_Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(RVAEP_Encoder, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.z_w = int(np.sqrt(latent_dim//2))
        self.lidar = Lidar2d(np.zeros((64,64)), 90, 32, 0.3125)
        # Constants
        num_hiddens = 64 #64
        num_residual_hiddens = 64 #64
        num_residual_layers = 2
        embedding_dim = 2 
        self.dropout_rate = 0.5
        # prediction encoder:
        self._convlstm = ConvLSTMCell(input_dim=self.input_channels,
                                    hidden_dim=num_hiddens//4,
                                    kernel_size=(3, 3),
                                    bias=True)
        #self._encoder = VAE_Encoder(num_hiddens//4 + self.input_channels,)
        self._encoder = VAE_Encoder(3*(num_hiddens//4 + self.input_channels),)
        # decoder:
        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=embedding_dim, 
                                    out_channels=num_hiddens,
                                    kernel_size=1, 
                                    stride=1)
        self._decoder = Decoder(self.output_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        self.conv=nn.Sequential(nn.Conv2d(in_channels=(num_hiddens//4 + self.input_channels), 
                                    out_channels=self.output_channels,
                                    kernel_size=1, 
                                    stride=1),
                                    nn.Sigmoid())
        self.fc=nn.Sequential(nn.Linear((num_hiddens//4 + self.input_channels)*32*32, 32*32),nn.Sigmoid())
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
    def vae_reparameterize(self, z_mu, z_log_sd):
        """
        :param mu: mean from the encoder's latent space
        :param log_sd: log standard deviation from the encoder's latent space
        :output: reparameterized latent variable z, Monte carlo KL divergence
        """
        # reshape:
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)
        # define the z probabilities (in this case Normal for both)
        # p(z): N(z|0,I)
        pz = torch.distributions.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_log_sd))
        # q(z|x,phi): N(z|mu, z_var)
        qz_x = torch.distributions.Normal(loc=z_mu, scale=torch.exp(z_log_sd))

        # repameterization trick: z = z_mu + xi (*) z_log_var, xi~N(xi|0,I)
        z = qz_x.rsample()
        # Monte Carlo KL divergence: MCKL(p(z)||q(z|x,phi)) = log(p(z)) - log(q(z|x,phi))
        # sum over weight dim, leaves the batch dim 
        kl_divergence = (pz.log_prob(z) - qz_x.log_prob(z)).sum(dim=1)
        kl_loss = -kl_divergence.mean()

        return z, kl_loss 
    
    def forward(self, x_all, x_map_all,distance_check,robot_index):
        
        """
        Forward pass `input_img` through the network
        """
        # reconstruction: 
        # encode:
        # input reshape:
        distance_check=distance_check.reshape(distance_check.size(0),-1) #b 3
        robot_num,b,seq_len,h, w = x_all.size()
        x_all = x_all.reshape(robot_num,b, seq_len, 1, IMG_SIZE, IMG_SIZE)
        x_map_all = x_map_all.reshape(robot_num,b, 1, IMG_SIZE, IMG_SIZE)
        # find size of different input dimensions
        robot_num,b, seq_len, c, h, w = x_all.size()
        
        # encode: 
        # initialize hidden states
        x=x_all[robot_index]
        
        x_map=x_map_all[robot_index]
        
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t in range(seq_len): 
            x_in = x[:,t]
            h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                              cur_state=[h_enc, enc_state])
        
        enc_in_list=[]
        #mask=distance_check[:,robot_index].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,IMG_SIZE,IMG_SIZE)
        
        enc_in_ego = torch.cat([h_enc, x_map], dim=1)  # 64 17 32 32
        enc_in_list.append(enc_in_ego)
        for r in range(robot_num):
            if r!=robot_index:
                    x=x_all[r]
                    x_map=x_map_all[r]
                    h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
                    for t in range(seq_len):
                        x_in = x[:,t]
                        h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                                        cur_state=[h_enc, enc_state])
                    mask=distance_check[:,r].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,IMG_SIZE,IMG_SIZE)
                    enc_in_r = torch.cat([h_enc, x_map], dim=1) * mask
                    # print(enc_in_r.shape)
                    # print(enc_in_r)
                    enc_in_list.append(enc_in_r)
        combined_enc_in=torch.cat(enc_in_list,dim=1)
        #enc_in=torch.cat([h_enc, x_map], dim=1)
        #return prediction,torch.tensor(0.0).to(device)
        
        z_mu, z_log_sd = self._encoder(combined_enc_in)

        # get the latent vector through reparameterization:
        z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)
        
        # decode:
        # reshape:
        z = z.reshape(-1, 2, self.z_w, self.z_w)
        #print(distance_check[:,robot_index].shape) #b
        
        x_d = self._decoder_z_mu(z)
        # #x_d = self.dropout(x_d)
        # for r in range(robot_num):
        #     if r!=robot_index:
                
        #             x=x_all[r]
        #             x_map=x_map_all[r]
        #             h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        #             for t in range(seq_len):
        #                 x_in = x[:,t]
        #                 h_enc, enc_state = self._convlstm(input_tensor=x_in,
        #                                                 cur_state=[h_enc, enc_state])
        #             enc_in = torch.cat([h_enc, x_map,distance_check[]], dim=1)
                    
        #             z_mu, z_log_sd = self._encoder(enc_in)
        #             z, _ = self.vae_reparameterize(z_mu, z_log_sd)
        #             z = z.reshape(-1, 2, self.z_w, self.z_w)
        #             x_d_r = self._decoder_z_mu(z)
        #             x_d[distance_check[:,r],:,:,:]+=x_d_r[distance_check[:,r],:,:,:]
        #         # for bz in range(b):
        #         #     if distance_check[bz,r]:
        #         #        x_d[bz]+=x_d_r[bz] 
                
                
        prediction = self._decoder(x_d)
        
        
        return prediction, kl_loss


class RVAEP_Decoder(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(RVAEP_Decoder, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.z_w = int(np.sqrt(latent_dim//2))
        self.lidar = Lidar2d(np.zeros((64,64)), 90, 32, 0.3125)
        # Constants
        num_hiddens = 64 #64
        num_residual_hiddens = 64 #64
        num_residual_layers = 2
        embedding_dim = 2 
        self.dropout_rate = 0.5
        

        # decoder:
        self._batch_norm = nn.BatchNorm2d(num_hiddens)
        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=embedding_dim, 
                                    out_channels=num_hiddens,
                                    kernel_size=1, 
                                    stride=1)
        self._decoder = Decoder(self.output_channels,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        

    def vae_reparameterize(self, z_mu, z_log_sd):
        """
        :param mu: mean from the encoder's latent space
        :param log_sd: log standard deviation from the encoder's latent space
        :output: reparameterized latent variable z, Monte carlo KL divergence
        """
        # reshape:
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)
        # define the z probabilities (in this case Normal for both)
        # p(z): N(z|0,I)
        pz = torch.distributions.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_log_sd))
        # q(z|x,phi): N(z|mu, z_var)
        qz_x = torch.distributions.Normal(loc=z_mu, scale=torch.exp(z_log_sd))

        # repameterization trick: z = z_mu + xi (*) z_log_var, xi~N(xi|0,I)
        z = qz_x.rsample()
        # Monte Carlo KL divergence: MCKL(p(z)||q(z|x,phi)) = log(p(z)) - log(q(z|x,phi))
        # sum over weight dim, leaves the batch dim 
        kl_divergence = (pz.log_prob(z) - qz_x.log_prob(z)).sum(dim=1)
        kl_loss = -kl_divergence.mean()

        return z, kl_loss 
    
    def forward(self, z):
        
        
        
        # decode:
        # reshape:
        z = z.reshape(-1, 2, self.z_w, self.z_w)

        x_d = self._decoder_z_mu(z)
        #x_d = self.dropout(x_d)
        #x_d = self._batch_norm(x_d)
        prediction = self._decoder(x_d)
        
        
        return prediction
