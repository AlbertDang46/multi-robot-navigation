#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/train.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#  python train.py mdir trian_data val_data
#
# arguments:
#  mdir: the directory where the output model is stored
#  trian_data: the directory of training data
#  val_data: the directory of valiation data
#
# This script trains a SOGMP++ model
#------------------------------------------------------------------------------

# import pytorch modules
#
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import sys
sys.path.append('/home/liyiping/dev/ogm_pred/ogm_sogmp_trans_pos_delay/CrowdNav_Prediction_AttnGraph')

# visualize:
from tensorboardX import SummaryWriter
import numpy as np
import wandb
#wandb.init(project="ogm_pred_dataset")

# import the model and all of its variables/functions
#
from model import *
from local_occ_grid_map import LocalMap

# import modules
#
import sys
import os


#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
model_dir = './model/model.pth'  # the path of model storage 
NUM_ARGS = 3
NUM_EPOCHS = 50 #100
BATCH_SIZE = 64 #512 #64
LEARNING_RATE = "lr"
BETAS = "betas"
EPS = "eps"
WEIGHT_DECAY = "weight_decay"

# Constants
NUM_INPUT_CHANNELS = 1
NUM_LATENT_DIM = 128 # 16*16*2 
NUM_OUTPUT_CHANNELS = 1
BETA = 0.01

# Init map parameters
P_prior = 0.5	# Prior occupancy probability
P_occ = 0.7	    # Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 
MAP_X_LIMIT = [0, 6.4]      # Map limits on the x-axis
MAP_Y_LIMIT = [-3.2, 3.2]   # Map limits on the y-axis
RESOLUTION = 0.2       # Grid resolution in [m]'
TRESHOLD_P_OCC = 0.8    # Occupancy threshold

# for reproducibility, we seed the rng
#
set_seed(SEED1)       

# adjust_learning_rate
#　
def adjust_learning_rate(optimizer, epoch):
    lr = 1e-4
    if epoch > 30000:
        lr = 3e-4
    if epoch > 50000:
        lr = 2e-5
    if epoch > 48000:
       # lr = 5e-8
       lr = lr * (0.1 ** (epoch // 110000))
    #  if epoch > 8300:
    #      lr = 1e-9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def plot_ogm(ogm, filename):
    plt.figure(figsize=(6,6))
    plt.imshow(ogm[0].detach().cpu().numpy(), cmap='gray')  # Assuming the OGM is on GPU and single-channel
    plt.colorbar()
    plt.savefig(filename)  # Saves the image to a file
    print(f"Saved {filename}")
    plt.show()
    plt.close()

# train function:
def get_transform_coordinate(pos,current_pos):
    
    if pos.dim() == 3:
        pos_=pos.unsqueeze(1)
    else:
        pos_=pos
    if current_pos.dim() == 3:
        current_pos_=current_pos.unsqueeze(1)
    else:
        current_pos_=current_pos
        
    dx = pos_[:, :, :,0] - current_pos_[:, :, :,0]
    dy = pos_[:, :, :,1] - current_pos_[:, :, :,1]
    
    th = current_pos_[:, :, :,2]
    x_odom = torch.cos(th) * dx + torch.sin(th) * dy
    y_odom = torch.sin(-th) * dx + torch.cos(th) * dy
    theta_odom = pos_[:, :, :,2] - th
    
    return x_odom, y_odom, theta_odom


def convert_lidar_to_ogm(lidar_data,map_size):
    
    if lidar_data.dim() > 4:
        lidar_data=lidar_data[:,:,0,:,:]
    batch_size,seq_len,num_ray,_= lidar_data.shape
    
    cell_length=0.3125
    center_index = map_size // 2
    local_ogm = torch.full((batch_size,seq_len,2, map_size, map_size), -1, dtype=torch.float32, device=lidar_data.device)
    # Pre-calculate the angles for all rays, this remains constant across batches and sequences
    angles = torch.linspace(0, 2 * torch.pi, num_ray, device=device)
    distances = copy.deepcopy(lidar_data[..., 0])  # Shape: [batch_size, seq_len, num_ray]
    labels = lidar_data[..., 1]     # Shape: [batch_size, seq_len, num_ray]
    invalid_mask=(labels==0).reshape(batch_size,seq_len,num_ray)
    #distances[invalid_mask]=100
    # Calculate x and y indices for all batches and sequences
    x_indices = ((distances * torch.cos(angles)) // cell_length).long() + center_index
    
    y_indices = ((distances * torch.sin(angles))// cell_length).long() + center_index
    
    # Ensure indices are within the map bounds
    mask = (x_indices >= 0) & (x_indices < map_size) & (y_indices >= 0) & (y_indices < map_size)
    
    # Apply mask
    x_indices = x_indices[mask]
    y_indices = y_indices[mask]

    batch_indices, seq_indices, ray_indices = mask.nonzero(as_tuple=True)
    valid_labels = labels[mask]

    # Set cells as occupied
    local_ogm[batch_indices.long(), seq_indices.long(), 0, x_indices.long(), y_indices.long()] = 1
    local_ogm[batch_indices.long(), seq_indices.long(), 1, x_indices.long(), y_indices.long()] = valid_labels
    local_ogm[batch_indices.long(), seq_indices.long(), 0, 15, 15]=1
    local_ogm[batch_indices.long(), seq_indices.long(), 1, 15, 15]=3
   
    # Set unvisited cells to 0 (free space)
    local_ogm[local_ogm == -1] = 0
    
    # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
    static_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)
    dynamic_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)

    # Extract the two channels from the input tensor
    channel_0 = local_ogm[:,:,0,:,:]  # Shape: (batch_size, 32, 32)
    channel_1 = local_ogm[:,:,1,:,:]  # Shape: (batch_size, 32, 32)

    static_obst = channel_0 * (channel_1 == 1)
    dynamic_obst = channel_0 * (channel_1 >= 2)

    
    return static_obst, dynamic_obst
def transform_ogm(lidar_data, pos,x_odom, y_odom, theta_odom, robot_index,map_size=32):
        # Calculate initial angles from past_vel_pos
        if pos.dim() == 3:
            pos_=pos.unsqueeze(1)
        else:
            pos_=pos
        if lidar_data.dim() == 4:
            lidar_data_=lidar_data.unsqueeze(1) # b seq_l robot_num 90 2
        else:
            lidar_data_=lidar_data
        
        batch_size,seq_len,robot_num,num_ray,_= lidar_data_.shape
        #no fusion
        
        angles = torch.linspace(0,2*np.pi,num_ray).unsqueeze(0).unsqueeze(1).repeat(batch_size,seq_len,1).to(lidar_data.device)
        distances = lidar_data_[:,:,robot_index,:,0]
        
        theta_odom=theta_odom.unsqueeze(-1).repeat(1,1,1,num_ray)
        angles = angles + theta_odom[:,:,robot_index]  # Correct for current orientation
        
        distance_x = distances * torch.cos(angles)
        distance_y = distances * torch.sin(angles)

        # Apply translations
    
        x_odom_ = x_odom[:,:,robot_index].unsqueeze(-1).repeat(1,1,1,num_ray)
        y_odom_ = y_odom[:,:,robot_index].unsqueeze(-1).repeat(1,1,1,num_ray)
        distances_x = distance_x + x_odom_
        distances_y = distance_y + y_odom_
        
        #v2
        local_ogm = torch.full((batch_size,seq_len,2, map_size, map_size), -1, dtype=torch.float32, device=lidar_data.device)

        cell_length=0.3125
        center_index = map_size // 2
        labels = lidar_data_[:,:,robot_index,:,1]    # Shape: [batch_size, seq_len, num_ray]
        
        #distances[invalid_mask]=100
        # Calculate x and y indices for all batches and sequences
        x_indices = (distances_x// cell_length).long() + center_index
        
        y_indices = (distances_y/ cell_length).long() + center_index
        
        # Ensure indices are within the map bounds
        mask = (x_indices >= 0) & (x_indices < map_size) & (y_indices >= 0) & (y_indices < map_size)
        
        # Apply mask
        x_indices = x_indices[mask]
        y_indices = y_indices[mask]

        _,batch_indices, seq_indices, ray_indices = mask.nonzero(as_tuple=True)
        valid_labels = labels[mask[0]]

        # Set cells as occupied
        local_ogm[batch_indices.long(), seq_indices.long(), 0, x_indices.long(), y_indices.long()] = 1
        local_ogm[batch_indices.long(), seq_indices.long(), 1, x_indices.long(), y_indices.long()] = valid_labels
        local_ogm[:, :, 0, center_index, center_index]=1
        local_ogm[:, :, 1, center_index, center_index]=3
    
        # Set unvisited cells to 0 (free space)
        local_ogm[local_ogm == -1] = 0
        
        # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
        static_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)
        dynamic_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)

        # Extract the two channels from the input tensor
        channel_0 = local_ogm[:,:,0,:,:]  # Shape: (batch_size, 32, 32)
        channel_1 = local_ogm[:,:,1,:,:]  # Shape: (batch_size, 32, 32)

        static_obst = channel_0 * (channel_1 == 1)
        dynamic_obst = channel_0 * (channel_1 >= 2)

        return static_obst,dynamic_obst


def transform_ogm_early_fusion(lidar_data, pos,x_odom, y_odom, theta_odom, robot_index,map_size=32):
        
        # Calculate initial angles from past_vel_pos
        if pos.dim() == 3:
            pos_=pos.unsqueeze(1)
        else:
            pos_=pos
        if lidar_data.dim() == 4:
            lidar_data_=lidar_data.unsqueeze(1)
        else:
            lidar_data_=lidar_data
        
        batch_size,seq_len,robot_num,num_ray,_= lidar_data_.shape

        #ego
        
        angles = torch.linspace(0,2*np.pi,num_ray).unsqueeze(0).unsqueeze(1).repeat(batch_size,seq_len,1).to(lidar_data.device)
        
        theta_odom=theta_odom.unsqueeze(-1).repeat(1,1,1,num_ray)
        x_odom=x_odom.unsqueeze(-1).repeat(1,1,1,num_ray)
        y_odom=y_odom.unsqueeze(-1).repeat(1,1,1,num_ray)
        angles = angles + theta_odom[:,:,robot_index]
        distances = lidar_data_[:,:,robot_index,:,0]
        distances_x_ego = distances * torch.cos(angles) + x_odom[:,:,robot_index]
        distances_y_ego = distances * torch.sin(angles) + y_odom[:,:,robot_index]
        
        local_ogm = torch.full((batch_size,seq_len,2, map_size, map_size), -1, dtype=torch.float32, device=lidar_data.device)

        cell_length=0.3125
        center_index = map_size // 2
        labels = lidar_data_[:,:,robot_index,:,1]    # Shape: [batch_size, seq_len, num_ray 2]
        
        #distances[invalid_mask]=100
        # Calculate x and y indices for all batches and sequences
        x_indices = (distances_x_ego// cell_length).long() + center_index
        y_indices = (distances_y_ego// cell_length).long() + center_index
        
        # Ensure indices are within the map bounds
        mask = (x_indices >= 0) & (x_indices < map_size) & (y_indices >= 0) & (y_indices < map_size)
        
        # Apply mask
        x_indices = x_indices[mask]
        y_indices = y_indices[mask]
        valid_labels = labels[mask]
        
        batch_indices, seq_indices, ray_indices = mask.nonzero(as_tuple=True)
        
        # Set cells as occupied
        local_ogm[batch_indices.long(), seq_indices.long(), 0, x_indices.long(), y_indices.long()] = 1
        local_ogm[batch_indices.long(), seq_indices.long(), 1, x_indices.long(), y_indices.long()] = valid_labels
        local_ogm[:, :, 0, center_index, center_index]=1
        local_ogm[:, :, 1, center_index, center_index]=3
    
        # Set unvisited cells to 0 (free space)
        local_ogm[local_ogm == -1] = 0
        
        for r in range(robot_num):
            if r==robot_index:
                continue
            # transform past to current
            angles = torch.linspace(0,2*np.pi,num_ray).unsqueeze(0).unsqueeze(1).repeat(batch_size,seq_len,1).to(lidar_data.device)
            #theta_odom=theta_odom.unsqueeze(-1).repeat(1,1,1,num_ray)
            angles = angles + theta_odom[:,:,r]
            distances = lidar_data_[:,:,r,:,0]
            distances_x = distances * torch.cos(angles) + x_odom[:,:,r]
            distances_y = distances * torch.sin(angles)+ y_odom[:,:,r]

            dx = pos_[:,:,robot_index,0] - pos_[:,:,r,0]
            dy = pos_[:,:,robot_index,1] - pos_[:,:,r,1]
            distance_check = torch.sqrt(dx**2 + dy**2) < 10 # batchsize seq_len
            
            distance_check=distance_check.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,map_size,map_size)

            cell_length=0.3125
            center_index = map_size // 2
            labels = lidar_data_[:,:,r,:,1]    # Shape: [batch_size, seq_len, num_ray]
            # Calculate x and y indices for all batches and sequences
            x_indices = (distances_x// cell_length).long() + center_index
            
            y_indices = (distances_y// cell_length).long() + center_index
            # Ensure indices are within the map bounds
            
            current_labels = lidar_data_[:,:,robot_index,:,1]
            
            # Determine where the grid cells are unvisited or have a higher label value than the new data
            final_mask = (x_indices >= 0) & (x_indices < map_size) & \
                 (y_indices >= 0) & (y_indices < map_size)&\
                 ((torch.logical_and(current_labels == 0, labels > 0)) | (current_labels > labels))
 
            # Get indices where the final_mask is True
            final_batch_indices, final_seq_indices, final_ray_indices = final_mask.nonzero(as_tuple=True)
            
            # Extract the indices that are valid and within bounds
            
            final_x_indices = x_indices[final_batch_indices,final_seq_indices,final_ray_indices]
            final_y_indices = y_indices[final_batch_indices,final_seq_indices,final_ray_indices]
            
            # Update local_ogm at the indices specified by the mask
            merged_ogm=local_ogm.clone()
            #plot_ogm(merged_ogm[0,-1,0,:,:].unsqueeze(0),'merged_ogm.png')
            merged_ogm[final_batch_indices, final_seq_indices, 0, final_x_indices, final_y_indices] = 1
            #plot_ogm(merged_ogm[0,-1,0,:,:].unsqueeze(0),'merged_ogm.png')
            merged_ogm[final_batch_indices, final_seq_indices, 1, final_x_indices, final_y_indices] = valid_labels[final_ray_indices]
            
            new_merged_ogm=torch.where(distance_check,merged_ogm,local_ogm)

        # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
        static_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)
        dynamic_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)

        # Extract the two channels from the input tensor
        channel_0 = new_merged_ogm[:,:,0,:,:]  # Shape: (batch_size, 32, 32)
        channel_1 = new_merged_ogm[:,:,1,:,:]  # Shape: (batch_size, 32, 32)

        static_obst = channel_0 * (channel_1 == 1)
        dynamic_obst = channel_0 * (channel_1 >= 2)
        
        
        return static_obst,dynamic_obst

def train(model, dataloader, dataset, device, optimizer, criterion, epoch, epochs,fusion):
    
    
    model.train()
    # for each batch in increments of batch size:
    running_loss = 0.0
    # kl_divergence:
    kl_avg_loss = 0.0
    # CE loss:
    ce_avg_loss = 0.0
    total_wmse = 0
    total_ssim = 0  
    criterion_wmse = WeightedMSELoss()

    counter = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(dataset)/dataloader.batch_size)
    
    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
    
        counter += 1
        # collect the samples as a batch:
        scans = batch['scan']
        scans = scans.to(device) #  b seq_len robot_num 90 2
        positions = batch['position'] 
        positions = positions.to(device) # b seq_len robot 
        targets = batch['target']
        targets = targets.to(device)  # b seq_len robot_num FUTURE_STEP 90 2
        lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
        lidar_t=targets[:,SEQ_LEN-1,:,:,:,:] #b robot_num FUTURE_STEP 90 2
        
        
        batch_size,_, robot_num,_, _ = scans.shape
        
        
        for r in range(robot_num):

            assert fusion in ['no','early','middle','late']
            optimizer.zero_grad()
            prediction_list=[]
            mask_list=[]
            if fusion=='no':
                #no fusion
                pos = positions[:,:SEQ_LEN,r,:].unsqueeze(2).repeat(1,1,robot_num,1)
                current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
                static_obst, dynamic_obst=convert_lidar_to_ogm(lidar,map_size=32) #visualize ogm before coordinate transform
                # for s in range(SEQ_LEN):
                #     plot_ogm(static_obst[0,s,:,:].unsqueeze(0),f'static_obst_{s}.png')
                #     #plot_ogm(dynamic_obst[0,s,:,:].unsqueeze(0),f'dynamic_obst_{s}.png')
                static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                # for s in range(SEQ_LEN):
                #     plot_ogm(static_obst[0,s,:,:].unsqueeze(0),f'static_obst_{s}.png')
                #     #plot_ogm(dynamic_obst[0,s,:,:].unsqueeze(0),f'dynamic_obst_{s}.png')
                for t in range(FUTURE_STEP): 
                    #no fusion
                    ego_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                    current_pos = positions[:,SEQ_LEN-1,:,:]
                    x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,ego_pos)
                    static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                    mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                    mask_list.append(mask_map)
                    #plot_ogm(static_obst_t[0],f'mask_{r}_{t}.png')
                    past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:] # b _ 32 32
                    current_static_obst=static_obst[:,-1,:,:]
                    dynamic_obst=past_dogm
                
                    if len(prediction_list)>0:
                        pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                        pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                        dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                    
                    prediction, kl_loss = model(dynamic_obst,current_static_obst)
                    #plot_ogm(prediction[0],f'pred_{r}_{t}.png')
                    prediction_list.append(prediction)
            if fusion=='early':
    
                pos = positions[:,:SEQ_LEN,:,:]
                ego_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,ego_pos)
                # static_obst, dynamic_obst=convert_lidar_to_ogm(lidar,map_size=32) #visualize ogm before coordinate transform
                # for s in range(SEQ_LEN):
                #     plot_ogm(static_obst[0,s,:,:].unsqueeze(0),f'static_obst_{r}_{s}.png')
                #     plot_ogm(dynamic_obst[0,s,:,:].unsqueeze(0),f'dynamic_obst_{r}_{s}.png')
                static_obst, dynamic_obst=transform_ogm_early_fusion(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                for s in range(SEQ_LEN):
                    plot_ogm(static_obst[0,s,:,:].unsqueeze(0),f'static_obst_{r}_{s}.png')
                    plot_ogm(dynamic_obst[0,s,:,:].unsqueeze(0),f'dynamic_obst_{r}_{s}.png')
                for t in range(FUTURE_STEP): 
                    
                    current_pos = positions[:,SEQ_LEN-1,:,:]
                    ego_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                    x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,ego_pos)
                    static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                    
                    mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                    plot_ogm(static_obst_t[0],f'smask_{r}_{t}.png')
                    plot_ogm(dynamic_obst_t[0],f'dmask_{r}_{t}.png')
                    plot_ogm(mask_map[0],f'mask_{r}_{t}.png')
                    mask_list.append(mask_map)
                    
                    past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:] # b _ 32 32
                    current_static_obst=static_obst[:,-1,:,:]
                    dynamic_obst=past_dogm
                    
                    if len(prediction_list)>0:
                        pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                        pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                        dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                    
                    prediction, kl_loss = model(dynamic_obst,current_static_obst)
                    plot_ogm(prediction[0],f'pred_{r}_{t}.png')
                    prediction_list.append(prediction)
            if fusion=='middle':
                dynamic_obst_list=[]
                current_static_obst_list=[]
                for rv in range(robot_num):
                    current_pos=positions[:,SEQ_LEN-1,rv,:].unsqueeze(1).repeat(1,robot_num,1)
                    pos=positions[:,:SEQ_LEN,rv,:].unsqueeze(2).repeat(1,1,robot_num,1)
                    x_odom, y_odom, theta_odom= get_transform_coordinate(pos,current_pos)
                    lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
                    static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rv,map_size=32)
                    # if rv==0:
                    #     plot_ogm(static_obst[0,-1,:,:].unsqueeze(0),f'static_obst_{rv}.png')
                    #     plot_ogm(dynamic_obst[0,-1,:,:].unsqueeze(0),f'dynamic_obst_{rv}.png')
                    dynamic_obst_list.append(dynamic_obst)
                    current_static_obst_list.append(static_obst[:,-1,:,:])
                dynamic_obst_tensor=torch.stack(dynamic_obst_list,dim=0).squeeze(1) #rob_num b seq_l 32 32
                current_static_obst_tensor=torch.stack(current_static_obst_list,dim=0).squeeze(1) #rob_num b seq_l 32 32
                for t in range(FUTURE_STEP): 
                   
                    current_pos = positions[:,SEQ_LEN-1,:,:]
                    ego_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                    x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,ego_pos)
                    static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)

                    mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                    mask_list.append(mask_map)
                    

                    #middle fusion 
                    # b seq 32 32
                    past_dogm=dynamic_obst_tensor[r][:,len(prediction_list):SEQ_LEN,:,:]
                    
                    if len(prediction_list)>0:
                        pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                        pred_dogm=torch.abs(pred_ogm-current_static_obst_tensor[r].unsqueeze(1))
                        dynamic_obst_ego=torch.cat((past_dogm,pred_dogm),dim=1)
                        dynamic_obst_tensor[r]=dynamic_obst_ego
                    else:
                        dynamic_obst_tensor[r]=past_dogm
                    pos=positions[:,SEQ_LEN-1,:,:]#b robot_num 3
                    prediction, kl_loss = model(dynamic_obst_tensor,current_static_obst_tensor,pos,r)
                    
                    prediction_list.append(prediction)
                
            if fusion=='late':
                pass
            
            prediction_tensor=torch.stack(prediction_list,dim=1).squeeze(2)

            mask_tensor=torch.stack(mask_list,dim=1).squeeze(2)
            
            
            # for t in range(FUTURE_STEP):
            #     #plot_ogm(current_static_obst[0].unsqueeze(0),f'current_static_ogm_{t}.png')
            #     #plot_ogm(prediction_tensor[0][t].unsqueeze(0),'pred.png')
            #     plot_ogm(mask_tensor[0][t].unsqueeze(0),f'mask_{t}.png')   
            ce_loss = criterion(prediction_tensor, mask_tensor).div(batch_size)
            wmse = criterion_wmse(prediction_tensor, mask_tensor, calculate_weights(mask_tensor))
            loss = ce_loss+ BETA*kl_loss
            loss.backward(torch.ones_like(loss))
            
            optimizer.step()
            total_wmse += wmse.item()
            ssim_batch = calculate_ssim(prediction_tensor, mask_tensor)
            total_ssim += ssim_batch.item()
            # get the loss:
            # multiple GPUs:
            if torch.cuda.device_count() > 1:
                loss = loss.mean()  
                ce_loss = ce_loss.mean()
                kl_loss = kl_loss.mean()

            running_loss += loss.item()
            # kl_divergence:
            kl_avg_loss += kl_loss.item()
            # CE loss:
            ce_avg_loss += ce_loss.item()
            
            
        
        # display informational message:
        if(i % 128 == 0):
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
                    .format(epoch, epochs, i + 1, num_batches, loss.item(), ce_loss.item(), kl_loss.item()))
    train_loss = running_loss / counter 
    train_kl_loss = kl_avg_loss / counter
    train_ce_loss = ce_avg_loss / counter
    avg_wmse = total_wmse / counter
    avg_ssim = total_ssim / counter

    return train_loss, train_kl_loss, train_ce_loss, avg_wmse, avg_ssim


def calculate_occupied_grid_rate(grid_map):
    # Assuming grid_map is a binary map (0 for free space, 1 for occupied)
    occupied_cells = torch.sum(grid_map > 0).item()
    total_cells = grid_map.numel()
    occupied_rate = occupied_cells / total_cells
    return occupied_rate





def validate(model, dataloader, dataset, device, criterion,fusion): 
    flag=False
    model.eval()
    running_loss = 0.0
    kl_avg_loss = 0.0
    ce_avg_loss = 0.0
    total_wmse = 0
    total_ssim = 0  # Initialize total SSIM
    criterion_wmse = WeightedMSELoss()

    counter = 0
    num_batches = int(len(dataset) / dataloader.batch_size)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
    
            counter += 1
            # collect the samples as a batch:
            scans = batch['scan']
            scans = scans.to(device)
            positions = batch['position']
            positions = positions.to(device)
            targets = batch['target']
            targets = targets.to(device) 
            
            
            batch_size, _, robot_num,_, _ = scans.shape
            
            
            lidar=scans[:,:SEQ_LEN,:,:,:]
            lidar_t=targets[:,SEQ_LEN-1,:,:,:,:]
        
        
            for r in range(robot_num):

                assert fusion in ['no','early','middle','late']
                
                prediction_list=[]
                mask_list=[]
                if fusion=='no':
                    #no fusion
                    pos = positions[:,:SEQ_LEN,r,:].unsqueeze(2).repeat(1,1,robot_num,1)
                    current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                    x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
                    #static_obst, dynamic_obst=convert_lidar_to_ogm(lidar,map_size=32) #visualize ogm before coordinate transform
                    # for s in range(SEQ_LEN):
                    #     plot_ogm(static_obst[0,s,:,:].unsqueeze(0),f'static_obst_{s}.png')
                    #     plot_ogm(dynamic_obst[0,s,:,:].unsqueeze(0),f'dynamic_obst_{s}.png')
                    static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                    for t in range(FUTURE_STEP): 
                        #no fusion
                        current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                        x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,current_pos)
                        static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                        mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                        mask_list.append(mask_map)
                        
                        past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:] # b _ 32 32
                        current_static_obst=static_obst[:,-1,:,:]
                        dynamic_obst=past_dogm
                    
                        if len(prediction_list)>0:
                            pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                            pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                            dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                        
                        prediction, kl_loss = model(dynamic_obst,current_static_obst)
                        #plot_ogm(prediction[0],f'pred_{r}_{t}.png')
                        prediction_list.append(prediction)
                if fusion=='early':
        
                    pos = positions[:,:SEQ_LEN,:,:]
                    current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                    x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
                    #static_obst, dynamic_obst=transform_ogm_early_fusion(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                    static_obst, dynamic_obst=convert_lidar_to_ogm(lidar,map_size=32) #visualize ogm before coordinate transform
                    # for s in range(SEQ_LEN):
                    #     plot_ogm(static_obst[0,s,:,:].unsqueeze(0),f'static_obst_{r}_{s}.png')
                    #     plot_ogm(dynamic_obst[0,s,:,:].unsqueeze(0),f'dynamic_obst_{r}_{s}.png')
                    static_obst, dynamic_obst=transform_ogm_early_fusion(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                    # for s in range(SEQ_LEN):
                    #     plot_ogm(static_obst[0,s,:,:].unsqueeze(0),f'static_obst_{r}_{s}.png')
                    #     plot_ogm(dynamic_obst[0,s,:,:].unsqueeze(0),f'dynamic_obst_{r}_{s}.png')
                    for t in range(FUTURE_STEP): 
                        
                        current_pos = positions[:,SEQ_LEN-1,:,:]
                        ego_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                        x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,ego_pos)
                        static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                        
                        mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                        mask_list.append(mask_map)
                        # plot_ogm(static_obst_t[0],f'smask_{r}_{t}.png')
                        # plot_ogm(dynamic_obst_t[0],f'dmask_{r}_{t}.png')
                        past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:] # b _ 32 32
                        current_static_obst=static_obst[:,-1,:,:]
                        dynamic_obst=past_dogm
                        if len(prediction_list)>0:
                            pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                            pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                            dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)

                        prediction, kl_loss = model(dynamic_obst,current_static_obst)
                        #plot_ogm(prediction[0],f'pred_{r}_{t}.png')
                        prediction_list.append(prediction)
                if fusion=='middle':
                    dynamic_obst_list=[]
                    current_static_obst_list=[]
                    for rv in range(robot_num):
                        current_pos=positions[:,SEQ_LEN-1,rv,:].unsqueeze(1).repeat(1,robot_num,1)
                        pos=positions[:,:SEQ_LEN,rv,:].unsqueeze(2).repeat(1,1,robot_num,1)
                        x_odom, y_odom, theta_odom= get_transform_coordinate(pos,current_pos)
                        lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
                        static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rv,map_size=32)
                        # if rv==0:
                        #     plot_ogm(static_obst[0,-1,:,:].unsqueeze(0),f'static_obst_{rv}.png')
                        #     plot_ogm(dynamic_obst[0,-1,:,:].unsqueeze(0),f'dynamic_obst_{rv}.png')
                        dynamic_obst_list.append(dynamic_obst)
                        current_static_obst_list.append(static_obst[:,-1,:,:])
                    dynamic_obst_tensor=torch.stack(dynamic_obst_list,dim=0).squeeze(1) #rob_num b seq_l 32 32
                    current_static_obst_tensor=torch.stack(current_static_obst_list,dim=0).squeeze(1) #rob_num b seq_l 32 32
                    for t in range(FUTURE_STEP): 
                    
                        current_pos = positions[:,SEQ_LEN-1,:,:]
                        ego_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                        x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,ego_pos)
                        static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)

                        mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                        mask_list.append(mask_map)
                        

                        #middle fusion 
                        # b seq 32 32
                        past_dogm=dynamic_obst_tensor[r][:,len(prediction_list):SEQ_LEN,:,:]
                        
                        if len(prediction_list)>0:
                            pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                            pred_dogm=torch.abs(pred_ogm-current_static_obst_tensor[r].unsqueeze(1))
                            dynamic_obst_ego=torch.cat((past_dogm,pred_dogm),dim=1)
                            dynamic_obst_tensor[r]=dynamic_obst_ego
                        else:
                            dynamic_obst_tensor[r]=past_dogm
                        pos=positions[:,SEQ_LEN-1,:,:]#b robot_num 3
                        prediction, kl_loss = model(dynamic_obst_tensor,current_static_obst_tensor,pos,r)
                        #plot_ogm(prediction[0],f'pred_{r}_{t}.png')
                        prediction_list.append(prediction)
                    
                if fusion=='late':
                    pass
                prediction_tensor=torch.stack(prediction_list,dim=1).squeeze(2) 
                
                mask_tensor=torch.stack(mask_list,dim=1).squeeze(2)
                ce_loss = criterion(prediction_tensor, mask_tensor).div(batch_size)
                
                loss = ce_loss+ BETA*kl_loss
                wmse = criterion_wmse(prediction_tensor, mask_tensor, calculate_weights(mask_tensor))
                total_wmse += wmse.item()
                ssim_batch = calculate_ssim(prediction_tensor, mask_tensor)
                total_ssim += ssim_batch.item()
                # get the loss:
                # multiple GPUs:
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()  
                    ce_loss = ce_loss.mean()
                    kl_loss = kl_loss.mean()

                running_loss += loss.item()
                # kl_divergence:
                kl_avg_loss += kl_loss.item()
                # CE loss:
                ce_avg_loss += ce_loss.item()
                

    val_loss = running_loss / counter
    val_kl_loss = kl_avg_loss / counter
    val_ce_loss = ce_avg_loss / counter
    avg_wmse = total_wmse / counter
    avg_ssim = total_ssim / counter

    return val_loss, val_kl_loss, val_ce_loss, avg_wmse, avg_ssim


def calculate_ssim(pred_ogm, gt_ogm, C1=1e-4, C2=9e-4):
    """
    Calculate the Structural Similarity Index Measure (SSIM) from predicted and ground truth OGMs.
    
    Parameters:
    pred_ogm (torch.Tensor): Predicted occupancy grid map tensor.
    gt_ogm (torch.Tensor): Ground truth occupancy grid map tensor.
    C1 (float): Constant to avoid instability, default is 1e-4.
    C2 (float): Constant to avoid instability, default is 9e-4.
    
    Returns:
    torch.Tensor: SSIM value.
    """
    mu_pred = torch.mean(pred_ogm)
    mu_gt = torch.mean(gt_ogm)
    
    delta_pred = torch.var(pred_ogm)
    delta_gt = torch.var(gt_ogm)
    
    delta_pred_gt = torch.mean((pred_ogm - mu_pred) * (gt_ogm - mu_gt))
    
    numerator = (2 * mu_pred * mu_gt + C1) * (2 * delta_pred_gt + C2)
    denominator = (mu_pred**2 + mu_gt**2 + C1) * (delta_pred + delta_gt + C2)
    ssim = numerator / denominator
    
    return ssim

class WeightedMSELoss(nn.Module):
    def forward(self, input, target, weights):
        squared_diffs = (input - target) ** 2
        weighted_squared_diffs = weights * squared_diffs
        wmse = weighted_squared_diffs.sum() / weights.sum()
        return wmse

def calculate_weights(ground_truth):
    # Assuming ground_truth is a binary tensor where 1 represents occupied and 0 represents free
    total_cells = ground_truth.numel()
    occupied_count = ground_truth.sum()
    free_count = total_cells - occupied_count

    # Frequencies
    freq_occupied = occupied_count / total_cells
    freq_free = free_count / total_cells

    # Median frequency
    median_freq = torch.median(torch.tensor([freq_occupied, freq_free]))

    # Weights
    weight_occupied = median_freq / freq_occupied if freq_occupied > 0 else 0
    weight_free = median_freq / freq_free if freq_free > 0 else 0

    # Create weight map based on ground truth
    weights = torch.where(ground_truth == 1, weight_occupied, weight_free)
    return weights
#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# function: main
#
# arguments: none
#
# return: none
#
# This method is the main function.
#
def main():
    # # ensure we have the correct amount of arguments:
    # #global cur_batch_win
    # if(len(argv) != NUM_ARGS):
    #     print("usage: python train.py [MDL_PATH] [TRAIN_PATH] [VAL_PATH]")
    #     exit(-1)

    # # define local variables:
    # mdl_path = argv[0]
    # pTrain = argv[1]
    # pDev = argv[2]

    # # get the output directory name:
    # odir = os.path.dirname('new_trained_model')

    # if the odir doesn't exits, we make it:
    if not os.path.exists('new_trained_model'):
        os.makedirs('new_trained_model')

    # set the device to use GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('...Start reading data...')
    
    ### training data ###
    # training set and training data loader
    train_dataset = VaeTestDataset('SOGMP_plus/OGM-datasets/dataset_reset/train', 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, \
                                                   shuffle=True, drop_last=True, pin_memory=True)

    ### validation data ###
    # validation set and validation data loader
    dev_dataset = VaeTestDataset('SOGMP_plus/OGM-datasets/dataset_reset/val', 'val')
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, \
                                                 shuffle=True, drop_last=True, pin_memory=True)

    # # instantiate a model:
    model = RVAEP(input_channels=NUM_INPUT_CHANNELS,
                  latent_dim=NUM_LATENT_DIM,
                  output_channels=NUM_OUTPUT_CHANNELS)
    # #moves the model to device (cpu in our case so no change):

    model.to(device)
    
    # set the adam optimizer parameters:
    opt_params = { LEARNING_RATE: 0.001,
                   BETAS: (.9,0.999),
                   EPS: 1e-08,
                   WEIGHT_DECAY: .001 }
    # set the loss criterion and optimizer:
    criterion = nn.BCELoss(reduction='sum') #, weight=class_weights)
    criterion.to(device)
    # create an optimizer, and pass the model params to it:
    #all_params = list(encoder.parameters())
    optimizer = Adam(model.parameters(), **opt_params)
    
    # get the number of epochs to train on:
    epochs = NUM_EPOCHS

  
    start_epoch = 0
    print('No trained models, restart training')

    checkpoint = torch.load('ef_4FUTURE_40.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # multiple GPUs:
    # if torch.cuda.device_count() > 1:
    #     print("Let's use 2 of total", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model) #, device_ids=[0, 1])
    # # moves the model to device (cpu in our case so no change):
    # model.to(device)

    # tensorboard writer:
    writer = SummaryWriter('runs')

    epoch_num = 0
    for epoch in range(start_epoch+1, epochs):
        # adjust learning rate:
        adjust_learning_rate(optimizer, epoch)
        ################################## Train #####################################
        # for each batch in increments of batch size
        #
        
        
        train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss,wmse_train,ssim_train= train(
            model,train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs,fusion='early'
        )
        valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss,wmse,ssim= validate(
            model, dev_dataloader, dev_dataset, device, criterion, fusion='early'
        )
        
        

        
        
        
        # log the epoch loss
        writer.add_scalar('training loss',
                        train_epoch_loss,
                        epoch)
        writer.add_scalar('training kl loss',
                        train_kl_epoch_loss,
                        epoch)
        writer.add_scalar('training ce loss',
                train_ce_epoch_loss,
                epoch)
        writer.add_scalar('validation loss',
                        valid_epoch_loss,
                        epoch)
        writer.add_scalar('validation kl loss',
                        valid_kl_epoch_loss,
                        epoch)
        writer.add_scalar('validation ce loss',
                        valid_ce_epoch_loss,
                        epoch)

        # print('Train set: Average loss: {:.4f}'.format(train_epoch_loss))
        # print('Validation set: Average loss: {:.4f}, WMSE: {:.4f}, SSIM: {:.4f}'.format(valid_epoch_loss,wmse/3,ssim/3))
         #Log metrics to wandb
        # wandb.log({"Train Loss": train_epoch_loss, "Train KL Loss": train_kl_epoch_loss, "Train CE Loss": train_ce_epoch_loss,"Train_WMSE": wmse_train/3, "Train_SSIM": ssim_train/3,
        #         "Validation Loss": valid_epoch_loss, "Validation KL Loss": valid_kl_epoch_loss, "Validation CE Loss": valid_ce_epoch_loss,
        #         "Val_WMSE": wmse/3, "Val_SSIM": ssim/3})
        #"Train RESET": RESET_COUNT_train,"Val RESET": RESET_COUNT
        
        
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, train WMSE: {:.4f}, train SSIM: {:.4f}, val WMSE: {:.4f}, val SSIM: {:.4f}'.format(
            epoch, epochs, train_epoch_loss, valid_epoch_loss, wmse_train/3,ssim_train/3,wmse/3, ssim/3))
        
        
        
        # save the model:
        if(epoch % 10== 0):
            if torch.cuda.device_count() > 1: # multiple GPUS: 
                state = {'model':model.modules.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            else:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            path='ef_4FUTURE_' + str(epoch) +'.pth'
            torch.save(state, path)
            


        epoch_num = epoch

   

    # exit gracefully
    #

    return True
#
# end of function


# begin gracefully
#
if __name__ == '__main__':
    main()
#
# end of file
