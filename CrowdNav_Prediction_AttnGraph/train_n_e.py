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
wandb.init(project="ogm_pred_dataset")

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
def get_all_transform_coordinate(pos,current_pos):
    
    if pos.dim() == 3:
        pos_=pos.unsqueeze(1)
    else:
        pos_=pos
    if current_pos.dim() == 3:
        current_pos_=current_pos.unsqueeze(1)
    else:
        current_pos_=current_pos
    _,_,robot_num,_= pos_.shape
    x_odom_list=[]
    y_odom_list=[]
    theta_odom_list=[]
    for r in range(robot_num):
        x_odom, y_odom, theta_odom = get_transform_coordinate(pos_,current_pos_[:,:,r,:].unsqueeze(2).repeat(1,1,robot_num,1))
        
        x_odom_list.append(x_odom)
        y_odom_list.append(y_odom)
        theta_odom_list.append(theta_odom)
    x_odom_tensor=torch.stack(x_odom_list,dim=0)
    y_odom_tensor=torch.stack(y_odom_list,dim=0)
    theta_odom_tensor=torch.stack(theta_odom_list,dim=0)
    
    return x_odom_tensor, y_odom_tensor, theta_odom_tensor
# def transform_all_ogm(lidar_data, pos,x_odom, y_odom, theta_odom, robot_index,map_size=32):
        
#         if pos.dim() == 3:
#             pos_=pos.unsqueeze(1)
#         else:
#             pos_=pos
#         if lidar_data.dim() == 4:
#             lidar_data_=lidar_data.unsqueeze(1)
#         else:
#             lidar_data_=lidar_data
        
#         _,_,robot_num,num_ray,_= lidar_data_.shape
#         #print(x_odom.shape, y_odom.shape, theta_odom.shape)
#         static_list=[]
#         dynamic_list=[]
#         for r in range(robot_num):
#             # transform past to current
#             initial_angles = pos_[:,:,r,2]  # Assuming the 4th index is the angle in past_vel_pos
            
#             # Compute Cartesian coordinates from polar coordinates
#             distances = lidar_data_[:,:,r,:,0]
            
#             angles = initial_angles + theta_odom[:,:,r]  # Correct for current orientation
#             angles_ = angles.unsqueeze(-1).repeat(1,1,num_ray)
#             distance_x = distances * torch.cos(angles_)
#             distance_y = distances * torch.sin(angles_)
    
#             # Apply translations
        
#             x_odom_ = x_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
#             y_odom_ = y_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
#             distances_x = distance_x + x_odom_
#             distances_y = distance_y + y_odom_
            
#             # Calculate new distances from transformed coordinates
#             new_distances = torch.sqrt(distances_x**2 + distances_y**2)
            
#             lidar_data_[:,:,r,:,0] = new_distances
#             static_obst, dynamic_obst=convert_lidar_to_ogm(lidar_data_[:,:,r,:,:],map_size)
#             static_list.append(static_obst)
#             dynamic_list.append(dynamic_obst)
#         static_map_tensor=torch.stack(static_list,dim=0)
#         dynamic_map_tensor=torch.stack(dynamic_list,dim=0)
#         return static_map_tensor,dynamic_map_tensor

def convert_lidar_to_ogm(lidar_data,map_size):
    
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
        
        _,_,robot_num,num_ray,_= lidar_data_.shape
        #no fusion
        initial_angles = pos_[:,:,robot_index,2] # b seq_l robot_num 3
        
        
        distances = lidar_data_[:,:,robot_index,:,0]
        
        angles = initial_angles + theta_odom[:,:,robot_index]  # Correct for current orientation
        angles_ = angles.unsqueeze(-1).repeat(1,1,1,num_ray)
        distance_x = distances * torch.cos(angles_)
        distance_y = distances * torch.sin(angles_)

        # Apply translations
    
        x_odom_ = x_odom[:,:,robot_index].unsqueeze(-1).repeat(1,1,1,num_ray)
        y_odom_ = y_odom[:,:,robot_index].unsqueeze(-1).repeat(1,1,1,num_ray)
        distances_x = distance_x + x_odom_
        distances_y = distance_y + y_odom_
        
        # Calculate new distances from transformed coordinates
        new_distances = torch.sqrt(distances_x**2 + distances_y**2)
        
        lidar_data_[:,:,robot_index,:,0] = new_distances
        static_obst, dynamic_obst=convert_lidar_to_ogm(lidar_data_[:,:,robot_index,:,:],map_size)
        
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
        
        
        
        for r in range(robot_num):
            # transform past to current
            initial_angles = pos_[:,:,r,2]  # Assuming the 4th index is the angle in past_vel_pos
            
            # Compute Cartesian coordinates from polar coordinates
            distances = lidar_data_[:,:,r,:,0].clone()
            
            angles = initial_angles + theta_odom[:,:,r]  # Correct for current orientation
            angles_ = angles.unsqueeze(-1).repeat(1,1,num_ray)
            distance_x = distances * torch.cos(angles_)
            distance_y = distances * torch.sin(angles_)

            # Apply translations
        
            x_odom_ = x_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
            y_odom_ = y_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
            distances_x = distance_x + x_odom_
            distances_y = distance_y + y_odom_
            
            
            new_distances = torch.sqrt(distances_x**2 + distances_y**2)
            #print(lidar_data_[0,0,r,:,0])
            #lidar_data_[:,:,r,:,0] = new_distances
            #print(lidar_data_[0,0,r,:,0])
            # static_obst, dynamic_obst=convert_lidar_to_ogm(lidar_data_[:,:,r,:,:],map_size)

            # #plot_ogm(static_obst[0],f'static_obst_{r}.png')
            # static_obst_list.append(static_obst)
            # dynamic_obst_list.append(dynamic_obst)
            
            if r!=robot_index:
                dx = pos_[:,:,robot_index,0] - pos_[:,:,r,0]
                dy = pos_[:,:,robot_index,1] - pos_[:,:,r,1]
                distance_check = torch.sqrt(dx**2 + dy**2) < 10
                
                distance_check = distance_check.unsqueeze(-1).repeat(1,1,num_ray)  
                
                filtered_distances = torch.where(distance_check, new_distances,distances).to(new_distances.device)
                
                lidar_data_[:,:,r,:,0] = filtered_distances

                merged_lidar = lidar_data_[:,:,robot_index,:,:].clone()
                
                lidar2 = lidar_data_[:,:,r,:,:]
                
                mask_new_obstacle = (merged_lidar[:, :,:,1] == 0) & (lidar2[:,:,:, 1] != 0)
                merged_lidar[mask_new_obstacle] = lidar2[mask_new_obstacle]

                mask_both_inf = (merged_lidar[:, :,:, 0] == float('inf')) & (lidar2[:, :,:, 0] == float('inf'))
                merged_lidar[mask_both_inf] = lidar2[mask_both_inf]
                lidar_data_[:,:,robot_index,:,:] = merged_lidar
        # for r in range(robot_num):
        #     for b in range(batch_size):
        #         for s in range(seq_len):
        #             dx = pos_[b,s,robot_index,0] - pos_[b,s,r,0]
        #             dy = pos_[b,s,robot_index,1] - pos_[b,s,r,1]
        #             distance_check = torch.sqrt(dx**2 + dy**2) > 10
        #             if distance_check or r==robot_index:
        #                 continue
        #             static_obst_list[robot_index][b,s]+=static_obst_list[r][b,s]
        #             dynamic_obst_list[robot_index][b,s]+=dynamic_obst_list[r][b,s]
        # static_obst=static_obst_list[robot_index].clamp(0,1)
        # dynamic_obst=dynamic_obst_list[robot_index].clamp(0,1)     
        static_obst, dynamic_obst=convert_lidar_to_ogm(lidar_data_[:,:,robot_index,:,:],map_size)
        
        return static_obst,dynamic_obst
# def train(model,dataloader, dataset, device, optimizer, criterion, epoch, epochs,RESET_COUNT):
    
#     # set model to training mode:seq_len, batch_size, map_size, 
#     flag=False
    
#     model.train()
    
#     # for each batch in increments of batch size:
#     running_loss = 0.0
#     # kl_divergence:
#     kl_avg_loss = 0.0
#     # CE loss:
#     ce_avg_loss = 0.0
#     total_wmse = 0
#     total_ssim = 0  
#     criterion_wmse = WeightedMSELoss()

#     counter = 0
#     # get the number of batches (ceiling of train_data/batch_size):
#     num_batches = int(len(dataset)/dataloader.batch_size)
    
#     for i, batch in tqdm(enumerate(dataloader), total=num_batches):
    
#         counter += 1
#         # collect the samples as a batch:
#         scans = batch['scan']
#         scans = scans.to(device) #  b seq_len robot_num 90 2
#         positions = batch['position'] 
#         positions = positions.to(device) # b seq_len robot 
#         targets = batch['target']
#         targets = targets.to(device)  # b seq_len robot_num FUTURE_STEP 90 2
        
        
#         batch_size = scans.size(0)
#         batch_size,_, robot_num,_, _ = scans.shape
        
#         # robot positions:
#         pos = positions[:,:SEQ_LEN,:,:]
#         current_pos = positions[:,SEQ_LEN-1,:,:]
#         # Transform the robot past poses to the predicted reference frame.
#         #x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
#         #print(x_odom.shape,y_odom.shape,theta_odom.shape) # b seq_len 3
#         lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
        
#         lidar_t=targets[:,SEQ_LEN-1,:,:,:,:] #b robot_num FUTURE_STEP 90 2
#         # if not lidar[:,-1,:,:,:].equal(lidar_t[:,:,0,:,:]): #reset?
#         #         lidar_t=lidar[:,-1,:,:,:].unsqueeze(2).repeat(1,1,FUTURE_STEP,1,1)

#         for r in range(robot_num):
#             optimizer.zero_grad()
#             prediction_list=[]
#             mask_list=[]
            
            
#             #x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
            
            

#             for t in range(FUTURE_STEP): 
#                 #get target map
#                 pos=positions[:,SEQ_LEN-1,:,:]
#                 ego_pos=pos[:,r,:].unsqueeze(1).repeat(1,robot_num,1)
#                 x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(pos,ego_pos)
#                 static_obst_t, dynamic_obst_t=transform_ogm(lidar_t[:,:,t,:,:],pos,x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)

#                 # if r==0:
#                 #     plot_ogm(static_obst_t[0,-1,:,:].unsqueeze(0),f'static_obst_target_{t}.png')
#                 if t>0:
#                     #static_obst_prev, dynamic_obst_prev=transform_ogm(lidar_t[:,:,t-1,:,:].unsqueeze(1),current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
#                     static_obst_t,dynamic_obst_t,RESET_COUNT=update_static_obst(static_obst_t, static_obst_perv,dynamic_obst_t, dynamic_obst_perv,RESET_COUNT)
#                 static_obst_perv=static_obst_t.clone()
#                 dynamic_obst_perv=dynamic_obst_t.clone()
#                 mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                
#                 mask_list.append(mask_map)

#                 dogm_list=[]
#                 sogm_list=[]
#                 #plot_ogm(mask_map[0],'mask.png')
#                 for rr in range(robot_num):
                    
#                     #middle fusion
#                     if rr==r:
#                         current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
#                         pos=positions[:,:SEQ_LEN,r,:].unsqueeze(2).repeat(1,1,robot_num,1)
#                         lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
#                         x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
#                         static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rr,map_size=32)
#                         past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:] # b _ 32 32
#                         current_static_obst=static_obst[:,-1,:,:]
#                         #plot_ogm((static_obst[0,-1,:,:]+dynamic_obst[0,-1,:,:]).unsqueeze(0).clamp(0,1),f'static_obst_{rr}.png')
#                         if len(prediction_list)>0:
#                             pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
#                             #plot_ogm(mask_map[0],'mask.png')    
#                             #plot_ogm(pred_ogm[0][-1].unsqueeze(0),'pred_ogm.png')
#                             pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
#                             #plot_ogm(pred_dogm[0][-1].unsqueeze(0),'pred_dogm.png')
#                             dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
#                         else:
#                             dynamic_obst=past_dogm
#                         dogm_list.append(dynamic_obst)
#                         sogm_list.append(current_static_obst)
#                         #z_ego, kl_loss = encoder(dynamic_obst, current_static_obst)
                        
#                         #prediction=decoder(z_ego)
                        
                        
#                     if rr!=r:
                        
#                         current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
#                         pos=positions[:,:SEQ_LEN,rr,:].unsqueeze(2).repeat(1,1,robot_num,1)
#                         x_odom, y_odom, theta_odom= get_transform_coordinate(pos,current_pos)
#                         lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
#                         static_obst_, dynamic_obst_=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rr,map_size=32)
#                         dynamic_obst_=dynamic_obst_[:,:SEQ_LEN,:,:]
#                         current_static_obst_=static_obst_[:,-1,:,:]
#                         #plot_ogm((static_obst_[0,-1,:,:]+dynamic_obst_[0,-1,:,:]).unsqueeze(0).clamp(0,1),f'static_obst_{rr}.png')
#                         dogm_list.append(dynamic_obst_)
#                         sogm_list.append(current_static_obst_)
#                         # z, _ = encoder(dynamic_obst_.detach(), current_static_obst_.detach())
#                         # z_list.append(z.detach())

#                         # current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
#                         # print(current_pos[0,0])
#                         # pos=positions[:,:SEQ_LEN,rr,:].unsqueeze(2).repeat(1,1,robot_num,1)
#                         # x_odom, y_odom, theta_odom= get_transform_coordinate(pos,current_pos)
#                         # lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
#                         # static_obst_, dynamic_obst_=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rr,map_size=32)
#                         # dynamic_obst_=dynamic_obst_[:,:SEQ_LEN,:,:]
#                         # current_static_obst_=static_obst_[:,-1,:,:]
#                         # plot_ogm((static_obst_[0,-1,:,:]+dynamic_obst_[0,-1,:,:]).unsqueeze(0).clamp(0,1),f'static_obst_{r}_{rr}.png')
#                         # dogm_list.append(dynamic_obst_)
#                         # sogm_list.append(current_static_obst_)

                            

                            
                
#                 #fuse z
#                 dogm_tensor=torch.stack(dogm_list,dim=0).squeeze(1)
#                 sogm_tensor=torch.stack(sogm_list,dim=0).squeeze(1)
                
#                 distance_check = torch.sqrt(x_odom_t**2 + y_odom_t**2) < 10
#                 # plot_ogm(sogm_tensor[0].unsqueeze(0),f'current_static_obst_{t}.png')
#                 # plot_ogm(dogm_tensor[0].unsqueeze(0),f'static_obst_{t}.png')
#                 # plot_ogm(static_obst_t[0].unsqueeze(0),f'static_obst_{t}.png')
#                 prediction, kl_loss = model(dogm_tensor, sogm_tensor,distance_check,r)
                
                
                
#                 prediction_list.append(prediction)
                
                    
#                 # calculate the total loss:
#             prediction_tensor=torch.stack(prediction_list,dim=1).squeeze(2)

#             mask_tensor=torch.stack(mask_list,dim=1).squeeze(2)
            
#             # if r==0:
#             #     for t in range(FUTURE_STEP):
#             #         #plot_ogm(current_static_obst[0].unsqueeze(0),f'current_static_ogm_{t}.png')
#             #         #plot_ogm(prediction_tensor[0][t].unsqueeze(0),'pred.png')
#             #         plot_ogm(mask_tensor[0][t].unsqueeze(0),f'mask_{t}.png')
                
#             ce_loss = criterion(prediction_tensor, mask_tensor).div(batch_size)
#             # beta-vae:
#             # loss = ce_loss + BETA*kl_loss
#             # perform back propagation:
#             wmse = criterion_wmse(prediction_tensor, mask_tensor, calculate_weights(mask_tensor))
#             loss = ce_loss+ BETA*kl_loss
#             loss.backward(torch.ones_like(loss))
#             # for name, param in model.named_parameters():
#             #     print(name, param.grad)
#             optimizer.step()
#             # wmse = criterion_wmse(prediction, mask_map, calculate_weights(mask_map))
#             # loss = wmse + BETA*kl_loss
            
#             total_wmse += wmse.item()
#             ssim_batch = calculate_ssim(prediction_tensor, mask_tensor)
#             total_ssim += ssim_batch.item()
#             # get the loss:
#             # multiple GPUs:
#             if torch.cuda.device_count() > 1:
#                 loss = loss.mean()  
#                 ce_loss = ce_loss.mean()
#                 kl_loss = kl_loss.mean()

#             running_loss += loss.item()
#             # kl_divergence:
#             kl_avg_loss += kl_loss.item()
#             # CE loss:
#             ce_avg_loss += ce_loss.item()
            
            

#         # display informational message:
#         if(i % 128 == 0):
#             print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
#                     .format(epoch, epochs, i + 1, num_batches, loss.item(), ce_loss.item(), kl_loss.item()))
#     train_loss = running_loss / counter 
#     train_kl_loss = kl_avg_loss / counter
#     train_ce_loss = ce_avg_loss / counter
#     avg_wmse = total_wmse / counter
#     avg_ssim = total_ssim / counter

#     return train_loss, train_kl_loss, train_ce_loss, avg_wmse, avg_ssim,RESET_COUNT


def train(model, dataloader, dataset, device, optimizer, criterion, epoch, epochs,RESET_COUNT):
    
    # set model to training mode:seq_len, batch_size, map_size, 
    flag=False
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
        
        
        batch_size = scans.size(0)
        batch_size,_, robot_num,_, _ = scans.shape
        
        # robot positions:
        
        lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
        
        lidar_t=targets[:,SEQ_LEN-1,:,:,:,:] #b robot_num FUTURE_STEP 90 2
        

        for r in range(robot_num):
            optimizer.zero_grad()
            prediction_list=[]
            mask_list=[]
            # #no fusion
            # pos = positions[:,:SEQ_LEN,r,:].unsqueeze(2).repeat(1,1,robot_num,1)
            # current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
            #early fusion
            pos = positions[:,:SEQ_LEN,:,:]
            current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
            # Transform the robot past poses to the predicted reference frame.
            x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
            static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
            #middle fusion
            #all to ego coordination
            
            # if r==0:
            #     plot_ogm(static_obst[0,-1,:,:].unsqueeze(0),f'static_obst_{r}.png')
            # static_obst_perv=None
            # dynamic_obst_perv=None
            for t in range(FUTURE_STEP): 
                # static_obst_t, dynamic_obst_t=convert_lidar_to_ogm(lidar_t[:,r,t,:,:].unsqueeze(1), map_size=32)
                # # if r==0:
                # #    # plot_ogm((static_obst_t[0]+dynamic_obst_t[0]).clamp(0,1),f'mask_s_{t}.png')
                # #     plot_ogm(static_obst_t[0],f'mask_s_{t}.png')
                # mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                #early_fusion
                # print(lidar_t[0,:,0]) #same
                # print(lidar[0,-1]) #same
                # #early fusion
                current_pos = positions[:,SEQ_LEN-1,:,:]
                ego_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,ego_pos)
                static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                # #no fusion
                # current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                # x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,current_pos)
                # static_obst_t, dynamic_obst_t=transform_ogm(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                #middlle fusion
                
                # if r==0 and t==0:
                #     plot_ogm(static_obst_t[0,-1,:,:].unsqueeze(0),f'static_obst_target_{t}.png')
                # if t>0:
                #     #static_obst_prev, dynamic_obst_prev=transform_ogm(lidar_t[:,:,t-1,:,:].unsqueeze(1),current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                #     static_obst_t,dynamic_obst_t,RESET_COUNT=update_static_obst(static_obst_t, static_obst_perv,dynamic_obst_t, dynamic_obst_perv,RESET_COUNT)
                # static_obst_perv=static_obst_t.clone()
                # dynamic_obst_perv=dynamic_obst_t.clone()
                mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                mask_list.append(mask_map)
                # b seq 32 32
                past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:] # b _ 32 32
                current_static_obst=static_obst[:,-1,:,:]
               
                if len(prediction_list)>0:
                    
                    pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                    
                    pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                    # print(pred_ogm[0][-1])
                    # print(current_static_obst.unsqueeze(1))
                    # print(pred_dogm[0][-1])
                    # if r==0:
                    #     plot_ogm(pred_ogm[0][-1].unsqueeze(0),'pred_ogm.png')
                    #     plot_ogm(current_static_obst[0].unsqueeze(0),'current_static_obst.png')
                    #     plot_ogm(past_dogm[0][-1].unsqueeze(0),'past_dogm.png')
                    #     plot_ogm(pred_dogm[0][-1].unsqueeze(0),'pred_dogm.png')
                    dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                else:
                    dynamic_obst=past_dogm
                    
                
               
                prediction, kl_loss = model(dynamic_obst, current_static_obst)
                
                prediction_list.append(prediction)
                
                    
                # calculate the total loss:
            prediction_tensor=torch.stack(prediction_list,dim=1).squeeze(2)

            mask_tensor=torch.stack(mask_list,dim=1).squeeze(2)
            
            # if r==0:
            #     for t in range(FUTURE_STEP):
            #         #plot_ogm(current_static_obst[0].unsqueeze(0),f'current_static_ogm_{t}.png')
            #         #plot_ogm(prediction_tensor[0][t].unsqueeze(0),'pred.png')
            #         plot_ogm(mask_tensor[0][t].unsqueeze(0),f'mask_{t}.png')
                
            ce_loss = criterion(prediction_tensor, mask_tensor).div(batch_size)
            # beta-vae:
            # loss = ce_loss + BETA*kl_loss
            # perform back propagation:
            wmse = criterion_wmse(prediction_tensor, mask_tensor, calculate_weights(mask_tensor))
            loss = ce_loss+ BETA*kl_loss
            loss.backward(torch.ones_like(loss))
            # for name, param in model.named_parameters():
            #     print(name, param.grad)
            optimizer.step()
            # wmse = criterion_wmse(prediction, mask_map, calculate_weights(mask_map))
            # loss = wmse + BETA*kl_loss
            
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

    return train_loss, train_kl_loss, train_ce_loss, avg_wmse, avg_ssim,RESET_COUNT

def train_middle_fusion(model,encoder,decoder,dataloader, dataset, device, optimizer, criterion, epoch, epochs,RESET_COUNT):
    
    # all robots are connected 
    # set model to training mode:seq_len, batch_size, map_size, 
    print('Training Middle Fusion')
    model.train()
    encoder.train()
    decoder.train()
    
    
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
        
        
        batch_size = scans.size(0)
        batch_size,_, robot_num,_, _ = scans.shape
        
        # robot positions:
        pos = positions[:,:SEQ_LEN,:,:]
        current_pos = positions[:,SEQ_LEN-1,:,:]
        # Transform the robot past poses to the predicted reference frame.
        #x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
        #print(x_odom.shape,y_odom.shape,theta_odom.shape) # b seq_len 3
        lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
        
        lidar_t=targets[:,SEQ_LEN-1,:,:,:,:] #b robot_num FUTURE_STEP 90 2
        
        
        for r in range(robot_num):
            optimizer.zero_grad()
            prediction_list=[]
            mask_list=[]
            current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
            
            
            x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
            # static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
            
            # if r==0:
            #     plot_ogm(static_obst[0,-1,:,:].unsqueeze(0),f'static_obst_{r}.png')
            static_obst_perv=None
            dynamic_obst_perv=None
            for t in range(FUTURE_STEP): 
                
                x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(positions[:,SEQ_LEN-1,:,:],current_pos)
                static_obst_t, dynamic_obst_t=transform_ogm(lidar_t[:,:,t,:,:],positions[:,SEQ_LEN-1,:,:], x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                
                # if r==0 :
                #     plot_ogm(static_obst_t[0,-1,:,:].unsqueeze(0),f'static_obst_target_{t}.png')
                # if t>0:
                #     #static_obst_prev, dynamic_obst_prev=transform_ogm(lidar_t[:,:,t-1,:,:].unsqueeze(1),current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                #     static_obst_t,dynamic_obst_t,RESET_COUNT=update_static_obst(static_obst_t, static_obst_perv,dynamic_obst_t, dynamic_obst_perv,RESET_COUNT)
                # static_obst_perv=static_obst_t.clone()
                # dynamic_obst_perv=dynamic_obst_t.clone()
                mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                mask_list.append(mask_map)
                    
                z_list=[]
                # if r==0:
                #     plot_ogm(static_obst_t[0,-1,:,:].unsqueeze(0),f'static_obst_t_{r}.png')
                for rr in range(robot_num):
                    static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rr,map_size=32)
                    # if r==0:
                    #     plot_ogm(static_obst[0,-1,:,:].unsqueeze(0),f'static_obst_{rr}.png')
                    if rr!=r:
                        continue
                        # current_static_obst=static_obst[:,-1,:,:].detach()
                        # # past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:].detach()
                        # # if len(prediction_list)>0:
                        # #     pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                        # #     pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                        # #     dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                        # # else:
                        # #     dynamic_obst=past_dogm
                        # z, kl_l= encoder(dynamic_obst.detach(), current_static_obst)
                        # #kl_loss+=kl_l
                        # #z_list.append(z)
                        
                        
                    if rr==r:
                        current_static_obst=static_obst[:,-1,:,:]
                        # if r==0 and r==0:
                        #     plot_ogm(current_static_obst[0].unsqueeze(0),f'current_static_obst_{t}.png')
                        past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:]
                        if len(prediction_list)>0:
                            pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                            pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                            dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                        else:
                            dynamic_obst=past_dogm
                        #z, kl_loss= encoder(dynamic_obst, current_static_obst) # b 128 1
                        #z_list.append(z) 

                # distance_check=torch.sqrt((positions[:,SEQ_LEN-1,r,0].unsqueeze(1)-positions[:,SEQ_LEN-1,:,0])**2+(positions[:,SEQ_LEN-1,r,1].unsqueeze(1)-positions[:,SEQ_LEN-1,:,1])**2)<10
                # # print(distance_check[0])
                # # print(distance_check.float()[0])
                # distance_check=distance_check.unsqueeze(-1).unsqueeze(-1).repeat(1,1,128,1).float() # b 3
                
                # z_tensor=torch.stack(z_list,dim=1)  # b 3 128 1
                
                # combined_z=torch.mean(z_tensor,dim=1)

                # pred_1 = h_enc_tensor.view(batch_size,3*16, 32, 32)
                # combined_h_enc=model3(h_enc_tensor).reshape(batch_size,16,32,32)
                #prediction = decoder(z)
                prediction,kl_loss=model(dynamic_obst, current_static_obst)
                # if r==0:
                #     plot_ogm(prediction[0],f'pred_{r}.png')
                #     plot_ogm(mask_map[0],f'mask_{r}.png')
                    
                prediction_list.append(prediction)
                
                    
                # calculate the total loss:
            prediction_tensor=torch.stack(prediction_list,dim=1).squeeze(2)

            mask_tensor=torch.stack(mask_list,dim=1).squeeze(2)
            
            # if r==0:
            #     for t in range(FUTURE_STEP):
            #         plot_ogm(current_static_obst[0].unsqueeze(0),f'current_static_ogm_{t}.png')
            #         #plot_ogm(prediction_tensor[0][t].unsqueeze(0),'pred.png')
            #         plot_ogm(static_obst_t[0],f'mask_{t}.png')
                
            ce_loss = criterion(prediction_tensor, mask_tensor).div(batch_size)
            # beta-vae:
            # loss = ce_loss + BETA*kl_loss
            # perform back propagation:
            wmse = criterion_wmse(prediction_tensor, mask_tensor, calculate_weights(mask_tensor))
            loss = ce_loss+ BETA*kl_loss
            loss.backward(torch.ones_like(loss))
            #torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            #torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            # for name, param in encoder.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            # for name, param in decoder.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
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

    return train_loss, train_kl_loss, train_ce_loss, avg_wmse, avg_ssim,RESET_COUNT
def calculate_occupied_grid_rate(grid_map):
    # Assuming grid_map is a binary map (0 for free space, 1 for occupied)
    occupied_cells = torch.sum(grid_map > 0).item()
    total_cells = grid_map.numel()
    occupied_rate = occupied_cells / total_cells
    return occupied_rate


def update_static_obst(static_obst_t, static_obst_prev,dynamic_obst_t, dynamic_obst_prev,RESET_COUNT):
    # Get the dimensions
    batch_size, seq_len, map_size, _ = static_obst_t.shape

    for bz in range(batch_size):
        for sz in range(seq_len):
            obstacles_prev = (static_obst_prev[bz, sz] == 1)
            obstacles_current = (static_obst_t[bz, sz] == 1)

            # Calculate IoU
            intersection = (obstacles_prev & obstacles_current).sum().float()
            union = (obstacles_prev | obstacles_current).sum().float()

            if union > 0:  # Avoid division by zero
                iou = intersection / union
            else:
                iou = torch.tensor(1.0)  # If both are empty, consider them perfectly overlapping

            # Debug prints
            #print(f"Batch {bz}, Seq {sz}, IoU: {iou:.3f}")

            # Check if IoU is below a certain threshold
            if iou < 0.8:  # Assuming 0.8 is the threshold below which we revert to previous map
                RESET_COUNT+=1
                # print(iou)
                # plot_ogm(static_obst_t[bz, sz].unsqueeze(0), f'static_obst_t_batch{bz}_seq{sz}.png')
                # plot_ogm(static_obst_prev[bz, sz].unsqueeze(0), f'static_obst_prev_batch{bz}_seq{sz}.png')
                static_obst_t[bz, sz] = static_obst_prev[bz, sz].clone()
                dynamic_obst_t[bz, sz] = dynamic_obst_prev[bz, sz].clone()
                # plot_ogm(static_obst_t[bz, sz].unsqueeze(0), f'static_obst_t_batch{bz}_seq{sz}.png')
                # plot_ogm(static_obst_prev[bz, sz].unsqueeze(0), f'static_obst_prev_batch{bz}_seq{sz}.png')
            # else:
            #     print("IoU is satisfactory, no update needed.")
                
            

    return static_obst_t,dynamic_obst_t,RESET_COUNT


def validate(model, dataloader, dataset, device, criterion,RESET_COUNT): 
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
    #for i, batch in enumerate(dataloader, 0):
            counter += 1
            # collect the samples as a batch:
            scans = batch['scan']
            scans = scans.to(device)
            positions = batch['position']
            positions = positions.to(device)
            targets = batch['target']
            targets = targets.to(device) 
            
            
            
            # create occupancy maps:
            batch_size = scans.size(0)
            batch_size, _, robot_num,_, _ = scans.shape
            
            # robot positions:
            pos = positions[:,:SEQ_LEN,:,:]
            #current_pos = positions[:,SEQ_LEN-1,:,:]
            # Transform the robot past poses to the predicted reference frame.
            #x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
            #print(x_odom.shape,y_odom.shape,theta_odom.shape) # b seq_len 3
            lidar=scans[:,:SEQ_LEN,:,:,:]

            lidar_t=targets[:,SEQ_LEN-1,:,:,:,:]
        
        
            for r in range(robot_num):
                mask_list=[]
                prediction_list=[]
                # #no fusion
                # current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                # pos = positions[:,:SEQ_LEN,r,:].unsqueeze(2).repeat(1,1,robot_num,1)
                # x_odom, y_odom, theta_odom=get_transform_coordinate(pos,current_pos)
                # static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                #early fusion
                pos = positions[:,:SEQ_LEN,:,:]
                current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                # Transform the robot past poses to the predicted reference frame.
                x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
                static_obst, dynamic_obst=transform_ogm_early_fusion(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                
                # static_obst_perv=None
                # dynamic_obst_perv=None
                for t in range(FUTURE_STEP): 
                    # static_obst_t, dynamic_obst_t=convert_lidar_to_ogm(lidar_t[:,r,t,:,:].unsqueeze(1), map_size=32)
                    # mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1) 
                    # mask_list.append(mask_map)

                    # #no_fusion
                    # x_odom_t, y_odom_t, theta_odom_t =  get_transform_coordinate(current_pos,current_pos)
                    # static_obst_t, dynamic_obst_t=transform_ogm(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                    #early fusion
                    current_pos = positions[:,SEQ_LEN-1,:,:]
                    ego_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                    x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(current_pos,ego_pos)
                    static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t[:,:,t,:,:],current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                        
                    # if t>0:
                    #     static_obst_prev, dynamic_obst_prev=transform_ogm(lidar_t[:,:,t-1,:,:].unsqueeze(1),current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                    #     static_obst_t_=update_static_obst(static_obst_t, static_obst_prev)
                    # else:
                    #     static_obst_t_=static_obst_t
                    # if t>0:
                    # #static_obst_prev, dynamic_obst_prev=transform_ogm(lidar_t[:,:,t-1,:,:].unsqueeze(1),current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                    #     static_obst_t,dynamic_obst_t,RESET_COUNT=update_static_obst(static_obst_t, static_obst_perv,dynamic_obst_t, dynamic_obst_perv,RESET_COUNT)
                    # static_obst_perv=static_obst_t.clone()
                    # dynamic_obst_perv=dynamic_obst_t.clone()
                    mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                    mask_list.append(mask_map)    

                    # b seq 32 32
                    past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:] # b _ 32 32
                    current_static_obst=static_obst[:,-1,:,:]
                    
                    
                    if len(prediction_list)>0:
                        pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                        #plot_ogm(mask_map[0],'mask.png')    
                        #plot_ogm(pred_ogm[0][-1].unsqueeze(0),'pred_ogm.png')
                        pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                        #plot_ogm(pred_dogm[0][-1].unsqueeze(0),'pred_dogm.png')
                        dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                    else:
                        dynamic_obst=past_dogm
                    
                    
                    prediction, kl_loss = model(dynamic_obst, current_static_obst)
                    # if r==0:
                    #     plot_ogm(current_static_obst[0].unsqueeze(0),f'current_static_ogm_{t}.png')
                    #     plot_ogm((dynamic_obst)[0,-1,:,:].unsqueeze(0),f'current_dynamic_ogm_{t}.png')
                    #     plot_ogm(prediction[0],f'pred_ogm_{t}.png')
                    #     plot_ogm((static_obst_t[0]),f'static_obst_target_{t}.png')
                    #     plot_ogm(mask_map[0],f'mask_{t}.png')
                        
                    prediction_list.append(prediction)
                    # if r==0:
                    #     plot_ogm(prediction[0],'pred.png')
                    #     plot_ogm(mask_map[0],'mask.png')   
                    # calculate the total loss:
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

    return val_loss, val_kl_loss, val_ce_loss, avg_wmse, avg_ssim,RESET_COUNT

# def validate(model,dataloader, dataset, device, criterion,RESET_COUNT): 
#     flag=False
#     model.eval()
    
#     running_loss = 0.0
#     kl_avg_loss = 0.0
#     ce_avg_loss = 0.0
#     total_wmse = 0
#     total_ssim = 0  # Initialize total SSIM
#     criterion_wmse = WeightedMSELoss()

#     counter = 0
#     num_batches = int(len(dataset) / dataloader.batch_size)

#     with torch.no_grad():
#         for i, batch in tqdm(enumerate(dataloader), total=num_batches):
#     #for i, batch in enumerate(dataloader, 0):
#             counter += 1
#             # collect the samples as a batch:
#             scans = batch['scan']
#             scans = scans.to(device)
#             positions = batch['position']
#             positions = positions.to(device)
#             targets = batch['target']
#             targets = targets.to(device) 
            
            
            
#             # create occupancy maps:
#             batch_size = scans.size(0)
#             batch_size, _, robot_num,_, _ = scans.shape
            
#             # robot positions:
#             pos = positions[:,:SEQ_LEN,:,:]
#             #current_pos = positions[:,SEQ_LEN-1,:,:]
#             # Transform the robot past poses to the predicted reference frame.
#             #x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
#             #print(x_odom.shape,y_odom.shape,theta_odom.shape) # b seq_len 3
#             lidar=scans[:,:SEQ_LEN,:,:,:]

#             lidar_t=targets[:,SEQ_LEN-1,:,:,:,:]
#             # if not lidar[:,-1,:,:,:].equal(lidar_t[:,:,0,:,:]):
#             #     #print(lidar_t.shape,lidar.shape)
#             #     lidar_t=lidar[:,-1,:,:,:].unsqueeze(2).repeat(1,1,FUTURE_STEP,1,1)
#             # print(lidar[0,-1,0,:,0])
#             # print(lidar_t[0,0,0,:,0])
        
#             for r in range(robot_num):
#                 prediction_list=[]
#                 mask_list=[]
#                 for t in range(FUTURE_STEP): 
#                     #get target map
#                     pos=positions[:,SEQ_LEN-1,:,:]
#                     ego_pos=pos[:,r,:].unsqueeze(1).repeat(1,robot_num,1)
#                     x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(pos,ego_pos)
#                     static_obst_t, dynamic_obst_t=transform_ogm(lidar_t[:,:,t,:,:],pos,x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)

#                     # if r==0:
#                     #     plot_ogm(static_obst_t[0,-1,:,:].unsqueeze(0),f'static_obst_target_{t}.png')
#                     if t>0:
#                         #static_obst_prev, dynamic_obst_prev=transform_ogm(lidar_t[:,:,t-1,:,:].unsqueeze(1),current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
#                         static_obst_t,dynamic_obst_t,RESET_COUNT=update_static_obst(static_obst_t, static_obst_perv,dynamic_obst_t, dynamic_obst_perv,RESET_COUNT)
#                     static_obst_perv=static_obst_t.clone()
#                     dynamic_obst_perv=dynamic_obst_t.clone()
#                     mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                    
#                     mask_list.append(mask_map)

#                     dogm_list=[]
#                     sogm_list=[]
#                     #plot_ogm(mask_map[0],'mask.png')
#                     for rr in range(robot_num):
                        
#                         #middle fusion
#                         if rr==r:
#                             current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
#                             pos=positions[:,:SEQ_LEN,r,:].unsqueeze(2).repeat(1,1,robot_num,1)
#                             lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
#                             x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
#                             static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rr,map_size=32)
#                             past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:] # b _ 32 32
#                             current_static_obst=static_obst[:,-1,:,:]
#                             #plot_ogm((static_obst[0,-1,:,:]+dynamic_obst[0,-1,:,:]).unsqueeze(0).clamp(0,1),f'static_obst_{rr}.png')
#                             if len(prediction_list)>0:
#                                 pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
#                                 #plot_ogm(mask_map[0],'mask.png')    
#                                 #plot_ogm(pred_ogm[0][-1].unsqueeze(0),'pred_ogm.png')
#                                 pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
#                                 #plot_ogm(pred_dogm[0][-1].unsqueeze(0),'pred_dogm.png')
#                                 dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
#                             else:
#                                 dynamic_obst=past_dogm
#                             dogm_list.append(dynamic_obst)
#                             sogm_list.append(current_static_obst)
#                             #z_ego, kl_loss = encoder(dynamic_obst, current_static_obst)
                            
#                             #prediction=decoder(z_ego)
                            
                            
#                         if rr!=r:
#                             with torch.no_grad():
#                                 current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
#                                 pos=positions[:,:SEQ_LEN,rr,:].unsqueeze(2).repeat(1,1,robot_num,1)
#                                 x_odom, y_odom, theta_odom= get_transform_coordinate(pos,current_pos)
#                                 lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
#                                 static_obst_, dynamic_obst_=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rr,map_size=32)
#                                 dynamic_obst_=dynamic_obst_[:,:SEQ_LEN,:,:]
#                                 current_static_obst_=static_obst_[:,-1,:,:]
#                                 #plot_ogm((static_obst_[0,-1,:,:]+dynamic_obst_[0,-1,:,:]).unsqueeze(0).clamp(0,1),f'static_obst_{rr}.png')
#                                 dogm_list.append(dynamic_obst_)
#                                 sogm_list.append(current_static_obst_)
#                                 # z, _ = encoder(dynamic_obst_.detach(), current_static_obst_.detach())
#                                 # z_list.append(z.detach())

#                                 # current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
#                                 # print(current_pos[0,0])
#                                 # pos=positions[:,:SEQ_LEN,rr,:].unsqueeze(2).repeat(1,1,robot_num,1)
#                                 # x_odom, y_odom, theta_odom= get_transform_coordinate(pos,current_pos)
#                                 # lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
#                                 # static_obst_, dynamic_obst_=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rr,map_size=32)
#                                 # dynamic_obst_=dynamic_obst_[:,:SEQ_LEN,:,:]
#                                 # current_static_obst_=static_obst_[:,-1,:,:]
#                                 # plot_ogm((static_obst_[0,-1,:,:]+dynamic_obst_[0,-1,:,:]).unsqueeze(0).clamp(0,1),f'static_obst_{r}_{rr}.png')
#                                 # dogm_list.append(dynamic_obst_)
#                                 # sogm_list.append(current_static_obst_)

                                

                                
                    
#                     #fuse z
#                     dogm_tensor=torch.stack(dogm_list,dim=0).squeeze(1)
#                     sogm_tensor=torch.stack(sogm_list,dim=0).squeeze(1)
                    
#                     distance_check = torch.sqrt(x_odom_t**2 + y_odom_t**2) < 10
#                     # plot_ogm(sogm_tensor[0].unsqueeze(0),f'current_static_obst_{t}.png')
#                     # plot_ogm(dogm_tensor[0].unsqueeze(0),f'static_obst_{t}.png')
#                     # plot_ogm(static_obst_t[0].unsqueeze(0),f'static_obst_{t}.png')
#                     prediction, kl_loss = model(dogm_tensor, sogm_tensor,distance_check,r)
                    
                    
                    
#                     prediction_list.append(prediction)
                        
                    
#                     # if r==0:
#                     #     plot_ogm(prediction[0],'pred.png')
#                     #     plot_ogm(mask_map[0],'mask.png')   
#                     # calculate the total loss:
#                 prediction_tensor=torch.stack(prediction_list,dim=1).squeeze(2) 
                
#                 mask_tensor=torch.stack(mask_list,dim=1).squeeze(2)
#                 ce_loss = criterion(prediction_tensor, mask_tensor).div(batch_size)
                
#                 loss = ce_loss+ BETA*kl_loss
#                 wmse = criterion_wmse(prediction_tensor, mask_tensor, calculate_weights(mask_tensor))
#                 total_wmse += wmse.item()
#                 ssim_batch = calculate_ssim(prediction_tensor, mask_tensor)
#                 total_ssim += ssim_batch.item()
#                 # get the loss:
#                 # multiple GPUs:
#                 if torch.cuda.device_count() > 1:
#                     loss = loss.mean()  
#                     ce_loss = ce_loss.mean()
#                     kl_loss = kl_loss.mean()

#                 running_loss += loss.item()
#                 # kl_divergence:
#                 kl_avg_loss += kl_loss.item()
#                 # CE loss:
#                 ce_avg_loss += ce_loss.item()
                

#     val_loss = running_loss / counter
#     val_kl_loss = kl_avg_loss / counter
#     val_ce_loss = ce_avg_loss / counter
#     avg_wmse = total_wmse / counter
#     avg_ssim = total_ssim / counter

#     return val_loss, val_kl_loss, val_ce_loss, avg_wmse, avg_ssim,RESET_COUNT


def validate_middle_fusion(encoder,decoder,dataloader, dataset, device, criterion,RESET_COUNT):
    encoder.eval()
    decoder.eval()
    
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
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
        
            counter += 1
            # collect the samples as a batch:
            scans = batch['scan']
            scans = scans.to(device) #  b seq_len robot_num 90 2
            positions = batch['position'] 
            positions = positions.to(device) # b seq_len robot 
            targets = batch['target']
            targets = targets.to(device)  # b seq_len robot_num FUTURE_STEP 90 2
            
            
            batch_size = scans.size(0)
            batch_size,_, robot_num,_, _ = scans.shape
            
            # robot positions:
            pos = positions[:,:SEQ_LEN,:,:]
            current_pos = positions[:,SEQ_LEN-1,:,:]
            # Transform the robot past poses to the predicted reference frame.
            #x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
            #print(x_odom.shape,y_odom.shape,theta_odom.shape) # b seq_len 3
            lidar=scans[:,:SEQ_LEN,:,:,:] # b seq_len robot_num 90 2
            
            lidar_t=targets[:,SEQ_LEN-1,:,:,:,:] #b robot_num FUTURE_STEP 90 2
            

            for r in range(robot_num):
                
                prediction_list=[]
                mask_list=[]
                current_pos = positions[:,SEQ_LEN-1,r,:].unsqueeze(1).repeat(1,robot_num,1)
                
                # Transform the robot past poses to the predicted reference frame.
                
                x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
                # static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                
                # if r==0:
                #     plot_ogm(static_obst[0,-1,:,:].unsqueeze(0),f'static_obst_{r}.png')
                static_obst_perv=None
                dynamic_obst_perv=None
                for t in range(FUTURE_STEP): 
                    
                    x_odom_t, y_odom_t, theta_odom_t = get_transform_coordinate(positions[:,SEQ_LEN-1,:,:],current_pos)
                    static_obst_t, dynamic_obst_t=transform_ogm(lidar_t[:,:,t,:,:],positions[:,SEQ_LEN-1,:,:], x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                    
                    # if r==0 and t==0:
                    #     plot_ogm(static_obst_t[0,-1,:,:].unsqueeze(0),f'static_obst_target_{t}.png')
                    if t>0:
                        #static_obst_prev, dynamic_obst_prev=transform_ogm(lidar_t[:,:,t-1,:,:].unsqueeze(1),current_pos, x_odom_t, y_odom_t, theta_odom_t,r,map_size=32)
                        static_obst_t,dynamic_obst_t,RESET_COUNT=update_static_obst(static_obst_t, static_obst_perv,dynamic_obst_t, dynamic_obst_perv,RESET_COUNT)
                    static_obst_perv=static_obst_t.clone()
                    dynamic_obst_perv=dynamic_obst_t.clone()
                    mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                    mask_list.append(mask_map)
                        
                    z_list=[]
                    for rr in range(robot_num):
                        static_obst, dynamic_obst=transform_ogm(lidar, pos,x_odom, y_odom, theta_odom, rr,map_size=32)
                        if rr!=r:
                            continue
                            # current_static_obst=static_obst[:,-1,:,:].detach()
                            # # past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:].detach()
                            # # if len(prediction_list)>0:
                            # #     pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                            # #     pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                            # #     dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                            # # else:
                            # #     dynamic_obst=past_dogm
                            # z, _ = encoder(dynamic_obst.detach(), current_static_obst)
                            # #z_list.append(z)
                            
                        if rr==r:
                            current_static_obst=static_obst[:,-1,:,:]
                            past_dogm=dynamic_obst[:,len(prediction_list):SEQ_LEN,:,:]
                            if len(prediction_list)>0:
                                pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
                                pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
                                dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
                            else:
                                dynamic_obst=past_dogm
                            z, kl_loss= encoder(dynamic_obst, current_static_obst)
                            #print(h_enc.shape) # b 32 32 32
                            
                            z_list.append(z)

                    distance_check=torch.sqrt((positions[:,SEQ_LEN-1,r,0].unsqueeze(1)-positions[:,SEQ_LEN-1,:,0])**2+(positions[:,SEQ_LEN-1,r,1].unsqueeze(1)-positions[:,SEQ_LEN-1,:,1])**2)<10
                    # print(distance_check[0])
                    # print(distance_check.float()[0])
                    distance_check=distance_check.unsqueeze(-1).unsqueeze(-1).repeat(1,1,128,1)
                    z_tensor=torch.stack(z_list,dim=1)*distance_check.float()
                    combined_z=torch.sum(z_tensor,dim=1)
                    #print(combined_h_enc.shape)
                    # h_enc_tensor = h_enc_tensor.view(batch_size,3*16, 32, 32)
                    # combined_h_enc=model3(h_enc_tensor).reshape(batch_size,16,32,32)
                    
                    prediction = decoder(z_list[0])
                    
                    prediction_list.append(prediction)
                    
                        
                    # calculate the total loss:
                prediction_tensor=torch.stack(prediction_list,dim=1).squeeze(2)

                mask_tensor=torch.stack(mask_list,dim=1).squeeze(2)
                
                # if r==0:
                #     for t in range(FUTURE_STEP):
                #         #plot_ogm(current_static_obst[0].unsqueeze(0),f'current_static_ogm_{t}.png')
                #         #plot_ogm(prediction_tensor[0][t].unsqueeze(0),'pred.png')
                #         plot_ogm(mask_tensor[0][t].unsqueeze(0),f'mask_{t}.png')
                    
                ce_loss = criterion(prediction_tensor, mask_tensor).div(batch_size)
                # beta-vae:
                # loss = ce_loss + BETA*kl_loss
                # perform back propagation:
                wmse = criterion_wmse(prediction_tensor, mask_tensor, calculate_weights(mask_tensor))
                loss = ce_loss+ BETA*kl_loss
                
                # wmse = criterion_wmse(prediction, mask_map, calculate_weights(mask_map))
                # loss = wmse + BETA*kl_loss
                
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
            
            
        #     print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
        #             .format(epoch, epochs, i + 1, num_batches, loss.item(), ce_loss.item(), kl_loss.item()))
    val_loss = running_loss / counter 
    val_kl_loss = kl_avg_loss / counter
    val_ce_loss = ce_avg_loss / counter
    avg_wmse = total_wmse / counter
    avg_ssim = total_ssim / counter

    return val_loss, val_kl_loss, val_ce_loss, avg_wmse, avg_ssim,RESET_COUNT
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

    # checkpoint = torch.load('base_nf_model_FUTURE_40.pth')
    # model.load_state_dict(checkpoint['model'])
    #model2.load_state_dict(checkpoint['model'],strict=False)
    #optimizer.load_state_dict(checkpoint['optimizer'])
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
        
        # train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss,wmse_train,ssim_train,RESET_COUNT_train= train_middle_fusion(model,
        #     encoder,decoder,train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs,RESET_COUNT=0
        # )
        # valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss,wmse,ssim,RESET_COUNT = validate_middle_fusion(
        #     encoder,decoder,dev_dataloader, dev_dataset, device, criterion,RESET_COUNT=0
        # )
        
        # train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss,wmse_train,ssim_train,RESET_COUNT_train= train(
        #     model,train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs,RESET_COUNT=0
        # )
        # valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss,wmse,ssim,RESET_COUNT = validate(
        #     model, dev_dataloader, dev_dataset, device, criterion,RESET_COUNT=0
        # )
        #middle fusion
        train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss,wmse_train,ssim_train,RESET_COUNT_train= train(
            model,train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs,RESET_COUNT=0
        )
        valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss,wmse,ssim,RESET_COUNT = validate(
            model, dev_dataloader, dev_dataset, device, criterion,RESET_COUNT=0
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
        wandb.log({"Train Loss": train_epoch_loss, "Train KL Loss": train_kl_epoch_loss, "Train CE Loss": train_ce_epoch_loss,"Train_WMSE": wmse_train/3, "Train_SSIM": ssim_train/3,
                "Validation Loss": valid_epoch_loss, "Validation KL Loss": valid_kl_epoch_loss, "Validation CE Loss": valid_ce_epoch_loss,
                "Val_WMSE": wmse/3, "Val_SSIM": ssim/3})
        #"Train RESET": RESET_COUNT_train,"Val RESET": RESET_COUNT
        
        
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, train WMSE: {:.4f}, train SSIM: {:.4f}, val WMSE: {:.4f}, val SSIM: {:.4f}'.format(
            epoch, epochs, train_epoch_loss, valid_epoch_loss, wmse_train/3,ssim_train/3,wmse/3, ssim/3))
        
        
        
        # save the model:
        if(epoch % 10== 0):
            if torch.cuda.device_count() > 1: # multiple GPUS: 
                state = {'model':model.modules.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            else:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            path='mf_model_4FUTURE_' + str(epoch) +'.pth'
            torch.save(state, path)
            


        epoch_num = epoch

    # # save the final model
    # if torch.cuda.device_count() > 1: # multiple GPUS: 
    #     state = {'model1':model1.module.state_dict(), 'model2':model2.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    # else:
    #     state = {'model1':model1.state_dict(), 'model2':model2.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    # torch.save(state, 'mfusion_model_4FUTURE_STEP_final.pth')

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
