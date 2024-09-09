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
BATCH_SIZE = 16 #512 #64
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
def transform_all_ogm(lidar_data, pos,x_odom, y_odom, theta_odom, robot_index,map_size=32):
        
        if pos.dim() == 3:
            pos_=pos.unsqueeze(1)
        else:
            pos_=pos
        if lidar_data.dim() == 4:
            lidar_data_=lidar_data.unsqueeze(1)
        else:
            lidar_data_=lidar_data
        
        _,_,robot_num,num_ray,_= lidar_data_.shape
        #print(x_odom.shape, y_odom.shape, theta_odom.shape)
        static_list=[]
        dynamic_list=[]
        for r in range(robot_num):
            # transform past to current
            initial_angles = pos_[:,:,r,2]  # Assuming the 4th index is the angle in past_vel_pos
            
            # Compute Cartesian coordinates from polar coordinates
            distances = lidar_data_[:,:,r,:,0]
            
            angles = initial_angles + theta_odom[:,:,r]  # Correct for current orientation
            angles_ = angles.unsqueeze(-1).repeat(1,1,num_ray)
            distance_x = distances * torch.cos(angles_)
            distance_y = distances * torch.sin(angles_)
    
            # Apply translations
        
            x_odom_ = x_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
            y_odom_ = y_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
            distances_x = distance_x + x_odom_
            distances_y = distance_y + y_odom_
            
            # Calculate new distances from transformed coordinates
            new_distances = torch.sqrt(distances_x**2 + distances_y**2)
            
            lidar_data_[:,:,r,:,0] = new_distances
            static_obst, dynamic_obst=convert_lidar_to_ogm(lidar_data_[:,:,r,:,:],map_size)
            static_list.append(static_obst)
            dynamic_list.append(dynamic_obst)
        static_map_tensor=torch.stack(static_list,dim=0)
        dynamic_map_tensor=torch.stack(dynamic_list,dim=0)
        return static_map_tensor,dynamic_map_tensor
def transform_ogm(lidar_data, pos,x_odom, y_odom, theta_odom, robot_index,map_size=32):
    
    
    # Calculate initial angles from past_vel_pos
    
    if pos.dim() == 3:
        pos_=pos.unsqueeze(1)
    else:
        pos_=pos
    if lidar_data.dim() == 4:
        lidar_data_=lidar_data.unsqueeze(1)
    else:
        lidar_data_=lidar_data
    
    _,_,robot_num,num_ray,_= lidar_data_.shape
    #no fusion
    initial_angles = pos_[:,:,robot_index,2]  # Assuming the 4th index is the angle in past_vel_pos
    
    # Compute Cartesian coordinates from polar coordinates
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
    
    # Convert the transformed points to grid coordinates

    # #early fusion
    # for r in range(robot_num):
    #     # transform past to current
    #     initial_angles = pos_[:,:,r,2]  # Assuming the 4th index is the angle in past_vel_pos
        
    #     # Compute Cartesian coordinates from polar coordinates
    #     distances = lidar_data_[:,:,r,:,0]
        
    #     angles = initial_angles + theta_odom[:,:,r]  # Correct for current orientation
    #     angles_ = angles.unsqueeze(-1).repeat(1,1,num_ray)
    #     distance_x = distances * torch.cos(angles_)
    #     distance_y = distances * torch.sin(angles_)

    #     # Apply translations
    
    #     x_odom_ = x_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
    #     y_odom_ = y_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
    #     distances_x = distance_x + x_odom_
    #     distances_y = distance_y + y_odom_
        
    #     # Calculate new distances from transformed coordinates
    #     new_distances = torch.sqrt(distances_x**2 + distances_y**2)
        
    #     lidar_data_[:,:,r,:,0] = new_distances
    #     if r!=robot_index:
    #         # transform to ego and merge
            
    #         dx = pos_[:,:,robot_index,0]-pos_[:,:,r,0]
    #         dy = pos_[:,:,robot_index,1]-pos_[:,:,r,1]
    #         th = pos_[:,:,robot_index,2]-pos_[:,:,r,2]
    #         x_odom_e = torch.cos(th) * dx + torch.sin(th) * dy
    #         y_odom_e = torch.sin(-th) * dx + torch.cos(th) * dy
    #         theta_odom_e = pos_[:,:,robot_index,2] - th
    #         initial_angles = pos_[:,:,r,2]
    #         distances = lidar_data_[:,:,r,:,0]
    #         angles = initial_angles + theta_odom_e
    #         angles_ = angles.unsqueeze(-1).repeat(1,1,num_ray)
    #         distance_x = distances * torch.cos(angles_)
    #         distance_y = distances * torch.sin(angles_)
    #         x_odom_e_ = x_odom_e.unsqueeze(-1).repeat(1,1,num_ray)
    #         y_odom_e_ = y_odom_e.unsqueeze(-1).repeat(1,1,num_ray)
    #         distances_x = distance_x + x_odom_e_
    #         distances_y = distance_y + y_odom_e_
    #         new_distances = torch.sqrt(distances_x**2 + distances_y**2)
    #         lidar_data_[:,:,r,:,0] = new_distances
            
    #         merged_lidar = copy.deepcopy(lidar_data_[:,:,robot_index,:,:])
    #         lidar2=lidar_data_[:,:,r,:,:]
    #         mask_new_obstacle=(merged_lidar[:,:,:,0] == 0) & (lidar2[:,:,:,0] != 0)
    #         merged_lidar[mask_new_obstacle] = lidar2[mask_new_obstacle]
            
    #         mask_both_inf = (merged_lidar[:,:,:,0] == float('inf')) & (lidar2[:,:,:,0] == float('inf'))
    #         merged_lidar[mask_both_inf] = lidar2[mask_both_inf]
    #         lidar_data_[:,:,robot_index,:,:]=merged_lidar
    # static_obst, dynamic_obst=convert_lidar_to_ogm(lidar_data_[:,:,robot_index,:,:],map_size)
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
    
    _,_,robot_num,num_ray,_= lidar_data_.shape
    
    #early fusion
    for r in range(robot_num):
        # transform past to current
        initial_angles = pos_[:,:,r,2]  # Assuming the 4th index is the angle in past_vel_pos
        
        # Compute Cartesian coordinates from polar coordinates
        distances = lidar_data_[:,:,r,:,0]
        
        angles = initial_angles + theta_odom[:,:,r]  # Correct for current orientation
        angles_ = angles.unsqueeze(-1).repeat(1,1,num_ray)
        distance_x = distances * torch.cos(angles_)
        distance_y = distances * torch.sin(angles_)

        # Apply translations
    
        x_odom_ = x_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
        y_odom_ = y_odom[:,:,r].unsqueeze(-1).repeat(1,1,num_ray)
        distances_x = distance_x + x_odom_
        distances_y = distance_y + y_odom_
        
        # Calculate new distances from transformed coordinates
        new_distances = torch.sqrt(distances_x**2 + distances_y**2)
        
        lidar_data_[:,:,r,:,0] = new_distances
        if r!=robot_index:
            # transform to ego and merge
            
            dx = pos_[:,:,robot_index,0]-pos_[:,:,r,0]
            dy = pos_[:,:,robot_index,1]-pos_[:,:,r,1]
            th = pos_[:,:,robot_index,2]-pos_[:,:,r,2]
            x_odom_e = torch.cos(th) * dx + torch.sin(th) * dy
            y_odom_e = torch.sin(-th) * dx + torch.cos(th) * dy
            theta_odom_e = pos_[:,:,robot_index,2] - th
            initial_angles = pos_[:,:,r,2]
            distances = lidar_data_[:,:,r,:,0]
            angles = initial_angles + theta_odom_e
            angles_ = angles.unsqueeze(-1).repeat(1,1,num_ray)
            distance_x = distances * torch.cos(angles_)
            distance_y = distances * torch.sin(angles_)
            x_odom_e_ = x_odom_e.unsqueeze(-1).repeat(1,1,num_ray)
            y_odom_e_ = y_odom_e.unsqueeze(-1).repeat(1,1,num_ray)
            distances_x = distance_x + x_odom_e_
            distances_y = distance_y + y_odom_e_
            new_distances = torch.sqrt(distances_x**2 + distances_y**2)
            lidar_data_[:,:,r,:,0] = new_distances
            
            merged_lidar = copy.deepcopy(lidar_data_[:,:,robot_index,:,:])
            lidar2=lidar_data_[:,:,r,:,:]
            mask_new_obstacle=(merged_lidar[:,:,:,0] == 0) & (lidar2[:,:,:,0] != 0)
            merged_lidar[mask_new_obstacle] = lidar2[mask_new_obstacle]
            
            mask_both_inf = (merged_lidar[:,:,:,0] == float('inf')) & (lidar2[:,:,:,0] == float('inf'))
            merged_lidar[mask_both_inf] = lidar2[mask_both_inf]
            lidar_data_[:,:,robot_index,:,:]=merged_lidar
    static_obst, dynamic_obst=convert_lidar_to_ogm(lidar_data_[:,:,robot_index,:,:],map_size)
    return static_obst,dynamic_obst
def convert_lidar_to_ogm(lidar_data,map_size):
    
    batch_size,seq_len,num_ray,_= lidar_data.shape
    
    center_index = map_size // 2
    local_ogm = torch.full((batch_size,seq_len,2, map_size, map_size), -1, dtype=torch.float32, device=lidar_data.device)
    for b in range(batch_size):
            for s in range(seq_len):
                # Calculate angles and distances
                angles = torch.linspace(0, 2 * torch.pi, num_ray, device=lidar_data.device)
                distances = lidar_data[b,s,:, 0]
                labels = lidar_data[b,s,:, 1]
                
                

                # Calculate x and y indices
                x_indices = (distances * torch.cos(angles)).long() + center_index
                y_indices = (distances * torch.sin(angles)).long() + center_index

                # Filter indices that are within the map bounds
                mask = (x_indices >= 0) & (x_indices < map_size) & (y_indices >= 0) & (y_indices < map_size)
                x_indices = x_indices[mask]
                y_indices = y_indices[mask]
                labels = labels[mask]


                # Set occupancy and labels
                local_ogm[b,s,0, x_indices, y_indices] = 1  # Set occupied
                local_ogm[b,s,1, x_indices, y_indices] = labels  # Set labels

                # Set unvisited cells to 0 (free space)
                local_ogm[local_ogm == -1] = 0
    # angles=torch.linspace(0, 2 * torch.pi, num_ray, device=lidar_data.device)
    # distances = lidar_data[:,:,:,0]
    # labels = lidar_data[:,:,:,1]
    # x_indices = (distances * torch.cos(angles)).long() + center_index
    # y_indices = (distances * torch.sin(angles)).long() + center_index
    # mask = (x_indices >= 0) & (x_indices < map_size) & (y_indices >= 0) & (y_indices < map_size)
    # x_indices = x_indices[mask]
    # y_indices = y_indices[mask]
    # labels = labels[mask]
    # local_ogm[:,:,0, x_indices, y_indices] = 1  # Set occupied
    # local_ogm[:,:,1, x_indices, y_indices] = labels  # Set labels
    # local_ogm[local_ogm == -1] = 0
    
    # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
    static_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)
    dynamic_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)

    # Extract the two channels from the input tensor
    channel_0 = local_ogm[:,:,0,:,:]  # Shape: (batch_size, 32, 32)
    channel_1 = local_ogm[:,:,1,:,:]  # Shape: (batch_size, 32, 32)

    static_obst = channel_0 * (channel_1 == 1)
    dynamic_obst = channel_0 * (channel_1 >= 2)

    
    return static_obst, dynamic_obst
def train(model, dataloader, dataset, device, optimizer, criterion, epoch, epochs):
    # set model to training mode:seq_len, batch_size, map_size, 
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
    #for i, batch in enumerate(dataloader, 0):
        counter += 1
        # collect the samples as a batch:
        scans = batch['scan']
        scans = scans.to(device)
        positions = batch['position']
        positions = positions.to(device)
        velocities = batch['velocity']
        velocities = velocities.to(device)
        # print('scans:',scans.shape) # b 2*seq_len 3 90 2
        # print('positions:',positions.shape) # b 2*seq_len 3 3
        # print('velocities:',velocities.shape) # b 2*seq_len 3 2
        # print('scans:',scans[0,:10,0,:,0]) # b traj_num seq_len 3 90 2
        
        # create occupancy maps
        batch_size = scans.size(0)
        batch_size,_, robot_num,_, _ = scans.shape
        
        # robot positions:
        pos = positions[:,:SEQ_LEN,:,:]
        current_pos = positions[:,SEQ_LEN,:,:]
        # Transform the robot past poses to the predicted reference frame.
        x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
        #print(x_odom.shape,y_odom.shape,theta_odom.shape) # b seq_len 3
        
        lidar=scans[:,:SEQ_LEN,:,:,:]
        target_pos = positions[:,SEQ_LEN+1,:,:]
        x_odom_t, y_odom_t, theta_odom_t =  get_transform_coordinate(target_pos,current_pos)
        #print(x_odom_t.shape,y_odom_t.shape,theta_odom_t.shape) #b traj_num 1 3
        
        lidar_t=scans[:,SEQ_LEN+1,:,:,:]
        
        
        for r in range(robot_num):
            static_obst, dynamic_obst=transform_ogm_early_fusion(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)

            input_map=(static_obst+dynamic_obst).clamp(0,1)
            
            if epoch==10 :
                for i in range(SEQ_LEN):
                    plot_ogm(input_map[0,i].unsqueeze(0),f'nf_ogm_10th_{i}.png')

            
            static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t, target_pos,x_odom_t, y_odom_t, theta_odom_t, r,map_size=32)
            mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
            if epoch==10 :
                plot_ogm(mask_map[0][0].unsqueeze(0),'nf_mask_10th.png')
            
            
            # set all gradients to 0:
            optimizer.zero_grad()
            # feed the batch to the network:
            
            current_static_obst=static_obst[:,-1,:,:]
            
            prediction, kl_loss = model(dynamic_obst, current_static_obst)
            
            if epoch==10 :
                plot_ogm(prediction[0],'nf_pred_10th.png')
                
            
            # calculate the total loss:
            
            ce_loss = criterion(prediction, mask_map).div(batch_size)
            # beta-vae:
            loss = ce_loss + BETA*kl_loss
            # perform back propagation:
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            wmse = criterion_wmse(prediction, mask_map, calculate_weights(mask_map))
            total_wmse += wmse.item()
            ssim_batch = calculate_ssim(prediction, mask_map)
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
def train_middle_fusion(model, model1,model2,dataloader, dataset, device, optimizer, criterion, epoch, epochs):
    print('train_middle_fusion')
    # all robots are connected 
    # set model to training mode:seq_len, batch_size, map_size, 
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
    #for i, batch in enumerate(dataloader, 0):
        counter += 1
        # collect the samples as a batch:
        scans = batch['scan']
        scans = scans.to(device)
        positions = batch['position']
        positions = positions.to(device)
        velocities = batch['velocity']
        velocities = velocities.to(device)
        
        # create occupancy maps
        batch_size = scans.size(0)
        batch_size,_, robot_num,_, _ = scans.shape
        
        current_pos=positions[:,SEQ_LEN,:,:]
        pos = positions[:,:SEQ_LEN,:,:]
        x_odom_tensor, y_odom_tensor, theta_odom_tensor =  get_all_transform_coordinate(pos,current_pos) #robot_num b seq_len robot_num
        
        distances = torch.sqrt((x_odom_tensor[:, :, -1, :]**2 + y_odom_tensor[:, :, -1, :]**2))
        threshold = 5
        connected_mask = distances < threshold # robot_num b robot_num

        lidar=scans[:,:SEQ_LEN,:,:,:]
        target_pos = positions[:,SEQ_LEN+1,:,:]
        x_odom_t_tensor, y_odom_t_tensor, theta_odom_t_tensor =  get_all_transform_coordinate(target_pos,current_pos)
        
        lidar_t=scans[:,SEQ_LEN+1,:,:,:]
        
        for r in range(robot_num):
            static_obst_tensor,dynamic_obst_tensor=transform_all_ogm(lidar,pos,x_odom_tensor[r], y_odom_tensor[r], theta_odom_tensor[r], r,map_size=32)
            input_map=(static_obst_tensor+dynamic_obst_tensor).clamp(0,1)
            target_static_obst,target_dynamic_obst=transform_ogm_early_fusion(lidar_t,target_pos,x_odom_t_tensor[r], y_odom_t_tensor[r], theta_odom_t_tensor[r], r,map_size=32)
            mask_map=(target_static_obst+target_dynamic_obst).clamp(0,1)
            # print(connected_mask[r,0,:])
            
            # print('connected_mask:',connected_mask[r][0])
            # plot_ogm(mask_map[0][0].unsqueeze(0),'mask.png')
            # for rv in range(robot_num):
            #     plot_ogm(input_map[rv,0,-1,:,:].unsqueeze(0),f'ogm_{r}_{rv}.png')
            #     plot_ogm(static_obst_tensor[rv,0,-1,:,:].unsqueeze(0),f'sogm_{r}_{rv}.png')
            #     plot_ogm(dynamic_obst_tensor[rv,0,-1,:,:].unsqueeze(0),f'dogm_{r}_{rv}.png')
            # exit()        
            h_enc_list=[]
            for rr in range(robot_num):
                if rr!=r:
                    current_static_obst=static_obst_tensor[rr,:,-1,:,:].detach()
                    h_enc, _ = model1(input_map[rr,:,:,:,:].detach(), current_static_obst)
                    # if epoch==5 :
                    #     plot_ogm(input_map[rr,0,-1,:,:].unsqueeze(0),f'ogm_{rr}.png')
                    h_enc_list.append(h_enc)
                else:
                    current_static_obst=static_obst_tensor[rr,:,-1,:,:].detach()
                    h_enc, ego_static_obst= model1(input_map[rr,:,:,:,:], current_static_obst)
                    if epoch==5 :
                        plot_ogm(input_map[rr,0,-1,:,:].unsqueeze(0),f'ogm_ego_{r}_{rr}.png')
                    h_enc_list.append(h_enc)

            
            mask=connected_mask[r].permute(1,0) # 3 16
            mask_expand=mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            h_enc_tensor=torch.stack(h_enc_list)*mask_expand.float() # 3 16 32 32 32 
            
            combined_h_enc=torch.mean(h_enc_tensor,dim=0)
            optimizer.zero_grad()
            prediction, kl_loss = model2(combined_h_enc, ego_static_obst)
            if epoch==5 :
                plot_ogm(prediction[0],f'pred_ego_{r}.png')
            # beta-vae:
            ce_loss = criterion(prediction, mask_map).div(batch_size)
            loss = ce_loss + BETA*kl_loss
            # perform back propagation:
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            wmse = criterion_wmse(prediction, mask_map, calculate_weights(mask_map))
            total_wmse += wmse.item()
            ssim_batch = calculate_ssim(prediction, mask_map)
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
        
        
        # # display informational message:
        # if(i % 128 == 0):
            
        #     print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
        #             .format(epoch, epochs, i + 1, num_batches, loss.item(), ce_loss.item(), kl_loss.item()))
        
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


def validate(model, dataloader, dataset, device, criterion):
    
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
            velocities = batch['velocity']
            velocities = velocities.to(device)
            
            
            # create occupancy maps:
            batch_size = scans.size(0)
            batch_size, _, robot_num,_, _ = scans.shape
            
            # robot positions:
            pos = positions[:,:SEQ_LEN,:,:]
            current_pos = positions[:,SEQ_LEN,:,:]
            # Transform the robot past poses to the predicted reference frame.
            x_odom, y_odom, theta_odom =  get_transform_coordinate(pos,current_pos)
            #print(x_odom.shape,y_odom.shape,theta_odom.shape) # b traj_num seq_len 3
            lidar=scans[:,:SEQ_LEN,:,:,:]
            target_pos = positions[:,SEQ_LEN+1,:,:]
            x_odom_t, y_odom_t, theta_odom_t =  get_transform_coordinate(target_pos,current_pos)
            lidar_t=scans[:,SEQ_LEN+1,:,:,:]
            
            
            for r in range(robot_num):
                static_obst, dynamic_obst=transform_ogm_early_fusion(lidar, pos,x_odom, y_odom, theta_odom, r,map_size=32)
                input_map=(static_obst+dynamic_obst).clamp(0,1)
                static_obst_t, dynamic_obst_t=transform_ogm_early_fusion(lidar_t, target_pos,x_odom_t, y_odom_t, theta_odom_t, r,map_size=32)
                mask_map=(static_obst_t+dynamic_obst_t).clamp(0,1)
                
                #print(input_map.shape,mask_map.shape,static_obst.shape) # b traj_num seq_len 32 32
                
                # set all gradients to 0:
                
                # feed the batch to the network:
               
                current_static_obst=static_obst[:,-1,:,:]
                prediction, kl_loss = model(dynamic_obst, current_static_obst)
                
                ce_loss = criterion(prediction, mask_map).div(batch_size)
                wmse = criterion_wmse(prediction, mask_map, calculate_weights(mask_map))
                total_wmse += wmse.item()
                ssim_batch = calculate_ssim(prediction, mask_map)
                total_ssim += ssim_batch.item()

                loss = ce_loss + BETA * kl_loss
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                    ce_loss = ce_loss.mean()
                    kl_loss = kl_loss.mean()

                running_loss += loss.item()
                kl_avg_loss += kl_loss.item()
                ce_avg_loss += ce_loss.item()

    val_loss = running_loss / counter
    val_kl_loss = kl_avg_loss / counter
    val_ce_loss = ce_avg_loss / counter
    avg_wmse = total_wmse / counter
    avg_ssim = total_ssim / counter

    return val_loss, val_kl_loss, val_ce_loss, avg_wmse, avg_ssim
def validate_middle_fusion(model, model1,model2,dataloader, dataset, device, criterion):
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
            velocities = batch['velocity']
            velocities = velocities.to(device)
            
            
            # create occupancy maps
        batch_size = scans.size(0)
        batch_size,_, robot_num,_, _ = scans.shape
        
        current_pos=positions[:,SEQ_LEN,:,:]
        pos = positions[:,:SEQ_LEN,:,:]
        x_odom_tensor, y_odom_tensor, theta_odom_tensor =  get_all_transform_coordinate(pos,current_pos) #robot_num b seq_len robot_num
        
        distances = torch.sqrt((x_odom_tensor[:, :, -1, :]**2 + y_odom_tensor[:, :, -1, :]**2))
        threshold = 10  
        connected_mask = distances < threshold # robot_num b robot_num

        lidar=scans[:,:SEQ_LEN,:,:,:]
        target_pos = positions[:,SEQ_LEN+1,:,:]
        x_odom_t_tensor, y_odom_t_tensor, theta_odom_t_tensor =  get_all_transform_coordinate(target_pos,current_pos)
        
        lidar_t=scans[:,SEQ_LEN+1,:,:,:]
        
        for r in range(robot_num):
            static_obst_tensor,dynamic_obst_tensor=transform_all_ogm(lidar,pos,x_odom_tensor[r], y_odom_tensor[r], theta_odom_tensor[r], r,map_size=32)
            input_map=(static_obst_tensor+dynamic_obst_tensor).clamp(0,1)
            target_static_obst,target_dynamic_obst=transform_ogm_early_fusion(lidar_t,target_pos,x_odom_t_tensor[r], y_odom_t_tensor[r], theta_odom_t_tensor[r], r,map_size=32)
            mask_map=(target_static_obst+target_dynamic_obst).clamp(0,1)
            h_enc_list=[]
            for rr in range(robot_num):
                if rr!=r:
                    current_static_obst=static_obst_tensor[rr,:,-1,:,:].detach()
                    h_enc, _ = model1(input_map[rr].detach(), current_static_obst)
                    
                    h_enc_list.append(h_enc)
                else:
                    current_static_obst=static_obst_tensor[rr,:,-1,:,:].detach()
                    h_enc, ego_static_obst= model1(input_map[rr], current_static_obst)
                    
                    h_enc_list.append(h_enc)

            
            mask=connected_mask[r].unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,32,32,32).permute(1,0,2,3,4)
            h_enc_tensor=torch.stack(h_enc_list)*mask.float()
            
            combined_h_enc=torch.mean(h_enc_tensor,dim=0)
            
            prediction, kl_loss = model2(combined_h_enc, ego_static_obst)
            # beta-vae:
            ce_loss = criterion(prediction, mask_map).div(batch_size)
            loss = ce_loss + BETA*kl_loss
            # perform back propagation:
            
            
            wmse = criterion_wmse(prediction, mask_map, calculate_weights(mask_map))
            total_wmse += wmse.item()
            ssim_batch = calculate_ssim(prediction, mask_map)
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
        
        
        # # display informational message:
        # if(i % 128 == 0):
            
        #     print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, CE_Loss: {:.4f}, KL_Loss: {:.4f}'
        #             .format(epoch, epochs, i + 1, num_batches, loss.item(), ce_loss.item(), kl_loss.item()))
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
    train_dataset = VaeTestDataset('SOGMP_plus/OGM-datasets/dataset3/train', 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, \
                                                   shuffle=True, drop_last=True, pin_memory=True)

    ### validation data ###
    # validation set and validation data loader
    dev_dataset = VaeTestDataset('SOGMP_plus/OGM-datasets/dataset3/val', 'val')
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, \
                                                 shuffle=True, drop_last=True, pin_memory=True)

    # instantiate a model:
    model = RVAEP(input_channels=NUM_INPUT_CHANNELS,
                  latent_dim=NUM_LATENT_DIM,
                  output_channels=NUM_OUTPUT_CHANNELS)
    #moves the model to device (cpu in our case so no change):
    model.to(device)
    model1=RConvLSTM(input_channels=NUM_INPUT_CHANNELS,
                  latent_dim=NUM_LATENT_DIM,
                  output_channels=NUM_OUTPUT_CHANNELS)
    model2=RVAE(input_channels=NUM_INPUT_CHANNELS,
                  latent_dim=NUM_LATENT_DIM,
                  output_channels=NUM_OUTPUT_CHANNELS)
    model1.to(device)
    model2.to(device)

    # set the adam optimizer parameters:
    opt_params = { LEARNING_RATE: 0.00001,
                   BETAS: (.9,0.999),
                   EPS: 1e-08,
                   WEIGHT_DECAY: .001 }
    # set the loss criterion and optimizer:
    criterion = nn.BCELoss(reduction='sum') #, weight=class_weights)
    criterion.to(device)
    # create an optimizer, and pass the model params to it:
    all_params = list(model1.parameters()) + list(model2.parameters())
    optimizer = Adam(all_params, **opt_params)

    # get the number of epochs to train on:
    epochs = NUM_EPOCHS

    # # if there are trained models, continue training:
    # if os.path.exists(mdl_path):
    #     checkpoint = torch.load(mdl_path)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch']
    #     print('Load epoch {} success'.format(start_epoch))
    # else:
    start_epoch = 0
    print('No trained models, restart training')
    # checkpoint = torch.load('plus_model10.pth')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use 2 of total", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model) #, device_ids=[0, 1])
    # moves the model to device (cpu in our case so no change):
    model.to(device)

    # tensorboard writer:
    writer = SummaryWriter('runs')

    epoch_num = 0
    for epoch in range(start_epoch+1, epochs):
        # adjust learning rate:
        adjust_learning_rate(optimizer, epoch)
        ################################## Train #####################################
        # for each batch in increments of batch size
        #
        
        train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss,wmse_train,ssim_train= train_middle_fusion(
            model, model1,model2,train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs
        )
        valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss,wmse,ssim = validate_middle_fusion(
            model, model1,model2, dev_dataloader, dev_dataset, device, criterion
        )
        # train_epoch_loss, train_kl_epoch_loss, train_ce_epoch_loss,wmse_train,ssim_train= train(
        #     model,train_dataloader, train_dataset, device, optimizer, criterion, epoch, epochs
        # )
        # valid_epoch_loss, valid_kl_epoch_loss, valid_ce_epoch_loss,wmse,ssim = validate(
        #     model, dev_dataloader, dev_dataset, device, criterion
        # )
        
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
         # Log metrics to wandb
        wandb.log({"Train Loss": train_epoch_loss, "Train KL Loss": train_kl_epoch_loss, "Train CE Loss": train_ce_epoch_loss,"Train_WMSE": wmse_train/3, "Train_SSIM": ssim_train/3,
                "Validation Loss": valid_epoch_loss, "Validation KL Loss": valid_kl_epoch_loss, "Validation CE Loss": valid_ce_epoch_loss,
                "Val_WMSE": wmse/3, "Val_SSIM": ssim/3})
        
        
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, train WMSE: {:.4f}, train SSIM: {:.4f}, val WMSE: {:.4f}, val SSIM: {:.4f}'.format(
            epoch, epochs, train_epoch_loss, valid_epoch_loss, wmse_train/3,ssim_train/3,wmse/3, ssim/3))
        
        
        
        # save the model:
        if(epoch % 10 == 0):
            if torch.cuda.device_count() > 1: # multiple GPUS: 
                state = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            else:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            path='plus_model' + str(epoch) +'.pth'
            torch.save(state, path)

        epoch_num = epoch

    # save the final model
    if torch.cuda.device_count() > 1: # multiple GPUS: 
        state = {'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    else:
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch_num}
    torch.save(state, 'plus_model_final.pth')

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
