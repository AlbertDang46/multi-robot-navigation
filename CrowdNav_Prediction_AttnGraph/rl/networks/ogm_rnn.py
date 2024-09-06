from collections import deque
import copy
from signal import pause

from matplotlib import pyplot as plt
#from SOGMP.scripts.model import VAEP
from SOGMP_plus.scripts.model import RVAEP,RConvLSTM,RVAE
import torch.nn as nn
from torch.autograd import Variable
import torch
import cv2
import numpy as np
from torch.optim import Adam
from rl.networks.network_utils import init
from rl.networks.vector_visualize import tensor_to_map
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from rl.networks.torch_transformation_utils import warp_affine
from torch.utils.tensorboard import SummaryWriter
import  wandb
BETA = 0.01
LEARNING_RATE = "lr"
BETAS = "betas"
EPS = "eps"
WEIGHT_DECAY = "weight_decay"
wandb.init(project="ogm_pred")
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
def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))

class MainRNN(nn.Module):
    """
    The class for RNN with done masks
    """
    # edge: True -> edge RNN, False -> node RNN
    def __init__(self, args):
        super(MainRNN, self).__init__()
        self.args = args

        self.gru = nn.GRU(args.ogm_embedding_size + args.robot_info_embedding_size + args.detected_robots_info_embedding_size, args.rnn_hidden_size)
        self.output_linear = nn.Linear(args.rnn_hidden_size, args.actor_critic_output_size)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)


    def _forward_gru(self, x, hxs, masks):
        # for acting model, input shape[0] == hidden state shape[0]
        if x.size(0) == hxs.size(0):
            # use env dimension as batch
            # [1, 12, 6, ?] -> [1, 12*6, ?] or [30, 6, 6, ?] -> [30, 6*6, ?]
            seq_len, nenv, agent_num, _ = x.size()
            x = x.view(seq_len, nenv*agent_num, -1)
            mask_agent_num = masks.size()[-1]
            hxs_times_masks = hxs * (masks.view(seq_len, nenv, mask_agent_num, 1))
            hxs_times_masks = hxs_times_masks.view(seq_len, nenv*agent_num, -1)
            x, hxs = self.gru(x, hxs_times_masks) # we already unsqueezed the inputs in SRNN forward function
            x = x.view(seq_len, nenv, agent_num, -1)
            hxs = hxs.view(seq_len, nenv, agent_num, -1)

        # during update, input shape[0] * nsteps (30) = hidden state shape[0]
        else:

            # N: nenv, T: seq_len, agent_num: node num or edge num
            T, N, agent_num, _ = x.size()
            # x = x.view(T, N, agent_num, x.size(2))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # for the [29, num_env] boolean array, if any entry in the second axis (num_env) is True -> True
            # to make it [29, 1], then select the indices of True entries
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            # hxs = hxs.unsqueeze(0)
            # hxs = hxs.view(hxs.size(0), hxs.size(1)*hxs.size(2), hxs.size(3))
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                # x and hxs have 4 dimensions, merge the 2nd and 3rd dimension
                x_in = x[start_idx:end_idx]
                x_in = x_in.view(x_in.size(0), x_in.size(1)*x_in.size(2), x_in.size(3))
                hxs = hxs.view(hxs.size(0), N, agent_num, -1)
                hxs = hxs * (masks[start_idx].view(1, -1, 1, 1))
                hxs = hxs.view(hxs.size(0), hxs.size(1) * hxs.size(2), hxs.size(3))
                rnn_scores, hxs = self.gru(x_in, hxs)

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T, N, agent_num, -1)
            hxs = hxs.view(1, N, agent_num, -1)

        return x, hxs
    
    def forward(self, robot_info, ogm, detected_robots,hidden_state, masks):
        x = torch.cat((robot_info, ogm, detected_robots), dim=-1)

        output, new_hidden_state = self._forward_gru(x, hidden_state, masks)
        output = self.output_linear(output)

        return output, new_hidden_state

def pose_prediction(vel_pos,pred_step, noise_std=[0,0,0]):
        """
        Predict the future origin pose of the robot: find the predicted reference frame
        """
        
        pos_origin = torch.zeros(1, vel_pos.size(1),vel_pos.size(2),3)
        # Gaussian noise sampled from a distribution with MEAN=0.0 and STD=std
        x_noise = torch.randn(1, vel_pos.size(1),vel_pos.size(2))*noise_std[0]
        y_noise = torch.randn(1, vel_pos.size(1),vel_pos.size(2))*noise_std[1]
        th_noise = torch.randn(1, vel_pos.size(1),vel_pos.size(2))*noise_std[2]
        if torch.cuda.is_available():
            pos_origin = pos_origin.cuda()
            x_noise = x_noise.cuda()
            y_noise = y_noise.cuda()
            th_noise = th_noise.cuda()
        # r_Ao_No: predicted reference position: T = t+n th timestep
        
        d = vel_pos[-1,:,:,0]*0.1*pred_step
        theta = vel_pos[-1,:,:,1]*0.1*pred_step
        pos_origin[0,:,:,0] = vel_pos[-1,:,:,2] + d*torch.cos(vel_pos[-1,:,:,4]) + x_noise
        pos_origin[0,:,:,1] = vel_pos[-1,:,:,3] + d*torch.sin(vel_pos[-1,:,:,4]) + y_noise
        pos_origin[0,:,:,2] = vel_pos[-1,:,:,4] + theta + th_noise
        
        return pos_origin
def get_transform_coordinate(pos, pos_origin):
        """
        Transform the robot past poses to the predicted reference frame.
        """
        #1ahead now
        # # expand the tensor to the same size of pos:
        # pos_origin = pos_origin.unsqueeze(2).expand(pos.size(0), pos.size(2), pos.size(1)).permute(0,2,1)
        # Odometry measurements
        
        dx = pos[:, :, :,2] - pos_origin[:, :, :,2]
        
        dy = pos[:, :, :,3] - pos_origin[:, :, :,3]
        th = pos_origin[:, :, :,4]
        x_odom = torch.cos(th) * dx + torch.sin(th) * dy
        y_odom = torch.sin(-th) * dx + torch.cos(th) * dy
        theta_odom = pos[:, :, :,4] - th
        
        return x_odom, y_odom, theta_odom

def convert_to_bitmap(self, raw_data: np.ndarray, map_size: int) -> np.ndarray:
        assert raw_data.shape[0] == self.num_ray
        local_ogm = np.full((2,map_size,map_size), -1)
        center_index = map_size//2

        for ray in range(self.num_ray):
            # IF no obstacle, set all the cells to 0
            if raw_data[ray][1] == 0: 
                for i in range(1,len(self.rr[ray])):
                    x = self.rr[ray][i] + center_index
                    y = self.cc[ray][i] + center_index
                    if 0 <= x < map_size and 0 <= y < map_size and local_ogm[0,x,y] == -1:
                        local_ogm[1,x,y] = 0
                continue

            for i in range(1,len(self.rr[ray])):
                x = self.rr[ray][i] + center_index
                y = self.cc[ray][i] + center_index
                if 0 <= x < map_size and 0 <= y < map_size:
                    if self.length[ray][i] >= raw_data[ray][0]:
                        local_ogm[0,x,y] = 1
                        local_ogm[1,x,y] = raw_data[ray][1]
                        # x = int(self.rr[ray][i-1] + (map_size-1)/2)
                        # y = int(self.cc[ray][i-1] + (map_size-1)/2)
                        # local_ogm[x,y] = 1
                        break
                    elif local_ogm[0,x,y] != 1:
                        local_ogm[0,x,y] = 0
                        local_ogm[1,x,y] = 0
                else:
                    break
        local_ogm[0] = np.clip(local_ogm[0], 0, 1)

        return local_ogm

def transform_ogm(lidar_data, past_vel_pos,x_odom, y_odom, theta_odom, map_size):
    
    seq_len, batch_size, num_points, _ = lidar_data.size()

    # Calculate initial angles from past_vel_pos
    initial_angles = past_vel_pos[..., 4]  # Assuming the 4th index is the angle in past_vel_pos

    # Compute Cartesian coordinates from polar coordinates
    distances = lidar_data[..., 0]
    
    
    angles = initial_angles + theta_odom  # Correct for current orientation

    distance_x = distances * torch.cos(angles)
    distance_y = distances * torch.sin(angles)

    # Apply translations
    distances_x = distance_x + x_odom
    distances_y = distance_y + y_odom
    
    # Calculate new distances from transformed coordinates
    new_distances = torch.sqrt(distances_x**2 + distances_y**2)
    
    # Update lidar data with new distances
    # print(lidar_data.shape)
    # print(lidar_data[..., 0].shape)
    # print(new_distances.shape)
    lidar_data[..., 0] = new_distances
    static_obst, dynamic_obst=convert_lidar_to_ogm(lidar_data,map_size)

    # Convert the transformed points to grid coordinates

    return static_obst, dynamic_obst
def transform_lidar(lidar_data, pos,x_odom, y_odom, theta_odom):
    
    seq_len, batch_size, num_points, _ = lidar_data.size()

    # Calculate initial angles from past_vel_pos
    initial_angles = pos[..., 4]  # Assuming the 4th index is the angle in past_vel_pos

    # Compute Cartesian coordinates from polar coordinates
    distances = lidar_data[..., 0]
    
    
    angles = initial_angles + theta_odom  # Correct for current orientation

    distance_x = distances * torch.cos(angles)
    distance_y = distances * torch.sin(angles)

    # Apply translations
    distances_x = distance_x + x_odom
    distances_y = distance_y + y_odom
    
    # Calculate new distances from transformed coordinates
    new_distances = torch.sqrt(distances_x**2 + distances_y**2)
    
    
    lidar_data[..., 0] = new_distances

    return lidar_data

def convert_lidar_to_ogm(lidar_data,map_size):
    
    seq_len,batch_size, num_ray, _ = lidar_data.shape
    center_index = map_size // 2
    local_ogm = torch.full((seq_len,batch_size, 2, map_size, map_size), -1, dtype=torch.float32, device=lidar_data.device)
    for s in range(seq_len):
        for b in range(batch_size):
            # Calculate angles and distances
            angles = torch.linspace(0, 2 * torch.pi, num_ray, device=lidar_data.device)
            distances = lidar_data[s,b,:, 0]
            labels = lidar_data[s,b,:, 1]
            

            # Calculate x and y indices
            x_indices = (distances * torch.cos(angles)).long() + center_index
            y_indices = (distances * torch.sin(angles)).long() + center_index

            # Filter indices that are within the map bounds
            mask = (x_indices >= 0) & (x_indices < map_size) & (y_indices >= 0) & (y_indices < map_size)
            x_indices = x_indices[mask]
            y_indices = y_indices[mask]
            labels = labels[mask]


            # Set occupancy and labels
            local_ogm[s,b,0, x_indices, y_indices] = 1  # Set occupied
            local_ogm[s,b,1, x_indices, y_indices] = labels  # Set labels

            # Set unvisited cells to 0 (free space)
            local_ogm[local_ogm == -1] = 0
    
    
    # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
    static_obst = torch.zeros(seq_len, batch_size, map_size, map_size,dtype=local_ogm.dtype)
    dynamic_obst = torch.zeros(seq_len, batch_size, map_size, map_size,dtype=local_ogm.dtype)

    # Extract the two channels from the input tensor
    channel_0 = local_ogm[:,:,0,:,:]  # Shape: (batch_size, 32, 32)
    channel_1 = local_ogm[:,:,0,:,:]  # Shape: (batch_size, 32, 32)

    static_obst = channel_0 * (channel_1 == 1)
    dynamic_obst = channel_0 * (channel_1 >= 2)

    
    return static_obst, dynamic_obst
def plot_ogm(ogm, filename):
    plt.figure(figsize=(6,6))
    plt.imshow(ogm[0].detach().cpu().numpy(), cmap='gray')  # Assuming the OGM is on GPU and single-channel
    plt.colorbar()
    plt.savefig(filename)  # Saves the image to a file
    plt.close()



def check_distance_above_range(all_positions, broadcast_range):
    #print(all_positions.shape)# robot_num seq_len bz 1 3or5
    robot_num, batch_size, info_len = all_positions.size()
    
    all_positions = all_positions.permute(1,0,2)
    if info_len == 5:
        all_positions = all_positions[:,:,[2,3,4]]
        #print(all_positions.shape)
    # Calculate pairwise distances for all robots across each batch and each sequence step
    dist = torch.zeros((batch_size, robot_num, robot_num), device=all_positions.device)
    for b in range(batch_size):
        
        dist[b] = torch.cdist(all_positions[b], all_positions[b])

    # Check if all distances are above the broadcast range
    distance_checks = dist > broadcast_range
    # Mask out the self-distance (diagonal elements) since they are zero and not relevant
    distance_checks[:, torch.arange(robot_num), torch.arange(robot_num)] = False

    return distance_checks
def get_trans_matrix_2X3(pred_pos_tensor, robot_index,robot_num):
    
    ref_pos = pred_pos_tensor[robot_index]  # Using the robot at robot_index as reference
    ref_x, ref_y, ref_theta = ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2]

    # Initialize a list to store transformation matrices
    transformation_matrices = []

    for i in range(robot_num):
        # Current robot's position and orientation
        curr_x, curr_y, curr_theta = pred_pos_tensor[i][:, 0], pred_pos_tensor[i][:, 1], pred_pos_tensor[i][:, 2]

        # Compute the relative translation and rotation
        dx = curr_x - ref_x
        dy = curr_y - ref_y
        dtheta = curr_theta - ref_theta

        # Calculate rotation matrix components
        cos = torch.cos(dtheta)
        sin = torch.sin(dtheta)

        # Assembling the transformation matrix (2x3)
        T = torch.zeros(pred_pos_tensor.shape[1], 2, 3)  # Shape: [batch_size, 2, 3]
        T[:, 0, 0] = cos
        T[:, 0, 1] = -sin
        T[:, 1, 0] = sin
        T[:, 1, 1] = cos
        T[:, 0, 2] = dx
        T[:, 1, 2] = dy

        transformation_matrices.append(T)

    return torch.stack(transformation_matrices)
def get_trans_matrix_1X3(pred_pos_tensor, robot_index,robot_num=3):
    ref_pos = pred_pos_tensor[robot_index]  # Using the robot at robot_index as reference
    ref_x, ref_y, ref_theta = ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2]

    # Initialize a list to store transformation matrices
    transformation_matrices = []

    for i in range(robot_num):
        # Current robot's position and orientation
        curr_x, curr_y, curr_theta = pred_pos_tensor[i][:, 0], pred_pos_tensor[i][:, 1], pred_pos_tensor[i][:, 2]

        # Compute the relative translation and rotation
        dx = curr_x - ref_x
        dy = curr_y - ref_y
        dtheta = curr_theta - ref_theta

        # Assembling the transformation matrix (2x3)
        T = torch.zeros(pred_pos_tensor.shape[1],3)  # Shape: [batch_size, 2, 3]
        T[:, 0] = dx
        T[:, 1] = dy
        T[:, 2] = dtheta
        
        
        transformation_matrices.append(T)

    return torch.stack(transformation_matrices)
    

                
def calculate_combined_ogm_gt(all_positions,lidar_deque, connected, robot_index,robot_num=3):
    lidar_list=[]
    for robot_index in range(robot_num):
        #print(min(9,len(intermediate_feat_deque[robot_index]['h'])-1))
        lidar_list.append(lidar_deque[robot_index][-10]) # 1ahead now
    lidar_tensor = torch.stack(lidar_list).squeeze(1) # 3 16 90 2
    new_lidar_tensor = lidar_tensor.clone()
    merge_lidar_tensor = lidar_tensor.clone()
    robot_num, seq_len, batch_size, _, info_len = all_positions.size()
    all_positions = all_positions[:,-1,:,:,:].unsqueeze(1)
    all_positions = all_positions.permute(2, 1, 0, 3, 4).reshape(batch_size, seq_len, robot_num, info_len)
    if info_len != 5:
        exit()
    pos_tensor= all_positions[:,:,:,[2,3,4]].squeeze(1).permute(1,0, 2)
    T=get_trans_matrix_1X3(pos_tensor, robot_index).cuda()
    # T 3 16 3
    _, batchsize, _ = T.size()
    for i in range(robot_num):
        
        initial_angles = pos_tensor[i,:, 2]  # Assuming the 4th index is the angle in past_vel_pos
        distances = lidar_tensor[i,:,:,0] # 16 90
        
        angles = initial_angles + T[i,:,2] #16
        angles_ = angles.unsqueeze(1)
        angles_ = angles_.repeat(1, 90)
        distance_x = distances[:,:] * torch.cos(angles_)
        distance_y = distances[:,:] * torch.sin(angles_)
        distance_x = distance_x + T[i,:,0].unsqueeze(1).repeat(1, 90)
        distance_y = distance_y + T[i,:,1].unsqueeze(1).repeat(1, 90)
        new_distances = torch.sqrt(distance_x**2 + distance_y**2)
        new_lidar_tensor[i,:,:,0] = new_distances

        for bz in range(batchsize):
            if i != robot_index and connected[bz,0,robot_index,i]:
                
                # print(lidar_tensor[robot_index,bz,:,:])
                # print(new_lidar_tensor[i,bz,:,:])
                merge_lidar_tensor[robot_index,bz,:,:] = merge_lidar_data(lidar_tensor[robot_index,bz,:,:],new_lidar_tensor[i,bz,:,:])
                #print(merge_lidar_tensor[robot_index,bz,:,:])
               
    # for i in range(robot_num):
    #     if connected[bz,0,robot_index,i]:
    #         print(robot_index,i)
    #         print(lidar_tensor[i,0,:,:])
    #         target_static_map,target_dynamic_map=convert_lidar_to_ogm(lidar_tensor[i,:,:,:].unsqueeze(0),32)
    #         target_ogm_in_one=(target_static_map+target_dynamic_map).clamp(0,1)
    #         plot_ogm(target_ogm_in_one[0],f'lidar_{i}_old.png')  
    #         #print(new_lidar_tensor[i,0,:,:])
    #         target_static_map,target_dynamic_map=convert_lidar_to_ogm(new_lidar_tensor[i,:,:,:].unsqueeze(0),32)
    #         target_ogm_in_one=(target_static_map+target_dynamic_map).clamp(0,1)
    #         plot_ogm(target_ogm_in_one[0],f'lidar_{i}.png')
    #         #print(lidar_tensor[robot_index,0,:,:])
    #         target_static_map,target_dynamic_map=convert_lidar_to_ogm(lidar_tensor[robot_index,:,:,:].unsqueeze(0),32)
    #         target_ogm_in_one=(target_static_map+target_dynamic_map).clamp(0,1)
    #         plot_ogm(target_ogm_in_one[0],f'lidar_{robot_index}.png')
    #         #print(merge_lidar_tensor[robot_index,0,:,:])
    #         target_static_map,target_dynamic_map = convert_lidar_to_ogm(merge_lidar_tensor[robot_index,:,:,:].unsqueeze(0),32)        
    #         target_ogm_in_one=(target_static_map+target_dynamic_map).clamp(0,1)
    #         plot_ogm(target_ogm_in_one[0],f'lidar_fusion_{i}_{robot_index}.png')
    #         exit()
    target_static_map,target_dynamic_map = convert_lidar_to_ogm(merge_lidar_tensor[robot_index,:,:,:].unsqueeze(0),32)  
    return target_static_map,target_dynamic_map


def merge_lidar_data(lidar1, lidar2):
    # Ensure that lidar1 and lidar2 are on the same device and have the same shape
    assert lidar1.shape == lidar2.shape
    assert lidar1.device == lidar2.device
   
    # Initialize the merged lidar data with the same shape and device
    merged_lidar = lidar1.clone()
    mask_new_obstacle=(lidar1[:, 1] == 0) & (lidar2[:, 1] != 0)
    
    merged_lidar[mask_new_obstacle] = lidar2[mask_new_obstacle]
    
    mask_both_inf = (lidar1[:, 0] == float('inf')) & (lidar2[:, 0] == float('inf'))
    merged_lidar[mask_both_inf] = lidar1[mask_both_inf]  # Could choose either since both are inf
   
    return merged_lidar
class Ogm_RNN(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(Ogm_RNN, self).__init__()
        
        # Store required arguments for the model
        self.infer = infer
        self.is_recurrent = True
        self.args=args

        self.seq_length = args.seq_length
        self.nminibatch = args.num_mini_batch
        
        robot_info_size = 9
        
        self.robot_info_embedding_size = args.robot_info_embedding_size
        self.ogm_embedding_size = args.ogm_embedding_size
        self.detected_embedding_size = args.detected_robots_info_embedding_size
        self.output_size = args.actor_critic_output_size
        self.nenv = args.num_processes
        self.max_detected_robots_nums = 2
        self.ogm_deque={}
        self.robot_vel_pos_deque={}
        self.lidar_deque={}
        self.ogm_deque_val={}
        self.intermediate_feat_deque={}
        #self.last_static_ogm=None
        self.last_ogm=None
        self.robot_num=3 #
        
        # Initialize the model
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.robot_linear_encoder = nn.Sequential(
            init_(nn.Linear(robot_info_size, 256)), 
            nn.ReLU(),
            init_(nn.Linear(256, self.robot_info_embedding_size)),
            nn.ReLU()
            )
        
        # self.ogm_encoder = nn.Sequential(
        #     #nn.MaxPool2d(2, 2),
        #     #nn.Flatten(),
        #     init_(nn.Linear(32*32, 1024)),
        #     nn.ReLU(),
        #     init_(nn.Linear(1024, self.ogm_embedding_size // 2)),
        #     nn.ReLU()
        # )
        self.ogm_encoder_2 = nn.Sequential(
            #nn.MaxPool2d(2, 2),
            #nn.Flatten(),
            init_(nn.Linear(32*32, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, self.ogm_embedding_size)),
            nn.ReLU()
        )

        self.dynamic_obst_encoder = nn.Sequential(
            nn.Flatten(),
            init_(nn.Linear(1024, 512)),
            nn.ReLU(),
            init_(nn.Linear(512, 512)),
            nn.ReLU(),
            init_(nn.Linear(512, self.ogm_embedding_size // 2)),
            nn.ReLU()
        )

        
        self.detected_robots_encoder = nn.RNN(4, self.detected_embedding_size, 2, batch_first=True)
        self.counter=0
        self.wmse=torch.tensor(0.0)
        self.ssim=torch.tensor(0.0)
        self.loss=0.
        self.kl_loss=0.
        self.ce_loss=0.
        # self.detected_robots_encoder = nn.Sequential(
        #     init_(nn.Linear(self.max_detected_robots_nums * 4, self.max_detected_robots_nums * 16)),
        #     nn.Tanh(),
        #     init_(nn.Linear(self.max_detected_robots_nums * 16, self.detected_embedding_size)),
        #     nn.Tanh()
        # )


        self.main_rnn = MainRNN(args)
        
        # Initialize the OGM predictor from SOGMP
        # self.predictor=RVAEP(input_channels=1,
        #           latent_dim=128,
        #           output_channels=1)
        self.encoder=RConvLSTM(input_channels=1,latent_dim=128,output_channels=1)
        self.predictor=RVAE(input_channels=1,latent_dim=128,output_channels=1)
        
        self.criterion_wmse = WeightedMSELoss()
        self.criterion_ce = nn.BCELoss(reduction='sum') #, weight=class_weights)
        opt_params = { LEARNING_RATE: 0.0001,
                   BETAS: (.9,0.999),
                   EPS: 1e-08,
                   WEIGHT_DECAY: .001 }
        #self.optimizer = Adam(self.predictor.parameters(), **opt_params)
        
        all_parameters = list(self.encoder.parameters()) + list(self.predictor.parameters())
        self.optimizer = Adam(all_parameters, **opt_params)

        self.actor = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())
        

        self.critic = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))
        



        

        # self.human_num = obs_space_dict['spatial_edges'].shape[0]

        # 
        # 
        # 

        # # Store required sizes
        # self.human_node_rnn_size = args.human_node_rnn_size
        # self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        # self.output_size = args.human_node_output_size

        # # Initialize the Node and Edge RNNs
        


        # # Initialize attention module
        # self.attn = EdgeAttention_M(args)
        # # self.feature_extractor = CustomCNN(obs_space_dict['spatial_edges'])

        

        
        

        

        # num_inputs = hidden_size = self.output_size




        
        
        # self.human_node_final_linear=init_(nn.Linear(self.output_size,2))

        # if self.args.use_self_attn:
        #     self.spatial_attn = SpatialEdgeSelfAttn(args)
        #     self.spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
        # else:
        #     self.spatial_linear = nn.Sequential(init_(nn.Linear(obs_space_dict['spatial_edges'].shape[1], 128)), nn.ReLU(),
        #                                         init_(nn.Linear(128, 256)), nn.ReLU())


        # self.temporal_edges = [0]

    
    # convert the 2 channel occupancy map to 3 channel bitmap
    def convert_to_channel_bitmap(self, input_tensor):
        # Input tensor shape: (batch_size, 2, 32, 32)
        batch_size, _, height, width = input_tensor.shape

        # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
        static_obst = torch.zeros(batch_size, 1, height, width, dtype=input_tensor.dtype)
        dynamic_obst = torch.zeros(batch_size, 1, height, width, dtype=input_tensor.dtype)

        # Extract the two channels from the input tensor
        channel_0 = input_tensor[:, 0]  # Shape: (batch_size, 32, 32)
        channel_1 = input_tensor[:, 1]  # Shape: (batch_size, 32, 32)

        static_obst = channel_0 * (channel_1 == 1)
        dynamic_obst = channel_0 * (channel_1 >= 2)
        return static_obst, dynamic_obst
    def update_occupancy_map(self, robot_index, new_map):
        # Check if the robot's deque exists, create it if not
        if robot_index not in self.ogm_deque:
            self.ogm_deque[robot_index] = deque(maxlen=20)
        # Append new occupancy map to the deque
        self.ogm_deque[robot_index].append(new_map)
        if robot_index not in self.ogm_deque_val:
            self.ogm_deque_val[robot_index] = deque(maxlen=20)
        # Append new occupancy map to the deque
        self.ogm_deque_val[robot_index].append(new_map)
    
    def update_robot_vel_pos(self, robot_index, new_info):
        # Extracting vx, vy, angular, x, y, theta from new_info
        new_vel_pos = new_info[:,:,:,[7,8,6,0,1,6]]  # vx, vy, theta, x, y, theta 
        
        # Compute the norm of the velocity vector (vx, vy)
        v = torch.norm(new_vel_pos[:,:,:,:2], dim=-1, keepdim=True)
        new_vel_pos = torch.cat([v, new_vel_pos[:,:,:,2:]], dim=-1)
        
        # Initialize the deque if not already done
        if robot_index not in self.robot_vel_pos_deque:
            self.robot_vel_pos_deque[robot_index] = deque(maxlen=20)
        
        # Check if the robot is stationary, clear the deques if true
        if torch.all(v == 0) and (len(self.robot_vel_pos_deque[robot_index]))>=1: #happen when reset? or crash? should be reset?
            self.robot_vel_pos_deque[robot_index].clear()
            if hasattr(self, 'lidar_deque') and (robot_index in self.lidar_deque):  # Check if lidar_deque exists
                self.lidar_deque[robot_index].clear()
            if hasattr(self, 'intermediate_feat_deque') and (robot_index in self.intermediate_feat_deque):
                self.intermediate_feat_deque.clear() # 
                
        else:
            # If deque is not empty, calculate the angular change
            if self.robot_vel_pos_deque[robot_index]:
                # Access the last element's theta value (last, :, :, 5)
                last_theta = self.robot_vel_pos_deque[robot_index][-1][:,:,:,-1]
                current_theta = new_vel_pos[:,:,:,-1]
                # Calculate angular difference
                delta_theta = (current_theta - last_theta + np.pi) % (2 * np.pi) - np.pi
                new_vel_pos[:,:,:,-1] = delta_theta
                #print("Angular change:", delta_theta)  # Print or handle the angular change as needed
                
            # Append the new velocity position to the deque
        self.robot_vel_pos_deque[robot_index].append(new_vel_pos)
        #self.robot_vel_pos_deque[robot_index].append(new_vel_pos)
        # if robot_index==0:
        #     print(new_vel_pos)
    def update_lidar(self, robot_index, new_lidar):
        if robot_index not in self.lidar_deque:
            self.lidar_deque[robot_index] = deque(maxlen=20)
        # Append new occupancy map to the deque
        self.lidar_deque[robot_index].append(new_lidar)
        #check if clear the deque
    def update_intermediate_feat(self, h_enc,pred_pos,ogm_in_one,robot_index):
        if robot_index not in self.intermediate_feat_deque:
            self.intermediate_feat_deque[robot_index] = {'h': None, 'pred_pos': None,'ogm_for_vis':None}
            self.intermediate_feat_deque[robot_index]['h'] = deque(maxlen=20)
            self.intermediate_feat_deque[robot_index]['pred_pos'] = deque(maxlen=20)
            #self.intermediate_feat_deque[robot_index]['ogm_for_vis'] = deque(maxlen=20)
        self.intermediate_feat_deque[robot_index]['h'].append(h_enc)
        self.intermediate_feat_deque[robot_index]['pred_pos'].append(pred_pos)
        #self.intermediate_feat_deque[robot_index]['ogm_for_vis'].append(ogm_in_one)
        
        #check if clear the deque
    def check_deque_not_empty(self):
        
        
        for r in range(self.robot_num):
            if r not in self.lidar_deque or r not in self.robot_vel_pos_deque:
                return False
            if len(self.lidar_deque[r]) == 1 or len(self.robot_vel_pos_deque[r]) == 1:
                return False
            
        return True
    def check_deque_long_enough(self):

        for r in range(self.robot_num):
            if r not in self.lidar_deque or r not in self.robot_vel_pos_deque:
                return False
            if len(self.lidar_deque[r]) < 11 or len(self.robot_vel_pos_deque[r]) <11:
                return False
            
        return True
    def check_intermediate_feat(self,robot_num=3): # 
        """ Check if all deques have the same length and at least one element. """
        length=[]
        for robot_index in range(robot_num):
            if robot_index not in self.intermediate_feat_deque:
                return False
            length.append(len(self.intermediate_feat_deque[robot_index]['pred_pos']))
            length.append(len(self.intermediate_feat_deque[robot_index]['h']))
        
        return min(length) > 0 and len(set(length)) == 1
    def calculate_combined_h(self,h_enc, pos,robot_index):
        """ Calculate the mean hidden state for the current robot based on connections. """
        # h_enc_list=[]
        # pred_pos_list=[]
        # for robot_index in range(robot_num):
        #     #print(min(9,len(intermediate_feat_deque[robot_index]['h'])-1))
        #     h_enc_list.append(self.intermediate_feat_deque[robot_index]['h'][min(9,len(self.intermediate_feat_deque[robot_index]['h'])-1)])
        #     pred_pos_list.append(self.intermediate_feat_deque[robot_index]['pred_pos'][min(9,len(self.intermediate_feat_deque[robot_index]['h'])-1)])
        # h_enc_tensor = torch.stack(h_enc_list) # 3 16 32 32 32
        # # print(h_enc_tensor.shape)
        # pred_pos_tensor = torch.stack(pred_pos_list).squeeze(1).squeeze(2) # 3 16 3

        
        pos_=pos.squeeze(1).squeeze(2)[:,:,[2,3,4]]
        
        T=get_trans_matrix_2X3(pos_,robot_index,self.robot_num).cuda()
        
        # T 3 16 2 3
        _, batchsize, _, _ = T.size()
        connected=check_distance_above_range(pos_,broadcast_range=10)
        for i in range(self.robot_num):
            
            new_h_enc = warp_affine(h_enc[i,:,:,:,:], T[i], [32, 32])
            
            for bz in range(batchsize):
                if i != robot_index and connected[bz,robot_index,i]:
                    h_enc[i,bz,:,:,:] = new_h_enc[bz,:,:,:]
                    #add to h_enc_tensor[robot_index,bz,:,:,:] and calculate mean
        # Compute mean of updated hidden states where connected
        sum_h_enc = torch.zeros_like(h_enc[robot_index])
        count_h_enc = torch.zeros_like(h_enc[robot_index, :, 0, 0, 0])  # for averaging
        
        for i in range(self.robot_num):
            for bz in range(batchsize):
                if connected[bz, robot_index, i]:

                    sum_h_enc[bz] += h_enc[i, bz]
                    count_h_enc[bz] += 1

        # Avoid division by zero
        count_h_enc[count_h_enc == 0] = 1
        mean_h_enc = sum_h_enc / count_h_enc.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        return mean_h_enc
    def merge_all_lidar_data(self,lidar,target_pos_tensor,robot_index):
        lidar1=lidar[robot_index]
        merged_lidar = copy.deepcopy(lidar1)
        target_pos_tensor_=target_pos_tensor.squeeze(1).squeeze(2)
        robot_num, batchsize, info_len = target_pos_tensor_.size()
        connected=check_distance_above_range(target_pos_tensor_[:,:,[2,3,4]],broadcast_range=10)
        for bz in range(batchsize):
            for r in range(self.robot_num):
                if r != robot_index and connected[bz,robot_index,r]:
                    lidar2=lidar[r]
                    
                    assert lidar1.shape == lidar2.shape
                    assert lidar1.device == lidar2.device
                    # Initialize the merged lidar data with the same shape and device
                    
                    mask_new_obstacle=(lidar1[:,bz,:, 1] == 0) & (lidar2[:,bz,:, 1] != 0)
                    
                    merged_lidar[0,bz,:,:][mask_new_obstacle[0]] = lidar2[0,bz,:,:][mask_new_obstacle[0]]
                    
                    mask_both_inf = (lidar1[:,bz,:, 0] == float('inf')) & (lidar2[:,bz,:, 0] == float('inf'))
                    merged_lidar[0,bz,:,:][mask_both_inf[0]] = lidar1[0,bz,:,:][mask_both_inf[0]]  # Could choose either since both are inf
    
        return merged_lidar
    def forward(self, inputs, rnn_hxs, masks, robot_index,infer=False):
        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv
        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        # Extract the required inputs
        robot_info = reshapeT(inputs['robot_info'], seq_length, nenv)
        occupancy_map= reshapeT(inputs['occupancy_map'], seq_length, nenv)
        lidar=reshapeT(inputs['lidar'], seq_length, nenv)
        
        #version2 predict dynamic ogm only using SOGMP
                
        detected_robots_info = reshapeT(inputs['detected_robots_info'], seq_length, nenv)
        hidden_states_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        # Get the batch size, number of environments, number of channels, height and width of the occupancy map
        batch_size, num_envs, num_channels, height, width = occupancy_map.shape
        
        # encode the robot info
        robot_states = self.robot_linear_encoder(robot_info)
        
        # encode the occupancy map
        static_ogm , dynamic_ogm = self.convert_to_channel_bitmap(occupancy_map.view(batch_size*num_envs, num_channels, height, width))
        

        ogm_in_one=(static_ogm+dynamic_ogm).clamp(0,1)
        
        import copy
        occupancy_map_for_vis=copy.deepcopy(occupancy_map)
        
        
        # #SOGMP only predict dynamic
        # if infer:
            
        #     self.update_robot_vel_pos(robot_index, robot_info)
        #     self.update_lidar(robot_index, lidar)
            
        #     current_robot_vel_pos_list=[] # 1ahead now
        #     current_lidar_list=[] # 1ahead now
        #     prediction_list=[]
            
        #     for i in range(1): # 10: prediction steps
        #         h_enc_list=[]
        #         pos_list=[]  
        #         robot_vel_pos_list=[torch.tensor(item).clone().detach() for item in self.robot_vel_pos_deque[robot_index]]
        #         lidar_list=[torch.tensor(item).clone().detach() for item in self.lidar_deque[robot_index]]
        #         current_robot_vel_pos_list=robot_vel_pos_list[len(prediction_list):10]
        #         current_lidar_list=lidar_list[len(prediction_list):10]
                
        #         if self.check_deque_not_empty():
                    
        #             all_robot_vel_pos_list=[[torch.tensor(item).clone().detach() for item in self.robot_vel_pos_deque[r]] for r in range(self.robot_num)]
        #             all_lidar_list=[[torch.tensor(item).clone().detach() for item in self.lidar_deque[r]] for r in range(self.robot_num)]
        #             local_static_map=None
        #             # if len(prediction_list)>=1:
        #             #     for item in prediction_list:
        #             #         current_ogm_list.append(item)
        #             #past_ogm=torch.cat(current_ogm_list[:10],dim=0) # seq_len b 1 32 32
        #             for r in range(self.robot_num):
        #                 if r!=robot_index:
        #                     ind=len(current_robot_vel_pos_list)-1
                            
        #                     other_vel_pos_list=all_robot_vel_pos_list[r][len(prediction_list):ind]
        #                     other_lidar_list=all_lidar_list[r][len(prediction_list):ind]
        #                     if other_vel_pos_list:
        #                         past_robot_vel_pos=torch.cat(other_vel_pos_list,dim=0)
        #                         past_lidar=torch.cat(other_lidar_list,dim=0)
        #                         current_pos=current_robot_vel_pos_list[-1]
                                
        #                         x_odom, y_odom, theta_odom =  get_transform_coordinate(past_robot_vel_pos, current_pos)
        #                         static_map,dynamic_map=transform_ogm(past_lidar, past_robot_vel_pos,x_odom, y_odom, theta_odom,map_size=32)
        #                         ogm_in_one_=(static_map+dynamic_map).clamp(0,1).unsqueeze(2)
        #                         h_enc,_=self.encoder(ogm_in_one_.permute(1,0,2,3,4),static_map[-1])
        #                         h_enc_list.append(h_enc)
        #                         pos_list.append(current_robot_vel_pos_list[-1])
        #                 else:
        #                     past_robot_vel_pos=torch.cat(current_robot_vel_pos_list[:10],dim=0)# seq_len b 1 9
        #                     past_lidar=torch.cat(current_lidar_list[:10],dim=0)# seq_len b 90 2
        #                     current_pos=past_robot_vel_pos[-1].unsqueeze(0)
                            
        #                     x_odom, y_odom, theta_odom =  get_transform_coordinate(past_robot_vel_pos, current_pos)
        #                     static_map,dynamic_map=transform_ogm(past_lidar, past_robot_vel_pos,x_odom, y_odom, theta_odom,map_size=32)
        #                     ogm_in_one_input=(static_map+dynamic_map).clamp(0,1).unsqueeze(2)
        #                     h_enc,local_static_map=self.encoder(ogm_in_one_input.permute(1,0,2,3,4),static_map[-1])
        #                     h_enc_list.append(h_enc)
        #                     pos_list.append(current_robot_vel_pos_list[-1])
        #             h_enc_tensor=torch.stack(h_enc_list)
        #             pos_tensor=torch.stack(pos_list)
        #             combined_h_enc=self.calculate_combined_h(h_enc_tensor, pos_tensor,robot_index)
        #             prediction,_=self.predictor(combined_h_enc,local_static_map)
                    
        #         else:
        #             past_robot_vel_pos=torch.cat(current_robot_vel_pos_list[:10],dim=0)
        #             past_lidar=torch.cat(current_lidar_list[:10],dim=0)
        #             current_pos=past_robot_vel_pos[-1].unsqueeze(0)
                    
        #             x_odom, y_odom, theta_odom = get_transform_coordinate(past_robot_vel_pos, current_pos)
        #             static_map,dynamic_map=transform_ogm(past_lidar, past_robot_vel_pos,x_odom, y_odom, theta_odom,map_size=32)
        #             ogm_in_one_input=(static_map+dynamic_map).clamp(0,1).unsqueeze(2)
        #             h_enc,local_static_map=self.encoder(ogm_in_one_input.permute(1,0,2,3,4),static_map[-1])
        #             prediction,_=self.predictor(h_enc,local_static_map)
                
        #         prediction_list.append(prediction.unsqueeze(0))
                
        #         if prediction_list:
        #             # Stack tensors along a new dimension
        #             prediction_list_add_current_ogm =prediction_list
        #             #prediction_list_add_current_ogm.append(ogm_in_one.unsqueeze(0).unsqueeze(2))
        #             stacked_predictions = torch.stack(prediction_list_add_current_ogm, dim=0)
        #             # Compute the mean across the newly created dimension
        #             combined_prediction = torch.mean(stacked_predictions, dim=0)
                    
        #         else:
        #             combined_prediction = None # Or an appropriate default tensor

                
        #         combined_prediction_=combined_prediction.squeeze(2)
        #         occupancy_map_for_vis[:,:,0,:,:]=combined_prediction_
        #         ogm_in_one=combined_prediction_
                
        # else:
            
        #     # Training time
        #     # update SOGMP
            
        #     current_robot_vel_pos_list=[] # 1ahead now
        #     current_lidar_list=[] # 1ahead now
        #     prediction_list=[]
            
        #     for i in range(1): # 10: prediction steps
        #         h_enc_list=[]
        #         pos_list=[]
        #         target_lidar_list=[]
        #         target_pos_list=[]
        #         robot_vel_pos_list=[torch.tensor(item).clone().detach() for item in self.robot_vel_pos_deque[robot_index]]
        #         lidar_list=[torch.tensor(item).clone().detach() for item in self.lidar_deque[robot_index]]
        #         current_robot_vel_pos_list=robot_vel_pos_list[len(prediction_list):10]
        #         current_lidar_list=lidar_list[len(prediction_list):11]
        #         #print(self.check_deque_long_enough())
        #         #print(len(current_robot_vel_pos_list),len(current_lidar_list))
        #         if self.check_deque_long_enough():
        #             kl_loss=torch.tensor(0.0).cuda()
        #             ce_loss=torch.tensor(0.0).cuda()
        #             all_robot_vel_pos_list=[[torch.tensor(item).clone().detach() for item in self.robot_vel_pos_deque[r]] for r in range(self.robot_num)]
        #             all_lidar_list=[[torch.tensor(item).clone().detach() for item in self.lidar_deque[r]] for r in range(self.robot_num)]
        #             local_static_map=None
        #             # if len(prediction_list)>=1:
        #             #     for item in prediction_list:
        #             #         current_ogm_list.append(item)
        #             #past_ogm=torch.cat(current_ogm_list[:10],dim=0) # seq_len b 1 32 32
        #             for r in range(self.robot_num):
        #                 if r!=robot_index:
        #                     ind=len(current_robot_vel_pos_list)-1
        #                     other_vel_pos_list=all_robot_vel_pos_list[r][len(prediction_list):ind]
        #                     other_lidar_list=all_lidar_list[r][len(prediction_list):ind]
        #                     if other_vel_pos_list:
        #                         past_robot_vel_pos=torch.cat(other_vel_pos_list,dim=0)
        #                         past_lidar=torch.cat(other_lidar_list,dim=0)
        #                         current_pos=current_robot_vel_pos_list[-1]
                                
        #                         x_odom, y_odom, theta_odom =  get_transform_coordinate(past_robot_vel_pos, current_pos)
        #                         static_map,dynamic_map=transform_ogm(past_lidar, past_robot_vel_pos,x_odom, y_odom, theta_odom,map_size=32)
        #                         ogm_in_one_=(static_map+dynamic_map).clamp(0,1).unsqueeze(2)
        #                         h_enc,_=self.encoder(ogm_in_one_.permute(1,0,2,3,4),static_map[-1])
        #                         h_enc_list.append(h_enc)
        #                         pos_list.append(current_robot_vel_pos_list[-1])
        #                     target_pos=all_robot_vel_pos_list[r][ind+1]
        #                     target_pos_list.append(target_pos)
        #                     target_lidar=all_lidar_list[r][ind+1]
        #                     target_lidar_list.append(target_lidar)

        #                 else:
        #                     past_robot_vel_pos=torch.cat(current_robot_vel_pos_list[:10],dim=0)# seq_len b 1 9
        #                     past_lidar=torch.cat(current_lidar_list[:10],dim=0)# seq_len b 90 2
        #                     static, dynamic = self.convert_(past_lidar[:,-1,:,:].unsqueeze(1))
        #                     plot_ogm((static+dynamic).clamp(0,1),'past_lidar.png')
        #                     current_pos=past_robot_vel_pos[-1].unsqueeze(0)
                            
        #                     x_odom, y_odom, theta_odom =  get_transform_coordinate(past_robot_vel_pos, current_pos)
        #                     static_map,dynamic_map=transform_ogm(past_lidar, past_robot_vel_pos,x_odom, y_odom, theta_odom,map_size=32)
        #                     ogm_in_one_input=(static_map+dynamic_map).clamp(0,1).unsqueeze(2)
        #                     h_enc,local_static_map=self.encoder(ogm_in_one_input.permute(1,0,2,3,4),static_map[-1])
        #                     h_enc_list.append(h_enc)
        #                     pos_list.append(current_robot_vel_pos_list[-1])

        #                     target_pos=robot_vel_pos_list[10]
        #                     target_pos_list.append(target_pos)
        #                     target_lidar=lidar_list[10]
        #                     target_lidar_list.append(target_lidar)
        #             h_enc_tensor=torch.stack(h_enc_list)
        #             pos_tensor=torch.stack(pos_list)
        #             combined_h_enc=self.calculate_combined_h(h_enc_tensor, pos_tensor,robot_index)
        #             prediction,kl_l=self.predictor(combined_h_enc,local_static_map)
                    
        #             target_pos_tensor=torch.stack(target_pos_list)
        #             target_lidar_tensor=torch.stack(target_lidar_list)
        #             new_target_lidar=target_lidar_tensor.clone()
        #             for r in range(self.robot_num):
        #                 x_odom, y_odom,theta_odom = get_transform_coordinate(target_pos_tensor[r], current_pos)
        #                 new_target_lidar[r]=transform_lidar(target_lidar_tensor[r], target_pos_tensor[r],x_odom, y_odom, theta_odom)
        #             new_target_lidar_=self.merge_all_lidar_data(new_target_lidar,target_pos_tensor,robot_index)
        #             target_static_map,target_dynamic_map = convert_lidar_to_ogm(new_target_lidar_,32)
        #             target_ogm_in_one=(target_static_map+target_dynamic_map).clamp(0,1)
                    
        #             prediction_list.append(prediction.unsqueeze(0))
                    
        #             if prediction_list:
        #                 # Stack tensors along a new dimension
        #                 prediction_list_add_current_ogm =prediction_list
        #                 #prediction_list_add_current_ogm.append(ogm_in_one.unsqueeze(0).unsqueeze(2))
        #                 stacked_predictions = torch.stack(prediction_list_add_current_ogm, dim=0)
        #                 # Compute the mean across the newly created dimension
        #                 combined_prediction = torch.mean(stacked_predictions, dim=0)
                        
        #             else:
        #                 combined_prediction = None # Or an appropriate default tensor
        #             #print(combined_prediction.shape,target_ogm_in_one.shape)
        #             ce_l = self.criterion_ce(combined_prediction, target_ogm_in_one.unsqueeze(2)).div(target_ogm_in_one.size(0))
        #             kl_loss+=kl_l
        #             ce_loss+=ce_l
        #             # for r in range(3):
        #             #     plot_ogm(self.intermediate_feat_deque[r]['ogm_for_vis'][-11][0][0],'ogm_{}.png'.format(r))
                        
        #             # plot_ogm(prediction[0],'prediction.png')
        #             # plot_ogm(target_ogm_in_one[0][0],f'target_ogm.png')
        #             # exit()
                
        #             loss = ce_loss + BETA*kl_loss
        #             self.optimizer.zero_grad()
        #             loss.backward(torch.ones_like(loss))
        #             self.optimizer.step()
        #             self.loss+=loss.item()
        #             self.kl_loss+=kl_loss.item()
        #             self.ce_loss+=ce_loss.item()
        #             weights=calculate_weights(target_ogm_in_one)
        #             wmse = self.criterion_wmse(combined_prediction,target_ogm_in_one,weights)
        #             if not torch.isnan(wmse).any():
        #                 self.wmse+=wmse.item()
                    
        #             self.ssim+=calculate_ssim(combined_prediction,target_ogm_in_one).item()
        #             self.counter+=1
                    
                    
        #             if self.counter%3==0:
        #                 print('Predictor Training Loss: {} KL Loss: {} CE Loss: {}'.format(self.loss/3,self.kl_loss/3,self.ce_loss/3))
        #                 print('Avg wmse {} Avg ssim {}'.format(self.wmse/3,self.ssim/3))
                        
        #                 wandb.log({"Loss":self.loss/3,'KL Loss':self.kl_loss/3,'CE Loss':self.ce_loss/3,'WMSE':self.wmse/3,'SSIM':self.ssim/3})
        #                 self.loss=0.
        #                 self.kl_loss=0.
        #                 self.ce_loss=0.
                        
        #                 self.ssim=torch.tensor(0.0)
        #                 self.wmse=torch.tensor(0.0)
                       
        #print(ogm_in_one.shape)
        ogm_in_one=ogm_in_one.reshape(batch_size*num_envs, 32*32)
        
        encoded_ogm = self.ogm_encoder_2(ogm_in_one)
        encoded_ogm = encoded_ogm.view(batch_size, num_envs, 1, -1)
        
        # encode the detected robots info
        # while detected_robots_info.shape[2] < self.max_detected_robots_nums:
        #     detected_robots_info = torch.cat([detected_robots_info, detected_robots_info[:,:,-1,:]], dim=2)
        detected_robots_info = detected_robots_info[:,:,:self.max_detected_robots_nums,:].view(batch_size*num_envs, self.max_detected_robots_nums,4)
        encoded_detected_robots_info , _ = self.detected_robots_encoder(detected_robots_info)
        
        encoded_detected_robots_info = encoded_detected_robots_info[:,-1,:].view(batch_size, num_envs, 1, -1)
        
        # three_channel_ogm[:,:1,:,:] = 0
        # tensor_to_map(self.ogm_encoder(three_channel_ogm).view(-1), (8,16), 'encoded_detected_robots_info.png')



        # Do a forward pass through customised GRU
        outputs, new_hidden_states = self.main_rnn(robot_states, encoded_ogm, encoded_detected_robots_info, hidden_states_RNNs, masks)
        # use the output to get the actor and critic values
        hidden_critic = self.critic(outputs[:, :, 0, :])
        output_actor = self.actor(outputs[:, :, 0, :])
        
        rnn_hxs['human_node_rnn'] = new_hidden_states
        
        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), output_actor.squeeze(0), rnn_hxs,occupancy_map_for_vis
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), output_actor.view(-1, self.output_size), rnn_hxs,occupancy_map_for_vis

        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        

        # to prevent errors in old models that does not have sort_humans argument
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float() # [seq_len, nenv, max_human_num]
            # if no human is detected (human_masks are all False, set the first human to True)
            human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask


        
        


        if self.args.no_cuda:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        else:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        

        # # attention modules
        # if self.args.sort_humans:
        #     # human-human attention
        #     if self.args.use_self_attn:
        #         spatial_attn_out=self.spatial_attn(spatial_edges, detected_human_num).view(seq_length, nenv, self.human_num, -1)
        #     else:
        #         spatial_attn_out = spatial_edges
        #     output_spatial = self.spatial_linear(spatial_attn_out)

        #     # robot-human attention
        #     hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)
        # else:
        #     # human-human attention
        #     if self.args.use_self_attn:
        #         spatial_attn_out = self.spatial_attn(spatial_edges, human_masks).view(seq_length, nenv, self.human_num, -1)
        #     else:
        #         spatial_attn_out = spatial_edges
        #     output_spatial = self.spatial_linear(spatial_attn_out)

        #     # robot-human attention
        #     hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, human_masks)
       
        
        #hidden_attn_weighted = self.feature_extractor(occupancy_map.view(occupancy_map.shape[0]*occupancy_map.shape[1],occupancy_map.shape[2],occupancy_map.shape[3],-1), robot_node.view(robot_node.shape[0]*robot_node.shape[1]*robot_node.shape[2],-1)[:,3:5])
        
        
        # #grey_feat=hidden_attn_weighted[0][0].view(16,16).cpu().numpy()
        # x = copy.deepcopy(hidden_attn_weighted).cpu().detach().numpy()
        # x= x[0][0].reshape(16,16)
        # import matplotlib.pyplot as plt
        # import os

        # # Create a directory to save the plots
        # save_dir = "feat_plots_0"
        # os.makedirs(save_dir, exist_ok=True)

        # # Iterate over each channel (dim=1) of the tensor and save the plot
        # height, width = x.shape
        # for i in range(1):
        #     plt.figure()  # Create a new figure for each plot
        #     plt.imshow(x, cmap='gray')  # Plot the i-th channel of the first sample in the batch
        #     plt.colorbar()  # Add a colorbar to see the intensity scale
        #     plt.title(f"Channel {i} Grayscale Plot")
            
        #     # Save the plot
        #     plot_path = os.path.join(save_dir, f"channel_{i}_plot.png")
        #     plt.savefig(plot_path)
        #     plt.close()  # Close the figure to free up memory

        # print(f"All channel plots saved in '{save_dir}' directory.")
        

        



       
        


        # Update the hidden and cell states
        all_hidden_states_node_RNNs = new_hidden_states
        outputs_return = outputs

        


        # x is the output and will be sent to actor and critic
        

