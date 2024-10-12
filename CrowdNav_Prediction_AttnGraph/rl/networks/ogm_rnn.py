from collections import deque



from SOGMP_plus.scripts.model import RVAEP
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np

from rl.networks.network_utils import init
# from rl.networks.vector_visualize import tensor_to_map
# from crowd_sim.envs.utils.action import ActionRot, ActionXY

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
        self.in_channels = args.ogm_embedding_size + args.robot_info_embedding_size + args.detected_robots_info_embedding_size
        self.block1 = nn.Sequential(
            nn.Linear(self.in_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.in_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.in_channels),
            nn.ReLU()
        )
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
        x = x + self.block1(x)
        x = x + self.block2(x)
        
        output, new_hidden_state = self._forward_gru(x, hidden_state, masks)

        output = self.output_linear(output)
        return output, new_hidden_state
def plot_ogm(ogm, filename):
    plt.figure(figsize=(6,6))
    plt.imshow(ogm[0].detach().cpu().numpy(), cmap='gray')  # Assuming the OGM is on GPU and single-channel
    plt.colorbar()
    plt.savefig(filename)  # Saves the image to a file
    print(f"Saved {filename}")
    plt.show()
    plt.close()



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
def transform_ogm(lidar_data, pos,x_odom, y_odom, theta_odom, robot_index,map_size=32):
        # Calculate initial angles from past_vel_pos
        
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
        print(mask)
        exit()
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
            
            if r!=robot_index:
                # transform past to current
                
                #angles = torch.linspace(0,2*np.pi,num_ray).unsqueeze(0).unsqueeze(1).repeat(batch_size,seq_len,1).to(lidar_data.device)
                #theta_odom=theta_odom.unsqueeze(-1).repeat(1,1,1,num_ray)
                
                angles = angles + theta_odom[:,:,r,:]
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
                
                if not ((x_indices >= 0).all() & (x_indices < map_size).all() & \
                    (y_indices >= 0).all() & (y_indices < map_size).all() & \
                    ((torch.logical_and(current_labels == 0, labels > 0)).all() | (current_labels > labels).all())):
                    new_merged_ogm=local_ogm.clone()
                    continue

                new_merged_ogm=local_ogm.clone()
                # Get indices where the final_mask is True
                
                final_batch_indices, final_seq_indices, final_ray_indices = final_mask.nonzero(as_tuple=True)
                
                # Extract the indices that are valid and within bounds
                
                final_x_indices = x_indices[final_batch_indices,final_seq_indices,final_ray_indices]
                final_y_indices = y_indices[final_batch_indices,final_seq_indices,final_ray_indices]
                
                # Update local_ogm at the indices specified by the mask
                merged_ogm=local_ogm.clone()
                #plot_ogm(merged_ogm[0,-1,0,:,:].unsqueeze(0),'merged_ogm.png')
                merged_ogm[final_batch_indices, final_seq_indices, 0, final_x_indices, final_y_indices] = 1
                if len(valid_labels)!=0:
                    merged_ogm[final_batch_indices, final_seq_indices, 1, final_x_indices, final_y_indices] = valid_labels[final_ray_indices]
                
                new_merged_ogm=torch.where(distance_check,merged_ogm,local_ogm)
                local_ogm=new_merged_ogm
            
            
            
          

        # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
        static_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)
        dynamic_obst = torch.zeros(batch_size,seq_len, map_size, map_size,dtype=local_ogm.dtype)

        # Extract the two channels from the input tensor
        channel_0 = new_merged_ogm[:,:,0,:,:]  # Shape: (batch_size, 32, 32)
        channel_1 = new_merged_ogm[:,:,1,:,:]  # Shape: (batch_size, 32, 32)

        static_obst = channel_0 * (channel_1 == 1)
        dynamic_obst = channel_0 * (channel_1 >= 2)
        return static_obst,dynamic_obst
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
        
        #add param

        self.future_step=4
        self.fusion='middle'
        self.seq_len=4
        self.delay=1
        self.robot_num=3

        # add variation
        self.robot_vel_pos_deque={}
        self.lidar_deque={}


        # Initialize the model
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.robot_linear_encoder = nn.Sequential(
            init_(nn.Linear(robot_info_size, 256)), 
            nn.ReLU(),
            init_(nn.Linear(256, self.robot_info_embedding_size)),
            nn.ReLU()
            )
        
        self.ogm_encoder = nn.Sequential(
            nn.Flatten(),
            init_(nn.Linear(1024, 512)),
            nn.ReLU(),
            init_(nn.Linear(512, self.ogm_embedding_size // 2)),
            nn.ReLU()
        )
        # self.ogm_encoder_2= nn.Sequential(
        #     nn.Flatten(),
        #     init_(nn.Linear(32*32, 1024)),
        #     nn.ReLU(),
        #     init_(nn.Linear(1024, self.ogm_embedding_size)),
        #     nn.ReLU()
        # )
        

        self.dynamic_obst_encoder = nn.Sequential(
            nn.Flatten(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, self.ogm_embedding_size // 2)),
            nn.ReLU()
        )

        
        #self.detected_robots_encoder = nn.RNN(4, self.detected_embedding_size, 2, batch_first=True)
        
        self.detected_robots_encoder = nn.Sequential(
            init_(nn.Linear(self.max_detected_robots_nums * 4, self.detected_embedding_size)),
            nn.ReLU()
        )

        self.main_rnn = MainRNN(args)

        self.actor = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())
        

        self.critic = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))
        # predictor
        # self.predictor=RVAEP(input_channels=1,
        #           latent_dim=128,
        #           output_channels=1)
        # checkpoint = torch.load('mf_4FUTURE_30.pth') 
        # self.predictor.load_state_dict(checkpoint['model'])
        # self.predictor.eval()
    
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
    def update_robot_vel_pos(self, robot_index, new_info):

        new_vel_pos = new_info[:,:,:,[2,3,8,0,1,8]]  # vx, vy, theta, x, y, theta 
        
        v = torch.norm(new_vel_pos[:,:,:,:2], dim=-1, keepdim=True)
        new_vel_pos = torch.cat([v, new_vel_pos[:,:,:,2:]], dim=-1)

        if robot_index not in self.robot_vel_pos_deque:
            self.robot_vel_pos_deque[robot_index] = deque(maxlen=(self.seq_len+self.delay))
        
        if torch.all(v == 0) and (len(self.robot_vel_pos_deque[robot_index]))>=1:
            self.robot_vel_pos_deque[robot_index].clear()
            if hasattr(self, 'lidar_deque') and (robot_index in self.lidar_deque):
                self.lidar_deque[robot_index].clear()
         
        else:
            if self.robot_vel_pos_deque[robot_index]:
                
                last_theta = self.robot_vel_pos_deque[robot_index][-1][:,:,:,-1]
                current_theta = new_vel_pos[:,:,:,-1]
                
                delta_theta = (current_theta - last_theta + np.pi) % (2 * np.pi) - np.pi
                new_vel_pos[:,:,:,-1] = delta_theta
               
        self.robot_vel_pos_deque[robot_index].append(new_vel_pos)
       
    def update_lidar(self, robot_index, new_lidar):
        if robot_index not in self.lidar_deque:
            self.lidar_deque[robot_index] = deque(maxlen=(self.seq_len+self.delay))
        self.lidar_deque[robot_index].append(new_lidar)
    
    
    
    def check_deque_long_enough(self):

        for r in range(self.robot_num):
            if r not in self.lidar_deque or r not in self.robot_vel_pos_deque:
                return False
            if len(self.lidar_deque[r]) <(self.seq_len+self.delay) or len(self.robot_vel_pos_deque[r]) <(self.seq_len+self.delay):
                return False
            
        return True
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
        lidar=reshapeT(inputs['lidar'], seq_length, nenv)
        occupancy_map = reshapeT(inputs['occupancy_map'], seq_length, nenv)
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
        ogm_for_vis=ogm_in_one.clone()
        
        # if infer:
        #     self.update_robot_vel_pos(robot_index, robot_info)
        #     self.update_lidar(robot_index, lidar)
        #     #torch.Size([1, 1, 1, 9]) torch.Size([1, 1, 90, 2])
        #     if self.check_deque_long_enough():
                
        #         positions=[]
        #         lidars=[]
        #         for r in range(self.robot_num):
        #             if r == robot_index:
        #                 positions.append(torch.stack([torch.tensor(item)[:,:,:,[2,3,4]] for item in self.robot_vel_pos_deque[r]][self.delay:],dim=0).squeeze(1))
        #                 lidars.append(torch.stack([torch.tensor(item) for item in self.lidar_deque[r]][self.delay:],dim=0).squeeze(1))
                        
        #             else: 
                        
        #                 positions.append(torch.stack([torch.tensor(item)[:,:,:,[2,3,4]] for item in self.robot_vel_pos_deque[r]][:self.seq_len],dim=0).squeeze(1))
        #                 lidars.append(torch.stack([torch.tensor(item) for item in self.lidar_deque[r]][:self.seq_len],dim=0).squeeze(1))
                
        #         positions=torch.stack(positions,dim=0).squeeze(1).permute(2,1,0,3,4).squeeze(3) # b seq_l robot_num 3
        #         lidars=torch.stack(lidars,dim=0).squeeze(1).permute(2,1,0,3,4) # b seq_l robot_num 90 2
        #         _,_,robot_num,_=positions.shape
                
        #         prediction_list=[]
        #         if self.fusion=='no':
        #             pos=positions[:,:,robot_index].unsqueeze(2).repeat(1,1,robot_num,1)
        #             current_pos=positions[:,-1,robot_index].unsqueeze(1).repeat(1,robot_num,1)
        #             x_odom,y_odom,theta_odom=get_transform_coordinate(pos,current_pos)
        #             static_obst, dynamic_obst=transform_ogm(lidars, pos,x_odom, y_odom, theta_odom, robot_index,map_size=32)
        #             for t in range(self.future_step): 
        #                 past_dogm=dynamic_obst[:,len(prediction_list):self.seq_len,:,:] # b _ 32 32
        #                 current_static_obst=static_obst[:,-1,:,:]
        #                 dynamic_obst=past_dogm
        #                 if len(prediction_list)>0:
        #                     pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
        #                     pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
        #                     dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
        #                 prediction, _ = self.predictor(dynamic_obst,current_static_obst,fusion='no')
        #                 #plot_ogm(prediction[0],f'pred_{r}_{t}.png')
        #                 prediction_list.append(prediction)
        #         if self.fusion=='early':
        #             pos=positions[:,:,:]
        #             current_pos=positions[:,-1,robot_index].unsqueeze(1).repeat(1,robot_num,1)
        #             x_odom,y_odom,theta_odom=get_transform_coordinate(pos,current_pos)
        #             static_obst, dynamic_obst=transform_ogm_early_fusion(lidars, pos,x_odom, y_odom, theta_odom, robot_index,map_size=32)
        #             ogm_for_vis=(static_obst+dynamic_obst)[:,-2,:,:].clamp(0,1)
        #             for t in range(self.future_step): 
        #                 past_dogm=dynamic_obst[:,len(prediction_list):self.seq_len,:,:] # b _ 32 32
        #                 current_static_obst=static_obst[:,-1,:,:]
        #                 dynamic_obst=past_dogm
        #                 if len(prediction_list)>0:
        #                     pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
        #                     pred_dogm=torch.abs(pred_ogm-current_static_obst.unsqueeze(1))
        #                     dynamic_obst=torch.cat((past_dogm,pred_dogm),dim=1)
        #                 prediction, _ = self.predictor(dynamic_obst,current_static_obst,fusion='early')
                        
        #                 prediction_list.append(prediction)
                    
        #         if self.fusion=='middle':
                    
        #             dynamic_obst_list=[]
        #             current_static_obst_list=[]
        #             all_ogm_for_vis=torch.zeros_like(ogm_for_vis)
        #             for rv in range(robot_num):
        #                 pos=positions[:,:,rv].unsqueeze(2).repeat(1,1,robot_num,1)
        #                 current_pos=positions[:,-1,robot_index].unsqueeze(1).repeat(1,robot_num,1)
        #                 x_odom, y_odom, theta_odom= get_transform_coordinate(pos,current_pos)
                        
        #                 static_obst, dynamic_obst=transform_ogm(lidars, pos,x_odom, y_odom, theta_odom, rv,map_size=32)
                        
        #                 # if torch.sqrt((positions[0,-1,rv,0]-positions[0,-1,robot_index,0])**2+(positions[0,-1,rv,1]-positions[0,-1,robot_index,1])**2)<10:
        #                 #     print('add')
        #                 #     all_ogm_for_vis+=(static_obst+dynamic_obst)[:,-1,:,:].clamp(0,1)
        #                 dynamic_obst_list.append(dynamic_obst)
        #                 current_static_obst_list.append(static_obst[:,-1,:,:])
                    
                    
        #             dynamic_obst_tensor=torch.stack(dynamic_obst_list,dim=0).squeeze(1) #rob_num b seq_l 32 32
        #             current_static_obst_tensor=torch.stack(current_static_obst_list,dim=0).squeeze(1) #rob_num b seq_l 32 32
        #             if dynamic_obst_tensor.dim() == 4:
        #                 dynamic_obst_tensor=dynamic_obst_tensor.unsqueeze(1)
        #                 current_static_obst_tensor=current_static_obst_tensor.unsqueeze(1)
                        
        #             for t in range(self.future_step):   
        #                 past_dogm=dynamic_obst_tensor[r][:,len(prediction_list):self.seq_len,:,:]
                        
        #                 if len(prediction_list)>0:
        #                     pred_ogm=torch.stack(prediction_list,dim=1).squeeze(2)
        #                     pred_dogm=torch.abs(pred_ogm-current_static_obst_tensor[r].unsqueeze(1))
        #                     dynamic_obst_ego=torch.cat((past_dogm,pred_dogm),dim=1)
        #                     dynamic_obst_tensor[r]=dynamic_obst_ego
        #                 else:
        #                     dynamic_obst_tensor[r]=past_dogm
        #                 #positions[:,-1,:,:]
        #                 prediction, _ = self.predictor(dynamic_obst_tensor,current_static_obst_tensor,positions[:,-1,robot_index,:].unsqueeze(1).repeat(1,robot_num,1),robot_index,fusion='middle')
        #                 prediction_list.append(prediction)
        #         if self.fusion=='late':
        #             pass
                
        #         prediction_tensor=torch.stack(prediction_list,dim=1).squeeze(2)
        #         ogm_for_vis=prediction_tensor[:,-1,:,:]
        #         ogm_in_one=prediction_tensor[:,-1,:,:]
                

        # ogm_in_one=ogm_in_one.reshape(-1, 32*32)
        # encoded_ogm=self.ogm_encoder_2(ogm_in_one)
        #encoded_ogm = encoded_ogm.view(batch_size, num_envs, 1, -1)
        
        encoded_ogm = torch.cat([self.ogm_encoder(static_ogm), self.dynamic_obst_encoder(dynamic_ogm)], dim=-1)
        encoded_ogm = encoded_ogm.view(batch_size, num_envs, 1, -1)
        # #encode the detected robots info
        # while detected_robots_info.shape[2] < self.max_detected_robots_nums:
        #     detected_robots_info = torch.cat([detected_robots_info, detected_robots_info[:,:,-1,:]], dim=2)
        detected_robots_info = detected_robots_info[:,:,:self.max_detected_robots_nums,:].view(batch_size*num_envs, self.max_detected_robots_nums*4)
        encoded_detected_robots_info  = self.detected_robots_encoder(detected_robots_info)
        
        encoded_detected_robots_info = encoded_detected_robots_info.view(batch_size, num_envs, 1, -1)

        # Do a forward pass through customised GRU
        outputs, new_hidden_states = self.main_rnn(robot_states, encoded_ogm, encoded_detected_robots_info, hidden_states_RNNs, masks)
        # use the output to get the actor and critic values
        hidden_critic = self.critic(outputs[:, :, 0, :])
        output_actor = self.actor(outputs[:, :, 0, :])
        rnn_hxs['human_node_rnn'] = new_hidden_states
        
        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), output_actor.squeeze(0), rnn_hxs,ogm_for_vis,lidar,robot_info
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), output_actor.view(-1, self.output_size), rnn_hxs,ogm_for_vis,lidar,robot_info

        

