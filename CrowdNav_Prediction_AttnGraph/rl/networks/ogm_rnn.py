import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np

from rl.networks.network_utils import init
from rl.networks.vector_visualize import tensor_to_map

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
        
        # Initialize the model
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.robot_linear_encoder = nn.Sequential(
            init_(nn.Linear(robot_info_size, 256)), 
            nn.ReLU(),
            init_(nn.Linear(256, self.robot_info_embedding_size)),
            nn.ReLU()
            )
        
        self.ogm_encoder = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1),
            nn.Flatten(),
            init_(nn.Linear(3072, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 512)),
            nn.ReLU(),
            init_(nn.Linear(512, self.ogm_embedding_size)),
        )

        self.detected_robots_encoder = nn.Sequential(
            init_(nn.Linear(self.max_detected_robots_nums * 4, self.max_detected_robots_nums * 16)),
            nn.Tanh(),
            init_(nn.Linear(self.max_detected_robots_nums * 16, self.detected_embedding_size)),
            nn.Tanh()
        )


        self.main_rnn = MainRNN(args)

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
    def convert_to_3_channel_bitmap(self, input_tensor):
        # Input tensor shape: (batch_size, 2, 32, 32)
        batch_size, _, height, width = input_tensor.shape

        # Initialize the output tensor with zeros, shape: (batch_size, 3, 32, 32)
        output_tensor = torch.zeros(batch_size, 3, height, width, dtype=input_tensor.dtype)

        # Extract the two channels from the input tensor
        channel_0 = input_tensor[:, 0]  # Shape: (batch_size, 32, 32)
        channel_1 = input_tensor[:, 1]  # Shape: (batch_size, 32, 32)

        output_tensor[:, 0] = channel_0 * (channel_1 == 2)
        output_tensor[:, 1] = channel_0 * (channel_1 == 1)
        output_tensor[:, 2] = channel_0 * (channel_1 == 3)
        return output_tensor

    def forward(self, inputs, rnn_hxs, masks, infer=False):
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
        occupancy_map = reshapeT(inputs['occupancy_map'], seq_length, nenv)
        detected_robots_info = reshapeT(inputs['detected_robots_info'], seq_length, nenv)
        hidden_states_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        # Get the batch size, number of environments, number of channels, height and width of the occupancy map
        batch_size, num_envs, num_channels, height, width = occupancy_map.shape
        
        # encode the robot info
        robot_states = self.robot_linear_encoder(robot_info)
        
        # encode the occupancy map
        three_channel_ogm = self.convert_to_3_channel_bitmap(occupancy_map.view(batch_size*num_envs, num_channels, height, width)).cuda()
        encoded_ogm = self.ogm_encoder(three_channel_ogm)
        encoded_ogm = encoded_ogm.view(batch_size, num_envs, 1, -1)

        # encode the detected robots info
        while detected_robots_info.shape[2] < self.max_detected_robots_nums:
            detected_robots_info = torch.cat([detected_robots_info, detected_robots_info[:,:,-1,:]], dim=2)
        detected_robots_info = detected_robots_info[:,:,:self.max_detected_robots_nums,:].view(batch_size*num_envs, -1)
        encoded_detected_robots_info = self.detected_robots_encoder(detected_robots_info)
        encoded_detected_robots_info = encoded_detected_robots_info.view(batch_size, num_envs, 1, -1)
        
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
            return self.critic_linear(hidden_critic).squeeze(0), output_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), output_actor.view(-1, self.output_size), rnn_hxs

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
        

