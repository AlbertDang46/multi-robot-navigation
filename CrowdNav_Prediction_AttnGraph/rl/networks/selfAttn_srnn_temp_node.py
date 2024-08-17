import torch.nn.functional as F
from .custom_cnn_full import CustomCNN
from .srnn_model import *

class SpatialEdgeSelfAttn(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """
    def __init__(self, args):
        super(SpatialEdgeSelfAttn, self).__init__()
        self.args = args

        # Store required sizes
        # todo: hard-coded for now
        # with human displacement: + 2
        # pred 4 steps + disp: 12
        # pred 4 steps + no disp: 10
        # pred 5 steps + no disp: 12
        # pred 5 steps + no disp + probR: 17
        # Gaussian pred 5 steps + no disp: 27
        # pred 8 steps + no disp: 18
        if args.env_name in ['CrowdSimPred-v0', 'CrowdSimPredRealGST-v0']:
            self.input_size = 12
        elif args.env_name == 'CrowdSimVarNum-v0':
            self.input_size = 2 # 4
        else:
            raise NotImplementedError
        self.num_attn_heads=8
        self.attn_size=512
        

        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(nn.Linear(self.input_size, 128), nn.ReLU(),
                                             nn.Linear(128, self.attn_size), nn.ReLU()
                                             )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        self.multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)


    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len*nenv, max_human_num+1).cuda()
        mask[torch.arange(seq_len*nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        return mask


    def forward(self, inp, each_seq_len):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        '''
        # inp is padded sequence [seq_len, nenv, max_human_num, 2]
        
        seq_len, nenv,max_human_num, _ = inp.size()
        if self.args.sort_humans:
            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
            attn_mask = attn_mask.squeeze(1)  # if we use pytorch builtin function
        else:
            # combine the first two dimensions
            attn_mask = each_seq_len.reshape(seq_len*nenv, max_human_num)


        input_emb=self.embedding_layer(inp).view(seq_len*nenv, max_human_num, -1)
        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # if we use pytorch builtin function, v1.7.0 has no batch first option
        q=self.q_linear(input_emb)
        k=self.k_linear(input_emb)
        v=self.v_linear(input_emb)

        #z=self.multihead_attn(q, k, v, mask=attn_mask)
        z,_=self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function
        z=torch.transpose(z, dim0=0, dim1=1) # if we use pytorch builtin function
        return z

class EdgeAttention_M(nn.Module):
    '''
    Class for the robot-human attention module
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention_M, self).__init__()

        self.args = args

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size



        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer=nn.ModuleList()
        self.spatial_edge_layer=nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))



        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        attn = temporal_embed * spatial_embed
        attn = torch.sum(attn, dim=3)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # print(attn[0, 0, 0].cpu().numpy())

        # Compute weighted value
        # weighted_value = torch.mv(torch.t(h_spatials), attn)

        # reshape h_spatials and attn
        # shape[0] = seq_len, shape[1] = num of spatial edges (6*5 = 30), shape[2] = 256
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2,
                                                                                         1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]
        weighted_value = torch.bmm(h_spatials, attn)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn



    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, h_spatials, each_seq_len):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        each_seq_len:
            if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
            else, it is the mask itself
        '''
        seq_len, nenv, max_human_num, _ = h_spatials.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)

            if self.args.sort_humans:
                attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
                attn_mask = attn_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            else:
                attn_mask = each_seq_len
            weighted_value,attn=self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]

class EndRNN(RNNBase):
    '''
    Class for the GRU
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN, self).__init__(args, edge=False)

        self.args = args

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size # 64
        
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size # 256

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)


        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)



    def forward(self, robot_s, h_spatial_other, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.relu(self.encoder_linear(robot_s))
        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)
        
        x, h_new = self._forward_gru(concat_encoded, h, masks)

        outputs = self.output_linear(x)


        return outputs, h_new

class selfAttn_merge_SRNN(nn.Module):
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
        super(selfAttn_merge_SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args=args

        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNN(args)


        # Initialize attention module
        self.attn = EdgeAttention_M(args)
        # self.feature_extractor = CustomCNN(obs_space_dict['spatial_edges'])

        

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())


        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU()) # todo: check dim
        self.human_node_final_linear=init_(nn.Linear(self.output_size,2))

        if self.args.use_self_attn:
            self.spatial_attn = SpatialEdgeSelfAttn(args)
            self.spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
        else:
            self.spatial_linear = nn.Sequential(init_(nn.Linear(obs_space_dict['spatial_edges'].shape[1], 128)), nn.ReLU(),
                                                init_(nn.Linear(128, 256)), nn.ReLU())


        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num+1)

        dummy_human_mask = [0] * self.human_num
        dummy_human_mask[0] = 1
        if self.args.no_cuda:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        else:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())

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
        #print(inputs, rnn_hxs, masks, infer)
        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch
        #print(inputs['robot_node'].shape)
        robot_node = reshapeT(inputs['robot_info'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        occupancy_map = reshapeT(inputs['occupancy_map'], seq_length, nenv)

        # to prevent errors in old models that does not have sort_humans argument
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float() # [seq_len, nenv, max_human_num]
            # if no human is detected (human_masks are all False, set the first human to True)
            human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask


        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)


        if self.args.no_cuda:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        else:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)

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
        three_channel_bitmap = self.convert_to_3_channel_bitmap(occupancy_map.view(occupancy_map.shape[0]*occupancy_map.shape[1],occupancy_map.shape[2],occupancy_map.shape[3],-1)).cuda()
        hidden_attn_weighted = self.feature_extractor(three_channel_bitmap)
        hidden_attn_weighted = hidden_attn_weighted.view(spatial_edges.shape[0],spatial_edges.shape[1],1,-1)
        
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
        

        



        # Do a forward pass through GRU
        outputs, h_nodes \
            = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)


        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs


        # x is the output and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs



class MainRNN(nn.Module):
    """
    The class for RNN with done masks
    """
    # edge: True -> edge RNN, False -> node RNN
    def __init__(self, args):
        super(MainRNN, self).__init__()
        self.args = args

        self.gru = nn.GRU(args.ogm_embedding_size + args.robot_info_embedding_size, args.rnn_hidden_size)
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
    
    def forward(self, robot_info, ogm, hidden_state, masks):
        x = torch.cat((robot_info, ogm), dim=-1)

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
        
        robot_info_size = 7

        self.ogm_embedding_size = args.ogm_embedding_size
        self.robot_info_embedding_size = args.robot_info_embedding_size
        self.output_size = args.actor_critic_output_size
        self.nenv = args.num_processes
        
        # Initialize the model
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        self.robot_linear_encoder = nn.Sequential(init_(nn.Linear(robot_info_size, self.robot_info_embedding_size)), nn.ReLU())
        
        self.ogm_encoder = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1),
            nn.Flatten(),
            init_(nn.Linear(3072, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, self.ogm_embedding_size)),
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

        robot_info = reshapeT(inputs['robot_info'], seq_length, nenv)
        occupancy_map = reshapeT(inputs['occupancy_map'], seq_length, nenv)        
        hidden_states_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        batch_size, num_envs, num_channels, height, width = occupancy_map.shape
        
        robot_states = self.robot_linear_encoder(robot_info)
        
        three_channel_ogm = self.convert_to_3_channel_bitmap(occupancy_map.view(batch_size*num_envs, num_channels, height, width)).cuda()
        encoded_ogm = self.ogm_encoder(three_channel_ogm)
        encoded_ogm = encoded_ogm.view(batch_size, num_envs, 1, -1)


        # Do a forward pass through customised GRU
        outputs, new_hidden_states = self.main_rnn(robot_states, encoded_ogm, hidden_states_RNNs, masks)

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
        

        



def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))