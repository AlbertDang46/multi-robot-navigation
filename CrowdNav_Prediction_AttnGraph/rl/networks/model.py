import torch.nn as nn

from rl.networks.distributions import Bernoulli, Categorical, DiagGaussian
from .srnn_model import SRNN
from .ogm_rnn import Ogm_RNN

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    """ Class for a robot policy network """
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        # if base == 'srnn':
        #     base=SRNN
        # elif base == 'selfAttn_merge_srnn':
        #     base = selfAttn_merge_SRNN
        # else:
        #     raise NotImplementedError
        base = Ogm_RNN

        self.base = base(obs_shape, base_kwargs)
        self.srnn = True

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, robot_index,deterministic=False):
        if not hasattr(self, 'srnn'):
            self.srnn = False
        if self.srnn:
            #enter here
            value, actor_features, rnn_hxs,ogm_for_vis= self.base(inputs, rnn_hxs, masks, robot_index,infer=True)

        else:
           
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs,ogm_for_vis

    def get_value(self, inputs, rnn_hxs, masks,robot_index):

        value, _, _ ,_= self.base(inputs, rnn_hxs, masks, robot_index,infer=True)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, robot_index,action):
        #print("inputs",inputs)
        value, actor_features, rnn_hxs,_= self.base(inputs, rnn_hxs, masks,robot_index)
        #print("actor_features",actor_features)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



