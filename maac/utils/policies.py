import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from maac.utils.buffer import ReplayBuffer
from maac.utils.encoder import normalize


class DummyPolicy:
    """
    It's just a dummy policy
    """
    pass


LOG_STD_MAX = 2
LOG_STD_MIN = -10
ACT_SCALE = 0.5


class SquashedGaussianActor(nn.Module):
    """
    Base policy network
    Takes states as inputs, and outputs an action and the log prob
    """

    def __init__(self,
                 state_dim: int,
                 act_dim: int,
                 nonlin,
                 hidden_dim,
                 act_limit,
                 action_space,
                 action_scaling_coef,
                 init_w=3e-3
                 ):
        """
        Inputs:
            state_dim: Number of dimensions in state
            act_dim: Number of dimensions in action
            hidden_dim: Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(SquashedGaussianActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.mu_layer.weight.data.uniform_(-init_w, init_w)
        self.mu_layer.bias.data.uniform_(-init_w, init_w)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.bias.data.uniform_(-init_w, init_w)
        # need to later refer to rl.py for normalization
        self.act_limit = act_limit
        self.nonlin = nonlin
        self.reparam_noise = 1e-5

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.action_scale = torch.FloatTensor(
            action_scaling_coef * (action_space.high - action_space.low) / 2.)
        # self.action_bias = torch.FloatTensor(
        #     action_scaling_coef * (action_space.high + action_space.low) / 2.)

    def forward(self, obs, explore=True, with_logprob=False):
        """
        Inputs:
            obs (PyTorch Matrix): batch of observations
        Outputs:
            out (PyTorch Matrix): Actions
        """
        x = self.nonlin(self.fc1(obs))
        x = self.nonlin(self.fc2(x))

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if not explore:
            # only used for evaluating policy at test time
            # pi_action = torch.tanh(mu) * self.action_scale + self.action_bias
            pi_actions = mu
        else:
            pi_actions = pi_distribution.rsample()

        pi_action = torch.tanh(pi_actions) * torch.tensor(self.action_scale).to(self.device)

        if with_logprob:
            # compute logprob from Gaussian, and then apply correction for tanh squashing
            # note: the correction formula is a trick from the paper, a numerically-stable equivalent
            logp_pi = pi_distribution.log_prob(pi_actions)
            logp_pi -= torch.log(self.action_scale - pi_action.pow(2) + self.reparam_noise)
            logp_pi = logp_pi.sum(1, keepdim=True)
        else:
            logp_pi = None

        return pi_action, logp_pi

    def choose_action(self, obs, action_spaces, encoder, norm_mean, norm_std, device, explore=True):
        """
        Agent chooses an action to take a step. If explore, we just sample from the action spaces; if deterministic,
        we first normalize the states and pass through the policy net to get an action base on the network output
        :param obs:
        :param action_spaces:
        :param encoder:
        :param norm_mean:
        :param norm_std:
        :param device:
        :param explore:
        :return:
        """
        if explore:
            act = ACT_SCALE * action_spaces.sample()
            return act
        else:
            state_normalizer = [norm_mean, norm_std]
            obs_ = np.array([j for j in np.hstack(encoder * obs.detach().numpy()) if j is not None])
            obs_ = normalize(obs_, state_normalizer)
            obs_ = torch.FloatTensor(obs_).unsqueeze(0).to(device)

            with torch.no_grad():
                act, _ = self.forward(obs_, explore)
                return act.numpy()[0]
