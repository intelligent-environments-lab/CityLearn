import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from maac.utils.encoder import normalize


class SquashedGaussianActor(nn.Module):
    """
    Base policy network
    Takes states as inputs, and outputs an action and the log prob
    """

    def __init__(self,
                 state_dim: int,
                 act_dim: int,
                 action_space,
                 action_scaling_coef,
                 hidden_dim=None,
                 init_w=3e-2,
                 log_std_min=-10,
                 log_std_max=2,
                 reparam_noise=1e-5
                 ):
        """
        Inputs:
            state_dim: Number of dimensions in state
            act_dim: Number of dimensions in action
            hidden_dim: Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(SquashedGaussianActor, self).__init__()
        if hidden_dim is None:
            hidden_dim = [400, 300]

        self.linear1 = nn.Linear(state_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mu_layer = nn.Linear(hidden_dim[1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dim[1], act_dim)

        self.mu_layer.weight.data.uniform_(-init_w, init_w)
        self.mu_layer.bias.data.uniform_(-init_w, init_w)

        self.log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.bias.data.uniform_(-init_w, init_w)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.reparam_noise = reparam_noise

        self.action_scale = torch.FloatTensor(
            action_scaling_coef * (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            action_scaling_coef * (action_space.high + action_space.low) / 2.)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        """
        Inputs:
            obs (PyTorch Matrix): batch of observations
        Outputs:
            out (PyTorch Matrix): Actions
        """
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)

        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)

        # pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        x_t = pi_distribution.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_pi = pi_distribution.log_prob(x_t)
        log_pi -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.reparam_noise)
        log_pi = log_pi.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_pi, mu

    def choose_action(self, obs, action_spaces, encoder, norm_mean, norm_std, pca, explore=True, deterministic=False):
        """
        Agent chooses an action to take a step. If explore, we just sample from the action spaces; if deterministic,
        we first normalize the states and pass through the policy net to get an action base on the network output
        :param obs:
        :param action_spaces:
        :param encoder:
        :param norm_mean:
        :param norm_std:
        :param explore:
        :param deterministic:
        :return:
        """
        if explore:
            multiplier = 0.32
            hour_day = obs[2]
            a_dim = len(action_spaces.sample())

            act = [0.0 for _ in range(a_dim)]
            if 7 <= hour_day <= 11:
                act = [-0.05 * multiplier for _ in range(a_dim)]
            elif 12 <= hour_day <= 15:
                act = [-0.05 * multiplier for _ in range(a_dim)]
            elif 16 <= hour_day <= 18:
                act = [-0.11 * multiplier for _ in range(a_dim)]
            elif 19 <= hour_day <= 22:
                act = [-0.06 * multiplier for _ in range(a_dim)]

            # Early nighttime: store DHW and/or cooling energy
            if 23 <= hour_day <= 24:
                act = [0.085 * multiplier for _ in range(a_dim)]
            elif 1 <= hour_day <= 6:
                act = [0.1383 * multiplier for _ in range(a_dim)]
            return act
        else:
            state_normalizer = [norm_mean, norm_std]
            obs_ = np.array([j for j in np.hstack(encoder * obs.detach().numpy()) if j is not None])
            obs_ = normalize(obs_, state_normalizer)
            obs_ = pca.transform(obs_.reshape(1, -1))[0]
            obs_ = torch.FloatTensor(obs_).unsqueeze(0).to(self.device)

            if deterministic:
                _, _, act = self.sample(obs_)
            else:
                act, _, _ = self.sample(obs_)

            return act.detach().cpu().numpy()[0]
