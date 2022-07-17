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
LOG_STD_MIN = -20
ACT_SCALE = 5.


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
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        # need to later refer to rl.py for normalization
        self.act_limit = act_limit
        self.nonlin = nonlin

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
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # compute logprob from Gaussian, and then apply correction for tanh squashing
            # note: the correction formula is a trick from the paper, a numerically-stable equivalent
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def choose_action(self, obs, action_spaces, encoder, norm_mean, norm_std, device, explore=True):
        if explore:
            act = ACT_SCALE * action_spaces.sample()
            return act
        else:
            state_normalizer = [norm_mean, norm_std]
            obs_ = np.array([j for j in np.hstack(encoder * obs.detach().numpy()) if j is not None])
            obs_ = normalize(obs_, state_normalizer)
            obs_ = torch.FloatTensor(obs_).unsqueeze(0).to(device)

            with torch.no_grad():
                act, _ = self.forward(obs_, explore, False)
                return act.numpy()
