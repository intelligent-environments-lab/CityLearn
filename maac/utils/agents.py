import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from .policies import DummyPolicy, SquashedGaussianActor
from .buffer import ReplayBuffer


class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """

    def __init__(self,
                 dim_in_actor,
                 dim_out_actor,
                 act_limit,
                 norm_flag,
                 buffer_length,
                 action_spaces,
                 hidden_dim=(256, 256),
                 activation=F.leaky_relu,
                 reward_scaling=5.,
                 lr=0.01):
        """
        Inputs:
        :param dim_in_actor: number of dimensions for policy input
        :param dim_out_actor: number of dimensions for policy output
        :param hidden_dim:
        :param activation
        :param lr:
        """
        self.norm_mean = 0
        self.norm_std = 0
        self.r_norm_mean = 0
        self.r_norm_std = 0
        self.norm_flag = norm_flag
        self.buffer_length = buffer_length
        self.replay_buffer = ReplayBuffer(self.buffer_length)
        self.action_spaces = action_spaces
        self.reward_scaling = reward_scaling
        self.policy = SquashedGaussianActor(dim_in_actor, dim_out_actor, activation, hidden_dim, act_limit)
        self.target_policy = SquashedGaussianActor(dim_in_actor, dim_out_actor, activation, hidden_dim, act_limit)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        print("Attention agent created")

    def step(self,
             obs: torch.Tensor,
             action_spaces,
             encoder,
             device,
             explore: bool = False
             ) -> torch.Tensor:
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
        :param obs: Observations for this agent
        :param explore: Whether to sample or not
        Outputs:
        :return: action: Action for this agent
        """
        return self.policy.choose_action(obs, action_spaces, encoder, self.norm_mean, self.norm_std, device, explore)

    def update_critic(self, sample, soft=True):
        obs, acts, rews, next_obs, dones = sample
        # Q loss
        next_act = []
        next_log_pi = []
        pi = self.policy
        # in SAC, next action comes from current policy
        curr_next_act, curr_next_log_pi = pi(obs, with_logprob=True)

        return curr_next_act, curr_next_log_pi

    def normalize_buffer(self):
        if self.norm_flag == 0:
            # normalizing the states in replay buffer
            S = np.array([j[0] for j in self.replay_buffer.buffer])
            self.norm_mean = np.mean(S, axis=0)
            self.norm_std = np.std(S, axis=0) + 1e-5

            # normalizing the rewards in replay buffer
            R = np.array([j[2] for j in self.replay_buffer.buffer])
            self.r_norm_mean = np.mean(R)
            self.r_norm_std = np.std(R) / self.reward_scaling + 1e-5

            new_buffer = []
            for s, a, r, s2, dones in self.replay_buffer.buffer:
                s_buffer = np.hstack(((s - self.norm_mean) / self.norm_std).reshape(1, -1)[0])
                s2_buffer = np.hstack(((s2 - self.norm_mean) / self.norm_std).reshape(1, -1)[0])
                new_buffer.append(
                    (s_buffer, a, (r - self.r_norm_mean) / self.r_norm_std, s2_buffer, dones))

            self.replay_buffer.buffer = new_buffer
            self.norm_flag = 1

            return self.replay_buffer

    def add_to_buffer(self, encoder, state, act, reward, next_state, done):
        # Run once the regression model has been fitted. Normalize all the states using periodical normalization,
        # one-hot encoding, or -1, 1 scaling. It also removes states that are not necessary (solar radiation if
        # there are no solar PV panels).

        o = np.array([j for j in np.hstack(encoder * state) if j is not None])
        o2 = np.array([j for j in np.hstack(encoder * next_state) if j is not None])

        if self.norm_flag > 0:
            o = (o - self.norm_mean) / self.norm_std
            o2 = (o2 - self.norm_mean) / self.norm_std
            reward = (reward - self.r_norm_mean) / self.r_norm_std

        self.replay_buffer.push(o, act, reward, o2, done)
