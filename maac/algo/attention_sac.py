import gym
import torch
import numpy as np
from torch.optim import Adam
from maac.utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from maac.utils.agents import AttentionAgent
from maac.utils.critic import AttentionCritic
from typing import List, Tuple

MSELoss = torch.nn.MSELoss()


class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent task
    """

    def __init__(self, agent_init_params,
                 sa_size: List[Tuple[int, int]],
                 gamma: float = 0.98,
                 tau: float = 0.01,
                 reward_scale: float = 5.,
                 actor_lr: float = 0.01,
                 critic_lr: float = 0.01,
                 actor_hidden_dim: int = 128,
                 critic_hidden_dim: int = 128,
                 attend_heads: int = 4,
                 **kwargs):
        """
        Inputs:
        """
        self.num_agents = len(sa_size)
        self.agents = [AttentionAgent(lr=actor_lr,
                                      norm_flag=0,
                                      hidden_dim=actor_hidden_dim,
                                      **params)
                       for params in agent_init_params]
        self.critic = AttentionCritic(sa_size,
                                      hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size,
                                             hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-3)
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Created algo - AttentionSAC ")

    @property
    def policies(self):
        """
        Get each policy network from each agent
        :return:
        """
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        """
        Get each target policy network from each agent
        :return:
        """
        return [a.target_policy for a in self.agents]

    def step(self, observations, encoder, explore=False):
        """
        Each agent takes a step in the environment
        :param observations:
        :param encoder:
        :param explore:
        :return:
        """
        return [a.step(obs, a.action_spaces, e, device=self.device, explore=explore)
                for a, e, obs in zip(self.agents, encoder.values(), observations)]

    def update_critics(self, samples, soft=True):
        """
        Update central critic for all agents
        :param samples:
        :param soft:
        :return:
        """
        for i in range(len(samples)):  # 9 agents, 9 samples
            state = samples[i][0]
            action = samples[i][1]
            reward = samples[i][2]
            next_state = samples[i][3]
            done = samples[i][4]

            if self.device.type == "cuda":
                state = torch.cuda.FloatTensor(state).to(self.device)
                next_state = torch.cuda.FloatTensor(next_state).to(self.device)
                action = torch.cuda.FloatTensor(action).to(self.device)
                reward = torch.cuda.FloatTensor(reward).unsqueeze(1).to(self.device)
                done = torch.cuda.FloatTensor(done).unsqueeze(1).to(self.device)
                samples[i] = (state, action, reward, next_state, done)
            else:
                state = torch.FloatTensor(state).to(self.device)
                next_state = torch.FloatTensor(next_state).to(self.device)
                action = torch.FloatTensor(action).to(self.device)
                reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
                done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
                samples[i] = (state, action, reward, next_state, done)

        # Q loss
        next_acts, next_log_pis = zip(*[a.update_critic(sample) for a, sample in zip(self.agents, samples)])

        obs, acts, rews, next_obs, dones = zip(*samples)
        trgt_critic_in = list(zip(next_obs, next_acts))
        critic_in = list(zip(obs, acts))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True)

        q_loss = 0
        for a_i, next_q, log_pi, (q, regs) in zip(range(self.num_agents), next_qs, next_log_pis, critic_rets):
            if len(log_pi.shape) == 1:
                log_pi = log_pi.unsqueeze(dim=-1)
            target_q = (rews[a_i].view(-1, 1) + self.gamma * next_q * (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(q, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), 10 * self.num_agents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

    def update_policies(self, samples, soft=True, **kwargs):
        for i in range(len(samples)):
            state = samples[i][0]
            action = samples[i][1]
            reward = samples[i][2]
            next_state = samples[i][3]
            done = samples[i][4]

            if self.device.type == "cuda":
                state = torch.cuda.FloatTensor(state).to(self.device)
                next_state = torch.cuda.FloatTensor(next_state).to(self.device)
                action = torch.cuda.FloatTensor(action).to(self.device)
                reward = torch.cuda.FloatTensor(reward).unsqueeze(1).to(self.device)
                done = torch.cuda.FloatTensor(done).unsqueeze(1).to(self.device)
                samples[i] = (state, action, reward, next_state, done)
            else:
                state = torch.FloatTensor(state).to(self.device)
                next_state = torch.FloatTensor(next_state).to(self.device)
                action = torch.FloatTensor(action).to(self.device)
                reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
                done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
                samples[i] = (state, action, reward, next_state, done)

        obs, acts, rews, next_obs, dones = zip(*samples)

        samp_acts = []
        all_log_pis = []
        for a_i, pi, ob in zip(range(self.num_agents), self.policies, obs):
            curr_act,  log_pi = pi(ob, with_logprob=True)
            samp_acts.append(curr_act)
            all_log_pis.append(log_pi)

        critic_in = list(zip(obs, samp_acts))
        critic_rets = self.critic(critic_in)

        for a_i, log_pi, q in zip(range(self.num_agents), all_log_pis, critic_rets):
            if len(log_pi.shape) == 1:
                log_pi = log_pi.unsqueeze(dim=-1)
            curr_agent = self.agents[a_i]
            if soft:
                loss_pi = (log_pi / self.reward_scale - q).mean()
            else:
                loss_pi = (-q).mean()

            disable_gradients(self.critic)
            loss_pi.backward()
            enable_gradients(self.critic)

            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def norm_buffer(self):
        """
        Normalize the replay buffer for each agent
        :return:
        """
        return [a.normalize_buffer() for a in self.agents]

    def add_to_buffer(self, encoder, observations, actions, rewards, next_observations, done):
        """
        Add the observation encoder, obs, act, rew, next_obs into the replay buffer of each agent.
        :param encoder:
        :param observations:
        :param actions:
        :param rewards:
        :param next_observations:
        :param done:
        :return:
        """
        [a.add_to_buffer(e, obs, act, rew, next_obs, done) for a, e, obs, act, rew, next_obs
         in zip(self.agents, encoder.values(), observations, actions, rewards, next_observations)]

    def replay_buffer_length(self):
        """
        Get the 1st agent's buffer length for simplicity.
        :return:
        """
        return len(self.agents[0].replay_buffer)

    def sample(self, batch_size):
        """
        Sample a batch of length batch_size from the replay buffer of each agent.
        :param batch_size:
        :return:
        """
        return [a.replay_buffer.sample(batch_size) for a in self.agents]

    @classmethod
    def init_from_env(cls, env, state_dim, buffer_length):
        """
        Instantiate instance of this class from CityLearn environment
        :param env:
        :return:
        """
        print("AttentionSAC initialized from environment")
        sa_size = []
        agent_init_params = []

        _, actions_spaces = env.get_state_action_spaces()
        state_dim_values = state_dim.values()

        for act_space, obs_space in zip(actions_spaces,
                                        state_dim_values):
            sa_size.append((obs_space, act_space.shape[0]))
            agent_init_params.append({"dim_in_actor": obs_space,
                                      "dim_out_actor": act_space.shape[0],
                                      "act_limit": act_space.high[0],
                                      "action_spaces": act_space,
                                      "buffer_length": buffer_length})
        init_dict = {
            "sa_size": sa_size,
            "agent_init_params": agent_init_params
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
