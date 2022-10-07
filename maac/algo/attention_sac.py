import itertools
import torch
import torch.nn as nn
from torch.optim import Adam
from maac.utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from maac.utils.agents import AttentionAgent
from maac.utils.critic import AttentionCritic
from typing import List, Tuple


class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent task
    """

    def __init__(self, agent_init_params,
                 sa_size: List[Tuple[int, int]],
                 gamma: float = 0.99,
                 tau: float = 0.008,
                 alpha: float = 0.2,
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.001,
                 actor_hidden_dim: Tuple = (400, 300),
                 critic_hidden_dim: int = 256,
                 attend_heads: int = 4,
                 **kwargs):
        """
        Inputs:
        """
        self.num_agents = len(sa_size)
        self.agent_init_params = agent_init_params
        self.agents = [AttentionAgent(lr=actor_lr,
                                      hidden_dim=actor_hidden_dim,
                                      **params)
                       for params in agent_init_params]
        self.critic1 = AttentionCritic(sa_size,
                                       hidden_dim=critic_hidden_dim,
                                       attend_heads=attend_heads)
        self.target_critic1 = AttentionCritic(sa_size,
                                              hidden_dim=critic_hidden_dim,
                                              attend_heads=attend_heads)
        hard_update(self.target_critic1, self.critic1)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        self.critic2 = AttentionCritic(sa_size,
                                       hidden_dim=critic_hidden_dim,
                                       attend_heads=attend_heads)
        self.target_critic2 = AttentionCritic(sa_size,
                                              hidden_dim=critic_hidden_dim,
                                              attend_heads=attend_heads)
        hard_update(self.target_critic2, self.critic2)
        for p in self.target_critic2.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.critic1.parameters(), self.critic2.parameters())
        self.critic_optimizer = Adam(self.q_params, lr=critic_lr, weight_decay=1e-3)
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        # Optimizers/Loss using the Huber loss
        self.soft_q_criterion = nn.SmoothL1Loss()
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

    def step(self, observations, encoder, explore=False, deterministic=False):
        """
        Each agent takes a step in the environment
        :param observations:
        :param encoder:
        :param explore:
        :param deterministic
        :return:
        """
        return [a.step(obs, a.action_spaces, e, explore=explore, deterministic=deterministic)
                for a, e, obs in zip(self.agents, encoder.values(), observations)]

    def update_critics(self, samples):
        """
        Update central critic for all agents
        :param samples:
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
        obs, acts, rews, next_obs, dones = zip(*samples)

        # Q-value
        with torch.autograd.set_detect_anomaly(True):
            critic_in = list(zip(obs, acts))  # acts are from the replay buffer
            critic_rets1 = self.critic1(critic_in, regularize=True)
            critic_rets2 = self.critic2(critic_in, regularize=True)

            # target-Q
            with torch.no_grad():
                next_acts, next_log_pis, _ = zip(
                    *[a.policy.sample(next_ob) for a, next_ob in zip(self.agents, next_obs)])
                trgt_critic_in = list(zip(next_obs, next_acts))  # next_acts come from agents' current policy net
                backups = torch.zeros(self.num_agents, len(samples[0][0]), 1)
                target_qs1 = self.target_critic1(trgt_critic_in)
                target_qs2 = self.target_critic2(trgt_critic_in)
                target_qs = [torch.min(target_q1, target_q2) for target_q1, target_q2 in zip(target_qs1, target_qs2)]
                for a_i, target_q, log_pi in zip(range(self.num_agents), target_qs, next_log_pis):
                    backups[a_i] = (rews[a_i].view(-1, 1) +
                                    self.gamma * (target_q - self.alpha * log_pi) * (1 - dones[a_i].view(-1, 1)))

            # compute Q loss
            loss_qs1 = torch.zeros(self.num_agents)
            loss_qs2 = torch.zeros(self.num_agents)
            loss_qs = torch.zeros(self.num_agents)
            for a_i, (critic_ret1, reg1), (critic_ret2, reg2), backup in zip(range(self.num_agents),
                                                                             critic_rets1, critic_rets2, backups):
                loss_qs1[a_i] = self.soft_q_criterion(critic_ret1, backup)
                loss_qs1[a_i] += reg1[0]
                loss_qs2[a_i] = self.soft_q_criterion(critic_ret2, backup)
                loss_qs2[a_i] += reg2[0]
                loss_qs[a_i] = loss_qs1[a_i] + loss_qs2[a_i]

            q_loss = torch.sum(loss_qs)

            q_loss.backward()
            self.critic1.scale_shared_grads()
            self.critic2.scale_shared_grads()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

    def update_policies(self, samples, **kwargs):
        """
        Update the policy network of each agent
        :param samples:
        :param kwargs:
        :return:
        """
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
        with torch.autograd.set_detect_anomaly(True):
            for a_i, pi, ob in zip(range(self.num_agents), self.policies, obs):
                curr_act, log_pi, _ = pi.sample(ob)
                samp_acts.append(curr_act)
                all_log_pis.append(log_pi)

            critic_in = list(zip(obs, samp_acts))
            # critic_rets1, critic_rets2, vals1, vals2 = [], [], [], []
            critic_rets1 = self.critic1(critic_in)
            critic_rets2 = self.critic2(critic_in)
            q_pis = [torch.min(critic_ret1, critic_ret2)
                     for critic_ret1, critic_ret2 in zip(critic_rets1, critic_rets2)]

            loss_pi = []
            for a_i, log_pi, q_pi in zip(range(self.num_agents), all_log_pis, q_pis):
                if len(log_pi.shape) == 1:
                    log_pi = log_pi.unsqueeze(dim=-1)
                loss_pi.append((self.alpha * log_pi - q_pi).mean())

                disable_gradients(self.critic1)
                disable_gradients(self.critic2)
                loss_pi[a_i].backward(retain_graph=True)
                enable_gradients(self.critic1)
                enable_gradients(self.critic2)

            for a_i in range(self.num_agents):
                curr_agent = self.agents[a_i]
                curr_agent.policy_optimizer.step()
                curr_agent.policy_optimizer.zero_grad()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent) using polyak
        """
        with torch.no_grad():
            soft_update(self.target_critic1, self.critic1, self.tau)
            soft_update(self.target_critic2, self.critic2, self.tau)
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
        :param state_dim
        :param buffer_length
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
                                      "action_spaces": act_space,
                                      "action_scaling_coef": 1.,
                                      "buffer_length": buffer_length})
        init_dict = {
            "sa_size": sa_size,
            "agent_init_params": agent_init_params
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
