#!/usr/bin/env python
# coding: utf-8

"""
Implementation of Soft Actor Critic (SAC) network
using PyTorch.
See https://arxiv.org/pdf/1801.01290.pdf for algorithm details.

@author: Anjukan Kathirgamanathan 2020 (k.anjukan@gmail.com) and Kacper
Twardowski (kanexer@gmail.com) 

Project for CityLearn Competition. 

Based on code from Pranjal Tandon (https://github.com/pranz24/pytorch-soft-actor-critic)
Implementation here was for MuJoCo environment.
"""

"""
###################################
IMPORT PACKAGES
======

"""
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import math
import random
import numpy as np

# NEED TO ADD COMMENTING FOR THESE PARAMS
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

'''
###################################
# Agent class - the agent explores the environment, collecting experiences and adding them to the Replay buffer. 
Initialises an Agent and Critic for each building. Can also be used to test/run a trained network in the environment.
======
'''
class SAC(object):
    def __init__(self, env, num_inputs, action_space, args, constrain_action_space=False):

        self.env = env

        """
        ###################################
        HYPERPARAMETERS
        ======
        SAC PARAMETERS
            replay_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            alpha (float): Temperature parameter α determines the relative importance of the entropy\
            term against the reward (default: 0.2)
            automatic_entropy_tuning (boolean): Automatically adjust α
            target_update_interval (int): Value target update per no. of updates per step
            lr (float): learning rate
            tau (float): target smoothing coefficient(τ) for soft update
            hidden_size (int): Size of the hidden layer in networks

        """
        self.lr = 0.0005
        self.gamma = 0.99
        self.tau = 0.003
        self.alpha = 0.2
        self.replay_size = 2000000
        self.batch_size = 1024
        self.automatic_entropy_tuning = False
        self.target_update_interval = 1
        self.hidden_size = 256
        
        # Number of regressive terms to include of HVAC cooling load
        self.autoregressive_size = 1
        num_inputs = num_inputs + self.autoregressive_size

        # Reward shaping weights
        self.ramping_factor = 0
        self.action_factor = 0
        self.smooth_factor = 0
        self.peak_factor = 1

        self.policy_type = args.policy

        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], self.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.reset_action_tracker()

        # Replay Memory
        self.memory = ReplayMemory(self.replay_size)

        # Memory of Power Consumption
        self.autoregressive_memory = AutoRegressiveMemory(self.autoregressive_size)
        
        # Memory of Power Consumption
        self.autoregressive_action_memory = AutoRegressiveMemory(1)

        # Should the action space be constrained to avoid infeasible actions
        self.constrain_action_space = constrain_action_space

        # Size of state space
        self.obs_size = [box.shape[0] for box in self.env.get_state_action_spaces()[0]]

        # Size of action space
        self.act_size = [2 if id not in ["Building_3","Building_4"] else 1 for id in self.env.building_ids]
        
        # Which buildings are being simulated
        self.building = []
        for building in ['Building_1',"Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]:
            if building in env.building_ids:
                self.building.append(1) 
            else:
                self.building.append(0)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], self.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], self.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

    def reset_action_tracker(self):
        self.action_tracker = []

    def select_action(self, state, evaluate=False):

        # Create modified state space with autoregressive terms
        for j in range(0,self.autoregressive_size):
            state = np.append(state, self.autoregressive_memory.buffer[-j-1])
        state_copy = state
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        
        action = action.detach().cpu().numpy()[0]

        # Constrain action space to feasible values only if set to True
        if self.constrain_action_space == True:
            ba_idx = 0 # building action index
            bs_idx = 1 # building state index
            for building in range(0,len(self.act_size)):
                bs_end_idx = bs_idx + self.obs_size[building] - 1
                #print("\nBuilding {}".format(building+1))
                # Boundary constraint flags, -1 for 0, 1 for 1, 0 for anything in between
                soc_flags = [0 for actsiz in range(self.act_size[building])]
                #print("States:")
                # Find constraints and set flags
                for idx, b_idx in enumerate(range(bs_end_idx-self.act_size[building],bs_end_idx)):
                    
                    #print(state_copy[b_idx])

                    # Enable the SOC flag on extreme values
                    if state_copy[b_idx] < 0.05:
                        soc_flags[idx] = -1
                    elif state_copy[b_idx] > 0.95:
                        soc_flags[idx] = 1

                # Set constraints from flags
                for idx, flag in enumerate(soc_flags):
                    
                    #print("Action: {}".format(action[ba_idx+idx]))

                    # SOC is trying to go below 0
                    if flag == -1:
                        if action[ba_idx+idx] < 0:
                            # print("Activated flag {} == -1".format(idx))
                            # action[ba_idx+idx] = np.random.normal(0,0.1,1)
                            action[ba_idx+idx] = 0

                    # SOC is trying to go above 1
                    elif flag == 1:
                        if action[ba_idx+idx] > 0:
                            # print("Activated flag {} == 1".format(idx))
                            # action[ba_idx+idx] = -np.random.normal(0,0.1,1)
                            action[ba_idx+idx] = 0
                    #print(action[ba_idx+idx])
                ba_idx += self.act_size[building]
                bs_idx = bs_end_idx

        self.action_tracker.append(action)

        return action

    def append(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.
        
        Params
        ======
            states (Array): Array of states for every building
            actions (Array): Array of actions for each building
            rewards (Array): Array of rewards for each building
            next_states (Array): Array of updates states for each building after actions
            dones (Boolean): Whether episode is done (terminated) or not
        """
        
        # Calculate HVAC load for cooling (sum of cooling + dhw)
        HVAC_load = self.env.electric_consumption_cooling[-1] + self.env.electric_consumption_dhw[-1]
        
        # Append autoregressive terms of HVAC load
        for j in range(0,self.autoregressive_size):
            states = np.append(states, self.autoregressive_memory.buffer[-j-1])
        for j in range(0,self.autoregressive_size-1):
            next_states = np.append(next_states, self.autoregressive_memory.buffer[-j-1])
        next_states = np.append(next_states,HVAC_load)
        
        # Calculate Ramping component of reward function
        ramping = abs(HVAC_load - self.autoregressive_memory.buffer[-1])
        
        # Calculate Similar action component of reward function (if all agents are doing the same thing penalize)
        same_action = abs(self.action_factor*actions.mean())
        
        # Smooth action reward function
        smooth_action = abs(actions - self.autoregressive_action_memory.buffer[-1]).sum()

        # Reward bonus if agent charges during the night
        if (1 <= states[0] < 8) and actions.mean() > 0:
            night_charging_boost = 0
        else:
            night_charging_boost = 0
        
        # Apply reward shaping function
        #print("\nPeak reward {}".format(self.peak_factor*rewards))
        #print("\nRamp reward {}".format(-self.ramping_factor*ramping))
        #print("\nNight Boost reward {}".format(night_charging_boost))
        #print("\nSmooth Action reward {}".format(-self.smooth_factor*smooth_action))
        # total_rewards = self.peak_factor*rewards - self.ramping_factor*ramping - self.action_factor*same_action - self.smooth_factor*smooth_action + night_charging_boost
        total_rewards = self.peak_factor*rewards - self.ramping_factor*ramping - self.smooth_factor*smooth_action
        #print("\Total reward {}".format(rewards))
        
        # Save experience / reward
        self.memory.push(states, actions, rewards, next_states, dones)
        
        # Add HVAC load to autoregressive memory
        self.autoregressive_memory.push(HVAC_load)
        
        # Add action to autoregressive memory
        self.autoregressive_action_memory.push(actions)

        # Return shaped reward values
        return total_rewards, self.peak_factor*rewards, -self.ramping_factor*ramping, night_charging_boost, -self.smooth_factor*smooth_action

    # Update policy parameters
    def update_parameters(self, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, path):
        if not os.path.exists(path+'/monitor/checkpoints/'):
            os.makedirs(path+'/monitor/checkpoints/')

        actor_path = path+"/monitor/checkpoints/sac_actor"
        critic_path = path+"/monitor/checkpoints/sac_critic"
        #print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # for reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class AutoRegressiveMemory:
    def __init__(self, capacity):
        self.buffer = [0] * capacity

    def push(self, previous_consumption):
        self.buffer.append(previous_consumption)

    def __len__(self):
        return len(self.buffer)

class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day = states[0][0]
        
        # Daytime: release stored energy
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 9 and hour_day <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        # Early nightime: store DHW and/or cooling energy
        if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
            a = []
            for i in range(len(self.actions_spaces)):
                if len(self.actions_spaces[i].sample()) == 2:
                    a.append([0.091, 0.091])
                else:
                    a.append([0.091])

        self.action_tracker.append(a)
        return np.array(a)