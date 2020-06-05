#!/usr/bin/env python
# coding: utf-8

"""
Implementation of Deep Deterministic Policy Gradients (DDPG) network
using PyTorch.
See https://arxiv.org/abs/1509.02971 for algorithm details.

@author: Anjukan Kathirgamanathan 2020 (k.anjukan@gmail.com) 

Project for CityLearn Competition. 

Based on code from Mike Richardson (https://github.com/xkiwilabs/Multi-Agent-DDPG-using-PTtorch-and-ML-Agents)
Implementation here was for Unity Tennis environment.
"""

"""
###################################
IMPORT PACKAGES
======

"""
import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
###################################
HYPERPARAMETERS
======
DDPG PARAMETERS
        BUFFER_SIZE (int): replay buffer size
        BATCH_SIZE (int): minibatch size
        GAMMA (float): discount factor

# NETWORK PARAMETERS
        LR_ACTOR (float): learning rate of the actor
        LR_CRITIC (float): learning rate of the critic
        WEIGHT_DECAY (float): Coefficient for L2 weight regularisation in critic - if 0, no regularisation is performed
        TAU (float): for soft update of target parameters
        FC1_UNITS (int): Size of the first hidden layer in networks
        FC2_UNITS (int): Size of the second hidden layer in networks

# ORNSTEIN-UHLENBECK NOISE PROCESS PARAMETERS
        MU (float): Mean of noise term
        THETA (float): 1/time constant
        SIGMA (float): Standard Deviation of noise term
        SIGMA_MIN (float): Minimum value of standard deviation
        SIGMA_DECAY (float): Decay rate of OU Noise term
"""

# DDPG PARAMETERS
BUFFER_SIZE = int(5e5)
BATCH_SIZE = 128
GAMMA = 0.99

# NETWORK PARAMETERS
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0
TAU = 5e-2
FC1_UNITS = 256
FC2_UNITS = 128

# ORNSTEIN-UHLENBECK NOISE PROCESS PARAMETERS
MU = 0.0
THETA = 0.1
SIGMA= 0.5
SIGMA_MIN = 0.05
SIGMA_DECAY= 0.99
# 

# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
###################################
# Agent class - the agent explores the environment, collecting experiences and adding them to the Replay buffer. 
Initialises an Agent and Critic for each building. Can also be used to test/run a trained network in the environment.
======
'''
class Agent():
    
    def __init__(self, building_info, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            building_info (list): List of building parameters for each building
            state_size (list): List of list of dimensions of each state for each building
            action_size (list): List of list of dimensions of each action for each building
            random_seed (int): random seed
        """
        # Get total number of observations for current environment
        total_observations = 0
        for building in range(len(state_size)):
        	total_observations = total_observations + state_size[building].shape[0]

        self.building_info = building_info
        self.state_size = total_observations
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local, self.critic_local, self.actor_target, self.critic_target, self.actor_optimizer, self.critic_optimizer =  {}, {}, {}, {}, {}, {}
        self.noise = {}
        self.memory = {}

        self.critic_loss = 0
        self.actor_loss = 0
        
        # For each building, set up a actor and critic network
        for i, (o, a) in enumerate(zip(state_size, action_size)):
            # Actor Network (w/ Target Network)
            self.actor_local[i] = Actor(self.state_size, self.action_size[i].shape[0], random_seed).to(device)
            self.actor_target[i] = Actor(self.state_size, action_size[i].shape[0], random_seed).to(device)
            self.actor_optimizer[i] = optim.Adam(self.actor_local[i].parameters(), lr=LR_ACTOR)

            # Critic Network (w/ Target Network)
            self.critic_local[i] = Critic(self.state_size, self.action_size[i].shape[0], random_seed).to(device)
            self.critic_target[i] = Critic(self.state_size, self.action_size[i].shape[0], random_seed).to(device)
            self.critic_optimizer[i] = optim.Adam(self.critic_local[i].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

            # Noise process for each agent
            self.noise[i] = OUNoise(self.action_size[i].shape[0], random_seed)

            # Replay memory
            self.memory[i] = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.reset_action_tracker()

    def reset_action_tracker(self):
        self.action_tracker = []

    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.
        
        Params
        ======
            states (Array): Array of states for every building
            actions (Array): Array of actions for each building
            rewards (Array): Array of rewards for each building
            next_states (Array): Array of updates states for each building after actions
            dones (Boolean): Whether episode is done (terminated) or not
        """
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        # Save experience / reward
        for building in range(0,len(rewards)):
            self.memory[building].add(states, actions[building], rewards[building], next_states, dones)
        
        # Learn, if enough samples are available in memory
        for building in range(0,len(rewards)):
            if len(self.memory[building]) > BATCH_SIZE:
                experiences = self.memory[building].sample()
                self.learn(experiences, building)

    def select_action(self, states, add_noise=False):
        """Returns actions for given state as per current policy.

        Params
        ======
            states (Array): Array of states for every building
            add_noise (Boolean): Whether to add OU noise to the output action (default = False)
        """
        states_concat = np.concatenate(states, axis = 0)
        states_concat = torch.from_numpy(states_concat).float().to(device)
        acts = []
        with torch.no_grad():
            for building in range(0,len(self.building_info)):
                self.actor_local[building].eval()
                action = self.actor_local[building](states_concat).cpu().data.numpy()
                #print(action)
                acts.append(action)
                self.actor_local[building].train()
        # Add noise if training/exploring
        if add_noise:
            for building in range(0,len(self.building_info)):
                acts[building] += list(self.noise[building].sample())[0]
        # Determine action bounds
        action_bounds_low = np.zeros((2,len(self.building_info)))
        action_bounds_high = np.zeros((2,len(self.building_info)))
        #print(acts)
        for building in range(0,len(self.building_info)):
            break
            # Limit actions based on physical constraints (e.g. if SoC is 0)
            # DWH Storage Limits
            if states[building][-1] == 0:
                action_bounds_low[1,building] = 0
            else:
                action_bounds_low[1,building] = self.action_size[building].low[1]
            if states[building][-1] == 1:
                action_bounds_high[1,building] = 0
            else:
                action_bounds_high[1,building] = self.action_size[building].high[1]
            # Cooling Storage Limits
            if states[building][-2] == 0:
                action_bounds_low[0,building] = 0
            else:
                action_bounds_low[0,building] = self.action_size[building].low[0]
            if states[building][-2] == 1:
                action_bounds_high[0,building] = 0
            else:
                action_bounds_high[0,building] = self.action_size[building].high[0]
            # Constrain actions based on bounds
            acts[building] = np.clip(acts[building], action_bounds_low[:,building], action_bounds_high[:,building])

        self.action_tracker.append(acts)
        
        return acts

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, building):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            building (int): building number
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target[building](next_states)
        Q_targets_next = self.critic_target[building](next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local[building](states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_loss = critic_loss
        # Minimize the loss
        self.critic_optimizer[building].zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer[building].step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local[building](states)
        actor_loss = -self.critic_local[building](states, actions_pred).mean()
        self.actor_loss = actor_loss
        # Minimize the loss
        self.actor_optimizer[building].zero_grad()
        actor_loss.backward()
        self.actor_optimizer[building].step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local[building], self.critic_target[building], TAU)
        self.soft_update(self.actor_local[building], self.actor_target[building], TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

'''
###################################
# Actor class - The policy model. Takes the state and outputs an action. 
======
'''
class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

'''
###################################
# Critic class - The value model. The critic measures how good the action taken by the policy (Actor) is.
======
'''
class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

'''
###################################
# ReplayBuffer class - Fixed-size buffer to store experience tuples.
======
'''
class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

'''
###################################
# OU Noise class - Ornstein-Uhlenbeck Noise process.
======
'''
class OUNoise:

    def __init__(self, size, seed, mu= MU , theta= THETA , sigma= SIGMA , sigma_min = SIGMA_MIN , sigma_decay= SIGMA_DECAY):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state