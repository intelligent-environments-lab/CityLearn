import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
np.random.seed(1)

class Buffer:
    def __init__(self):
        self.buffer = []
        
    def append_sample(self, sample):
        self.buffer.append(sample)
        
    def sample(self, sample_size):
        s, a, r, s_next, done = [],[],[],[],[]
        
        if sample_size > len(self.buffer):
            sample_size = len(self.buffer)
            
        rand_sample = random.sample(self.buffer, sample_size)
        for values in rand_sample:
            s.append(values[0])
            a.append(values[1])
            r.append(values[2])
            s_next.append(values[3])
            done.append([4])
        return torch.tensor(s,dtype=torch.float32), torch.tensor(a,dtype=torch.float32), torch.tensor(r,dtype=torch.float32), torch.tensor(s_next,dtype=torch.float32), done
    
    def __len__(self):
         return len(self.buffer)
                           
class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 5)
        self.l2 = nn.Linear(5, 3)
        self.l3 = nn.Linear(3, action_dim)

        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 7)
        self.l2 = nn.Linear(7, 6)
        self.l3 = nn.Linear(6, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 7)
        self.l5 = nn.Linear(7, 6)
        self.l6 = nn.Linear(6, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
class TD3_Agents:
    def __init__(self, observation_spaces = None, action_spaces = None):
        
        #Hyper-parameters
        self.discount = 0.992 #Discount factor
        self.batch_size = 800 #Size of each MINI-BATCH
        self.iterations = 20 # Number of updates of the actor-critic networks every time-step
        self.policy_freq = 2 # Number of iterations after which the actor and target networks are updated
        self.tau = 0.0015*10 #Rate at which the target networks are updated
        self.lr_init = 5e-2
        self.lr_final = 3e-3
        self.lr_decay_rate = 1/10000
        self.expl_noise_init = 0.75 # Exploration noise at time-step 0
        self.expl_noise_final = 0.01 # Magnitude of the minimum exploration noise
        self.expl_noise_decay_rate = 1/14000  # Decay rate of the exploration noise in 1/h
        self.policy_noise = 0.025
        self.noise_clip = 0.04
        self.max_action = 0.25
        self.min_samples_training = 400 #Min number of tuples that are stored in the batch before the training process begins
        
        # Parameters
        self.device = "cpu"
        self.time_step = 0
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.n_buildings = len(observation_spaces)
        self.buffer = {i: Buffer() for i in range(self.n_buildings)}
        self.networks_initialized = False
        
        # Monitoring variables (one per agent)
        self.actor_loss_list = {i: [] for i in range(self.n_buildings)}
        self.critic1_loss_list = {i: [] for i in range(self.n_buildings)}
        self.critic2_loss_list = {i: [] for i in range(self.n_buildings)}
        self.q_val_list = {i: [] for i in range(self.n_buildings)}
        self.q1_list = {i: [] for i in range(self.n_buildings)}
        self.q2_list = {i: [] for i in range(self.n_buildings)}
        self.a_track1 = []
        self.a_track2 = []
        
        #Networks and optimizers (one per agent)
        self.actor, self.critic, self.actor_target, self.critic_target, self.actor_optimizer, self.critic_optimizer =  {}, {}, {}, {}, {}, {}
        for i, (o, a) in enumerate(zip(observation_spaces, action_spaces)):
            self.actor[i] = Actor(o.shape[0], a.shape[0], self.max_action).to(self.device)
            self.critic[i] = Critic(o.shape[0], a.shape[0]).to(self.device)
            self.actor_target[i] = copy.deepcopy(self.actor[i])
            self.critic_target[i] = copy.deepcopy(self.critic[i])
            self.actor_optimizer[i] = optim.Adam(self.actor[i].parameters(), lr=self.lr_init)
            self.critic_optimizer[i] = optim.Adam(self.critic[i].parameters(), lr=self.lr_init)
        
    def select_action(self, states):
        expl_noise = max(self.expl_noise_final, self.expl_noise_init * (1 - self.time_step * self.expl_noise_decay_rate))
        
        actions = []
        for i, state in enumerate(states):
            a = self.actor[i](torch.tensor(state, dtype=torch.float32))
            self.a_track1.append(a)
            a = a.cpu().detach().numpy() + expl_noise * np.random.normal(loc = 0, scale = self.max_action, size=a.shape)
            self.a_track2.append(a)
            a = np.clip(a, -self.max_action, self.max_action)
            actions.append(a)
        return actions
    
    def add_to_buffer(self, states, actions, rewards, next_states, dones):
        dones = [dones for _ in range(self.n_buildings)]
        
        for i, (s, a, r, s_next, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            s = (s - self.observation_spaces[i].low)/(self.observation_spaces[i].high - self.observation_spaces[i].low + 0.00001)
            s_next = (s_next - self.observation_spaces[i].low)/(self.observation_spaces[i].high - self.observation_spaces[i].low + 0.00001)
            self.buffer[i].append_sample((s, a, r, s_next, done))

        lr = max(self.lr_final, self.lr_init * (1 - self.time_step * self.lr_decay_rate))
        for i in range(self.n_buildings):
            self.actor_optimizer[i] = optim.Adam(self.actor[i].parameters(), lr=lr)
            self.critic_optimizer[i] = optim.Adam(self.critic[i].parameters(), lr=lr)
            
        #One TD3 control agent for each building
        for i in range(self.n_buildings):
            
            #Learning begins when a minimum number of tuples have beena added to the buffer
            if len(self.buffer[i]) > self.min_samples_training:
                
                #Every time-step we randomly sample 'self.iterations' number of minibatches from the buffer of experiences and perform 'self.iterations' number of updates of the networks.
                for k in range(self.iterations):
                    state, action, reward, next_state, dones_mask = self.buffer[i].sample(self.batch_size)
                    target_Q = reward.unsqueeze(dim=-1)

                    with torch.no_grad():
                        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                        
                        # Select action according to policy
                        next_action = (self.actor_target[i](next_state) + noise).clamp(-self.max_action, self.max_action)
                        
                        # Compute the target Q value
                        target_Q1, target_Q2 = self.critic_target[i](next_state, next_action)
                        target_Q = torch.min(target_Q1, target_Q2)
                        target_Q = reward.unsqueeze(dim=-1) + target_Q * self.discount
                        
                    # Get current Q estimates
                    current_Q1, current_Q2 = self.critic[i](state, action)    
                    
                    # Compute critic loss
                    critic1_loss = F.mse_loss(current_Q1, target_Q)
                    critic2_loss = F.mse_loss(current_Q2, target_Q)
                    critic_loss = critic1_loss + critic2_loss
                    
                    # Optimize the critic
                    self.critic_optimizer[i].zero_grad()
                    critic_loss.backward()  
                    self.critic_optimizer[i].step()
                    
                    # Save values
                    self.q_val_list[i].append(target_Q)
                    self.q1_list[i].append(current_Q1)
                    self.q2_list[i].append(current_Q2)
                    self.critic1_loss_list[i].append(critic1_loss)
                    self.critic2_loss_list[i].append(critic2_loss)
                    
                    # Delayed policy updates
                    if k % self.policy_freq == 0:
                        
                        # Compute actor loss
                        actor_loss = -self.critic[i].Q1(state, self.actor[i](state)).mean()
                        self.actor_loss_list[i].append(actor_loss)
                                        
                        # Optimize the actor
                        self.actor_optimizer[i].zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer[i].step()

                        # Update the frozen target models
                        for param, target_param in zip(self.critic[i].parameters(), self.critic_target[i].parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                        for param, target_param in zip(self.actor[i].parameters(), self.actor_target[i].parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.time_step += 1
                            