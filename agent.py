import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
np.random.seed(1)

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, act_size),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 8),
            nn.ReLU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(8 + act_size, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )
    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))
    
class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
        
    
class Batch:
    def __init__(self):
        self.batch = []
        
    def append_sample(self, sample):
        self.batch.append(sample)
        
    def sample(self, sample_size):
        s, a, r, s_next, done = [],[],[],[],[]
        
        if sample_size > len(self.batch):
            sample_size = len(self.batch)
            
        rand_sample = random.sample(self.batch, sample_size)
        for values in rand_sample:
            s.append(values[0])
            a.append(values[1])
            r.append(values[2])
            s_next.append(values[3])
            done.append([4])
        return torch.tensor(s,dtype=torch.float32), torch.tensor(a,dtype=torch.float32), torch.tensor(r,dtype=torch.float32), torch.tensor(s_next,dtype=torch.float32), done
    
    def __len__(self):
         return len(self.batch)
        
    
class RL_Agents:
    def __init__(self, observation_spaces = None, action_spaces = None):
        self.device = "cpu"
        self.epsilon = 0.3
        self.n_buildings = len(observation_spaces)
        self.batch = {}
        self.frame_idx = {}
        self.actor_loss_list, self.critic_loss_list = {i: [] for i in range(self.n_buildings)}, {i: [] for i in range(self.n_buildings)}
        for i in range(len(observation_spaces)):
            self.batch[i] = Batch()
            
        #Hyper-parameters
        #They need to be properly calibrated to improve robustness. With these values for the hyper-parameters below, the agent does not always converge to the optimal solution
        LEARNING_RATE_ACTOR = 1e-4
        LEARNING_RATE_CRITIC = 1e-3
        self.MIN_REPLAY_MEMORY = 600 #Min number of tuples that are stored in the batch before the training process begins
        self.BATCH_SIZE = 800 #Size of each MINI-BATCH
        self.ITERATIONS = 20 #Number of iterations used to update the networks for every time-step. It can be set to 1 if we are going to run the agent for an undefined number of episodes (in the main file). In this case, in the main file, we would use a stopping criterion based on a given threshold for the cost. To learn the policy within a single episode, we should increase the value of ITERATIONS
        self.GAMMA = 0.99 #Discount factor
        #Epsilon multiplies the exploration noise using in the action selection. 
        self.EPSILON_START = 1.3 #Epsilon at time-step 0
        self.EPSILON_FINAL = 0.01 #Epsilon at time-step EPSILON_DECAY_LAST_FRAME
        self.EPSILON_DECAY_LAST_FRAME = 22000 #4900 #Time-step in which Epsilon reaches the value EPSILON_FINAL and the steady state
        self.SYNC_RATE = 0.01 #Rate at which the target networks are updated to converge towards the actual critic and actor networks. Decreasing SYNC_RATE may require to increase the hyper-parameter ITERATIONS
        
        self.hour_idx = 0
        i = 0
        self.act_net, self.crt_net, self.tgt_act_net, self.tgt_crt_net, self.act_opt, self.crt_opt = {}, {}, {}, {}, {}, {}
        for o, a in zip(observation_spaces, action_spaces):
            self.act_net[i] = DDPGActor(o.shape[0], a.shape[0]).to(self.device)
            self.crt_net[i] = DDPGCritic(o.shape[0], a.shape[0]).to(self.device)
            self.tgt_act_net[i] = TargetNet(self.act_net[i])
            self.tgt_crt_net[i] = TargetNet(self.crt_net[i])
            self.act_opt[i] = optim.Adam(self.act_net[i].parameters(), lr=LEARNING_RATE_ACTOR)
            self.crt_opt[i] = optim.Adam(self.crt_net[i].parameters(), lr=LEARNING_RATE_CRITIC)
            i += 1
        
    def select_action(self, states):
        i, actions = 0, []
        action_magnitude = 0.33
        for state in states:
            a = action_magnitude*self.act_net[i](torch.tensor(state))
            a = a.cpu().detach().numpy() + self.epsilon * np.random.normal(loc = 0, scale = action_magnitude, size=a.shape)
            a = np.clip(a, -action_magnitude, action_magnitude)
            actions.append(a)
            i += 1
        return actions
    
    def add_to_batch(self, states, actions, rewards, next_states, dones):
        i = 0
        dones = [dones for _ in range(self.n_buildings)]
        for s, a, r, s_next, done in zip(states, actions, rewards, next_states, dones):
            self.batch[i].append_sample((s, a, r, s_next, done))
            i += 1
            
        batch, states_v, actions_v, rewards_v, dones_mask, states_next_v, q_v, last_act_v, q_last_v, q_ref_v, critic_loss_v, cur_actions_v, actor_loss_v = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}  
        
        self.epsilon = max(self.EPSILON_FINAL, self.EPSILON_START - self.hour_idx / self.EPSILON_DECAY_LAST_FRAME)
        self.hour_idx += 1
        #There is one DDPG control agent for each building
        for i in range(self.n_buildings):
            #The learning begins when a minimum number of tuples have beena added to the batch
            if len(self.batch[i]) > self.MIN_REPLAY_MEMORY:
                #Every time-step we sample a random minibatch from the batch of experiences and perform the updates of the networks. We do this self.ITERATIONS times every time-step
                for k in range(self.ITERATIONS):
                    states_v[i], actions_v[i], rewards_v[i], states_next_v[i], dones_mask[i] = self.batch[i].sample(self.BATCH_SIZE)

                    # TRAIN CRITIC
                    self.crt_opt[i].zero_grad()
                    #Obtaining Q' using critic net with parameters teta_Q'
                    q_v[i] = self.crt_net[i](states_v[i], actions_v[i])

                    #Obtaining estimated optimal actions a|teta_mu from target actor net and from s_i+1.
                    last_act_v[i] = self.tgt_act_net[i].target_model(states_next_v[i]) #<----- Actor is used to train the Critic

                    #Obtaining Q'(s_i+1, a|teta_mu) from critic net Q'
                    q_last_v[i] = self.tgt_crt_net[i].target_model(states_next_v[i], last_act_v[i])
                    # q_last_v[i][dones_mask[i]] = 0.0

                    #Q_target used to train critic net Q'
                    q_ref_v[i] = rewards_v[i].unsqueeze(dim=-1) + q_last_v[i] * self.GAMMA
                    critic_loss_v[i] = F.mse_loss(q_v[i], q_ref_v[i].detach())
                    self.critic_loss_list[i].append(critic_loss_v[i])
                    critic_loss_v[i].backward()  
                    self.crt_opt[i].step()

                    # TRAIN ACTOR
                    self.act_opt[i].zero_grad()
                    #Obtaining estimated optimal current actions a|teta_mu from actor net and from s_i
                    cur_actions_v[i] = self.act_net[i](states_v[i])

                    #Actor loss = mean{ -Q_i'(s_i, a|teta_mu) }
                    actor_loss_v[i] = -self.crt_net[i](states_v[i], cur_actions_v[i]) #<----- Critic is used to train the Actor
                    actor_loss_v[i] = actor_loss_v[i].mean()
                    self.actor_loss_list[i].append(actor_loss_v[i])
                    #Find gradient of the loss and backpropagate to perform the updates of teta_mu
                    actor_loss_v[i].backward()
                    self.act_opt[i].step()

                    #Gradually copies the actor and critic networks to the target networks. Using target networks should make the algorithm more stable.
                    self.tgt_act_net[i].alpha_sync(alpha=1 - self.SYNC_RATE)
                    self.tgt_crt_net[i].alpha_sync(alpha=1 - self.SYNC_RATE)
                     
                    
#MANUALLY OPTIMIZED RULE BASED CONTROLLER
class RBC_Agent:
    def __init__(self):
        self.hour = 3500

    def select_action(self, states):
        self.hour += 1
        hour_day = states[0][0]
        #DAYTIME
        a = 0.0
        if hour_day >= 12 and hour_day <= 19:
            #SUMMER (RELEASE COOLING)
            if self.hour >= 2800 and self.hour <= 7000:
                a = -0.34
        #NIGHTTIME       
        elif hour_day >= 2 and hour_day <= 9:
            #SUMMER (STORE COOLING)
            if self.hour >= 2800 and self.hour <= 7000:
                a = 0.2
        return np.array([[a] for _ in range(len(states))])

    def add_to_batch(self, states, actions, rewards, next_states, dones):
        pass
    
    
###############################################################################################    
class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 4)
        self.l2 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, action_dim)

        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 6)
        self.l5 = nn.Linear(6, 4)
        self.l6 = nn.Linear(4, 1)


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
        self.device = "cpu"
        self.epsilon = 0.3
        self.n_buildings = len(observation_spaces)
        self.batch = {}
        self.frame_idx = {}
        self.actor_loss_list, self.critic_loss_list = {i: [] for i in range(self.n_buildings)}, {i: [] for i in range(self.n_buildings)}
        for i in range(len(observation_spaces)):
            self.batch[i] = Batch()
            
        #Hyper-parameters
        #They need to be properly calibrated to improve robustness. With these values for the hyper-parameters below, the agent does not always converge to the optimal solution
        LEARNING_RATE_ACTOR = 3e-4
        LEARNING_RATE_CRITIC = 3e-4
        self.MIN_REPLAY_MEMORY = 600 #Min number of tuples that are stored in the batch before the training process begins
        self.BATCH_SIZE = 100 #Size of each MINI-BATCH
        self.policy_freq = 4
        self.ITERATIONS = 2 #Number of iterations used to update the networks for every time-step. It can be set to 1 if we are going to run the agent for an undefined number of episodes (in the main file). In this case, in the main file, we would use a stopping criterion based on a given threshold for the cost. To learn the policy within a single episode, we should increase the value of ITERATIONS
        self.GAMMA = 0.99 #Discount factor
        #Epsilon multiplies the exploration noise using in the action selection. 
        self.EPSILON_START = 1.0 #Epsilon at time-step 0
        self.EPSILON_FINAL = 0.01 #Epsilon at time-step EPSILON_DECAY_LAST_FRAME
        self.EPSILON_DECAY_LAST_FRAME = 22000 #4900 #Time-step in which Epsilon reaches the value EPSILON_FINAL and the steady state
        self.SYNC_RATE = 0.05 #Rate at which the target networks are updated to converge towards the actual critic and actor networks. Decreasing SYNC_RATE may require to increase the hyper-parameter ITERATIONS
        
        self.hour_idx = 0
        i = 0
        self.act_net, self.crt_net, self.tgt_act_net, self.tgt_crt_net, self.act_opt, self.crt_opt, self.a_track1, self.a_track2 = {}, {}, {}, {}, {}, {}, [], []
        for o, a in zip(observation_spaces, action_spaces):
            self.act_net[i] = Actor(o.shape[0], a.shape[0], 0.2).to(self.device)
            self.crt_net[i] = Critic(o.shape[0], a.shape[0]).to(self.device)
            self.tgt_act_net[i] = TargetNet(self.act_net[i])
            self.tgt_crt_net[i] = TargetNet(self.crt_net[i])
            self.act_opt[i] = optim.Adam(self.act_net[i].parameters(), lr=LEARNING_RATE_ACTOR)
            self.crt_opt[i] = optim.Adam(self.crt_net[i].parameters(), lr=LEARNING_RATE_CRITIC)
            i += 1
        
    def select_action(self, states):
        i, actions = 0, []
        action_magnitude = 0.2
        for state in states:
            a = self.act_net[i](torch.tensor(state))
            self.a_track1.append(a)
            a = a.cpu().detach().numpy() + self.epsilon * np.random.normal(loc = 0, scale = action_magnitude, size=a.shape)
            self.a_track2.append(a)
            a = np.clip(a, -action_magnitude, action_magnitude)
            actions.append(a)
            i += 1
        return actions
    
    def add_to_batch(self, states, actions, rewards, next_states, dones):
        i = 0
        dones = [dones for _ in range(self.n_buildings)]
        for s, a, r, s_next, done in zip(states, actions, rewards, next_states, dones):
            self.batch[i].append_sample((s, a, r, s_next, done))
            i += 1
            
        batch, states_v, actions_v, rewards_v, dones_mask, states_next_v, current_Q1, current_Q2, last_act_v, q_last_v, q_ref_v, critic_loss_v, cur_actions_v, actor_loss_v = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}  
        
        self.epsilon = max(self.EPSILON_FINAL, self.EPSILON_START - self.hour_idx / self.EPSILON_DECAY_LAST_FRAME)
        self.hour_idx += 1
        #There is one DDPG control agent for each building
        for i in range(self.n_buildings):
            #The learning begins when a minimum number of tuples have beena added to the batch
            if len(self.batch[i]) > self.MIN_REPLAY_MEMORY:
                #Every time-step we sample a random minibatch from the batch of experiences and perform the updates of the networks. We do this self.ITERATIONS times every time-step
                for k in range(self.ITERATIONS):
                    states_v[i], actions_v[i], rewards_v[i], states_next_v[i], dones_mask[i] = self.batch[i].sample(self.BATCH_SIZE)

                    # TRAIN CRITIC
                    self.crt_opt[i].zero_grad()
                    #Obtaining Q' using critic net with parameters teta_Q'
                    current_Q1[i], current_Q2[i] = self.crt_net[i](states_v[i], actions_v[i])
                    
                    #Obtaining estimated optimal actions a|teta_mu from target actor net and from s_i+1.
                    last_act_v[i] = self.tgt_act_net[i].target_model(states_next_v[i]) #<----- Actor is used to train the Critic

                    #Obtaining Q'(s_i+1, a|teta_mu) from critic net Q'
                    target_Q1, target_Q2 = self.tgt_crt_net[i].target_model(states_next_v[i], last_act_v[i])
                    q_last_v[i] = torch.min(target_Q1, target_Q2)
                    # q_last_v[i][dones_mask[i]] = 0.0

                    #Q_target used to train critic net Q'
                    q_ref_v[i] = rewards_v[i].unsqueeze(dim=-1) + q_last_v[i] * self.GAMMA
                    critic_loss_v[i] = F.mse_loss(current_Q1[i], q_ref_v[i].detach()) + F.mse_loss(current_Q2[i], q_ref_v[i].detach())
                    self.critic_loss_list[i].append(critic_loss_v[i])
                    critic_loss_v[i].backward()  
                    self.crt_opt[i].step()
                    
                    # TRAIN ACTOR
                    if self.hour_idx % (self.policy_freq*self.ITERATIONS) == 0:
                        self.act_opt[i].zero_grad()
                        #Actor loss = mean{ -Q_i'(s_i, a|teta_mu) }
                        actor_loss_v[i] = -self.crt_net[i].Q1(states_v[i], self.act_net[i](states_v[i])).mean()
                        self.actor_loss_list[i].append(actor_loss_v[i])
                                        
                        #Find gradient of the loss and backpropagate to perform the updates of teta_mu
                        actor_loss_v[i].backward()
                        self.act_opt[i].step()

                        #Gradually copies the actor and critic networks to the target networks. Using target networks should make the algorithm more stable.
                        self.tgt_act_net[i].alpha_sync(alpha=1 - self.SYNC_RATE)
                        self.tgt_crt_net[i].alpha_sync(alpha=1 - self.SYNC_RATE)

class Q_Learning:
    def __init__(self, levels, min_action, max_action):
        self.Q = np.ones((24, 24, levels, levels))
        self.epsilon = 0.1
        self.n_actions = levels
        self.n_charges = levels
        self.gamma = 0.9999
        self.alpha = 0.1
        self.min_action = min_action
        self.max_action = max_action

    def discretize_states(self, states):
        states_copy = np.copy(states)
        states_copy[:,2] = np.floor(states_copy[:,2] * (self.n_charges - 1))
        states_copy[:,1] -= 17  # 17 is the minimum temp[]
        states_copy[:,0] -= 1   # Change 1-24 to 0-23
        # print("Disc", states, states_copy)
        return states_copy.astype(np.int)

    def discretize_actions(self, actions):
        actions = (actions - self.min_action) // (1/(self.n_actions - 1))
        return actions.astype(np.int)

    def undiscretize_actions(self, actions):
        a = self.min_action + (actions) / (self.n_actions - 1)
        return a

    def select_action(self, states):
        states = self.discretize_states(states)
        actions = np.zeros(states.shape[0])
        for i, state in enumerate(states):
            actions[i] = np.argmax(self.Q[state[0], state[1], state[2], :][::-1])
            if np.random.random() < self.epsilon: # * (1-episode/n_episodes + 0.01):
                actions[i] = np.random.choice(np.arange(self.n_actions))
        # print(actions)
        return np.array([np.expand_dims(self.undiscretize_actions(actions), axis=0)])

    def add_to_batch(self, states, actions, reward, next_states, dones):
        states = self.discretize_states(states[:])
        next_states = self.discretize_states(next_states[:])
        actions = self.discretize_actions(actions)
        for i in range(states.shape[0]):
            # print("ATB", states[i])
            self.Q[states[i,0], states[i,1], states[i,2], actions[i]] += \
                self.alpha * (reward + \
                                self.gamma * np.max(self.Q[next_states[i,0], next_states[i,1], next_states[i,2], :]) - \
                                self.Q[states[i,0], states[i,1], states[i,2], actions[i]])
