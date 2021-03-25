from common.preprocessing import *
from common.rl import *
import torch.optim as optim
from torch.optim import Adam
import json

class SAC:
    def __init__(self, building_ids,
                 buildings_states_actions,
                 building_info,
                 observation_spaces = None,
                 action_spaces = None,
                 hidden_dim=[256,256],
                 discount=0.99,
                 tau=5e-3,
                 lr=3e-4,
                 batch_size=256,
                 replay_buffer_capacity = 1e5,
                 start_training = 6000,
                 exploration_period = 7000,
                 action_scaling_coef = 0.5,
                 reward_scaling = 5.,
                 update_per_step = 2,
                 seed = 0):
                
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)
            
        self.building_ids = building_ids
        self.start_training = start_training
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        self.action_scaling_coef = action_scaling_coef
        self.reward_scaling = reward_scaling
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.deterministic = False
        self.update_per_step = update_per_step
        self.exploration_period = exploration_period
        
        self.action_list_ = []
        self.action_list2_ = []
        
        self.time_step = 0
        self.norm_flag = {uid : 0 for uid in building_ids}
        self.action_spaces = {uid : a_space for uid, a_space in zip(building_ids, action_spaces)}
        self.observation_spaces = {uid : o_space for uid, o_space in zip(building_ids, observation_spaces)}
        
        # Optimizers/Loss using the Huber loss
        self.soft_q_criterion = nn.SmoothL1Loss()
        
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:'+"cuda" if torch.cuda.is_available() else "cpu")
        self.critic1_loss_, self.critic2_loss_, self.actor_loss_, self.alpha_loss_, self.alpha_ = {}, {}, {}, {}, {}
        
        
        self.replay_buffer, self.soft_q_net1, self.soft_q_net2, self.target_soft_q_net1, self.target_soft_q_net2, self.policy_net, self.soft_q_optimizer1, self.soft_q_optimizer2, self.policy_optimizer, self.target_entropy, self.alpha, self.encoder, self.norm_mean, self.norm_std, self.r_norm_mean, self.r_norm_std, self.norm_mean, self.norm_std, self.r_norm_mean, self.r_norm_std, = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        
        for uid in building_ids:
            self.critic1_loss_[uid], self.critic2_loss_[uid], self.actor_loss_[uid], self.alpha_[uid] = [], [], [], []
            self.encoder[uid] = []
            state_n = 0
            for s_name, s in self.buildings_states_actions[uid]['states'].items():
                if not s:
                    self.encoder[uid].append(0)
                elif s_name in ["month", "hour"]:
                    self.encoder[uid].append(periodic_normalization(self.observation_spaces[uid].high[state_n]))
                    state_n += 1
                elif s_name == "day":
                    self.encoder[uid].append(onehot_encoding([1,2,3,4,5,6,7,8]))
                    state_n += 1
                elif s_name == "daylight_savings_status":
                    self.encoder[uid].append(onehot_encoding([0,1]))
                    state_n += 1
                elif s_name == "net_electricity_consumption":
                    self.encoder[uid].append(remove_feature())
                    state_n += 1
                else:
                    self.encoder[uid].append(normalize(self.observation_spaces[uid].low[state_n], self.observation_spaces[uid].high[state_n]))
                    state_n += 1  

            self.encoder[uid] = np.array(self.encoder[uid])

            # If there is no solar PV installed, remove solar radiation variables 
            if building_info[uid]['solar_power_capacity (kW)'] == 0:
                for k in range(12,20):
                    if self.encoder[uid][k] != 0:
                        self.encoder[uid][k] = -1
                if self.encoder[uid][24] != 0:
                    self.encoder[uid][24] = -1
            if building_info[uid]['Annual_DHW_demand (kWh)'] == 0 and self.encoder[uid][26] != 0:
                self.encoder[uid][26] = -1
            if building_info[uid]['Annual_cooling_demand (kWh)'] == 0 and self.encoder[uid][25] != 0:
                self.encoder[uid][25] = -1
            if building_info[uid]['Annual_nonshiftable_electrical_demand (kWh)'] == 0 and self.encoder[uid][23] != 0:
                self.encoder[uid][23] = -1

            self.encoder[uid] = self.encoder[uid][self.encoder[uid]!=0]
            self.encoder[uid][self.encoder[uid]==-1] = remove_feature()
            
            state_dim = len([j for j in np.hstack(self.encoder[uid]*np.ones(len(self.observation_spaces[uid].low))) if j != None])
                
            action_dim = self.action_spaces[uid].shape[0]
            self.alpha[uid] = 0.2
            
            self.replay_buffer[uid] = ReplayBuffer(int(replay_buffer_capacity))
            
            # init networks
            self.soft_q_net1[uid] = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.soft_q_net2[uid] = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

            self.target_soft_q_net1[uid] = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_soft_q_net2[uid] = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

            for target_param, param in zip(self.target_soft_q_net1[uid].parameters(), self.soft_q_net1[uid].parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_soft_q_net2[uid].parameters(), self.soft_q_net2[uid].parameters()):
                target_param.data.copy_(param.data)

            # Policy
            self.policy_net[uid] = PolicyNetwork(state_dim, action_dim, self.action_spaces[uid], self.action_scaling_coef, hidden_dim).to(self.device)
            self.soft_q_optimizer1[uid] = optim.Adam(self.soft_q_net1[uid].parameters(), lr=lr)
            self.soft_q_optimizer2[uid] = optim.Adam(self.soft_q_net2[uid].parameters(), lr=lr)
            self.policy_optimizer[uid] = optim.Adam(self.policy_net[uid].parameters(), lr=lr)
            self.target_entropy[uid] = -np.prod(self.action_spaces[uid].shape).item()
            
            
    def select_action(self, states):
        
        self.time_step += 1
        explore = self.time_step <= self.exploration_period
        actions = []
        k = 0
        
        deterministic = (self.time_step > 3*8760)
        
        for uid, state in zip(self.building_ids, states):
            if explore:
                actions.append(self.action_scaling_coef*self.action_spaces[uid].sample())
            else:
                state_ = np.array([j for j in np.hstack(self.encoder[uid]*state) if j != None])

                state_ = (state_  - self.norm_mean[uid])/self.norm_std[uid]
                state_ = torch.FloatTensor(state_).unsqueeze(0).to(self.device)

                if deterministic is False:
                    act, _, _ = self.policy_net[uid].sample(state_)
                else:
                    _, _, act = self.policy_net[uid].sample(state_)

                actions.append(act.detach().cpu().numpy()[0])
                    
        return np.array(actions), None
                
        
    def add_to_buffer(self, states, actions, rewards, next_states, done, coordination_vars, coordination_vars_next):
        
        for (uid, o, a, r, o2,) in zip(self.building_ids, states, actions, rewards, next_states):            
            # Run once the regression model has been fitted
            # Normalize all the states using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes states that are not necessary (solar radiation if there are no solar PV panels).
            
            o = np.array([j for j in np.hstack(self.encoder[uid]*o) if j != None])
            o2 = np.array([j for j in np.hstack(self.encoder[uid]*o2) if j != None])
            
            if self.norm_flag[uid] > 0:
                o = (o - self.norm_mean[uid])/self.norm_std[uid]
                o2 = (o2 - self.norm_mean[uid])/self.norm_std[uid]
                r = (r - self.r_norm_mean[uid])/self.r_norm_std[uid]

            self.replay_buffer[uid].push(o, a, r, o2, done)
            
        if self.time_step >= self.start_training and self.batch_size <= len(self.replay_buffer[self.building_ids[0]]): 
            for uid in self.building_ids:
                if self.norm_flag[uid] == 0:
                    X = np.array([j[0] for j in self.replay_buffer[uid].buffer])
                    self.norm_mean[uid] = np.mean(X, axis=0)
                    self.norm_std[uid] = np.std(X, axis=0) + 1e-5

                    R = np.array([j[2] for j in self.replay_buffer[uid].buffer])
                    self.r_norm_mean[uid] = np.mean(R)
                    self.r_norm_std[uid] = np.std(R)/self.reward_scaling + 1e-5

                    new_buffer = []
                    for s, a, r, s2, dones in self.replay_buffer[uid].buffer:
                        s_buffer = np.hstack(((s - self.norm_mean[uid])/self.norm_std[uid]).reshape(1,-1)[0])
                        s2_buffer = np.hstack(((s2 - self.norm_mean[uid])/self.norm_std[uid]).reshape(1,-1)[0])
                        new_buffer.append((s_buffer, a, (r - self.r_norm_mean[uid])/self.r_norm_std[uid], s2_buffer, dones))

                    self.replay_buffer[uid].buffer = new_buffer
                    self.norm_flag[uid] = 1
                    
            for _ in range(self.update_per_step):
                for uid in self.building_ids:
                    state, action, reward, next_state, done = self.replay_buffer[uid].sample(self.batch_size)

                    if self.device.type == "cuda":
                        state      = torch.cuda.FloatTensor(state).to(self.device)
                        next_state = torch.cuda.FloatTensor(next_state).to(self.device)
                        action     = torch.cuda.FloatTensor(action).to(self.device)
                        reward     = torch.cuda.FloatTensor(reward).unsqueeze(1).to(self.device)
                        done       = torch.cuda.FloatTensor(done).unsqueeze(1).to(self.device)
                    else:
                        state      = torch.FloatTensor(state).to(self.device)
                        next_state = torch.FloatTensor(next_state).to(self.device)
                        action     = torch.FloatTensor(action).to(self.device)
                        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
                        done       = torch.FloatTensor(done).unsqueeze(1).to(self.device)

                    with torch.no_grad():
                        # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) state and its associated log probability of occurrence.
                        new_next_actions, new_log_pi, _ = self.policy_net[uid].sample(next_state)

                        # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
                        target_q_values = torch.min(
                                        self.target_soft_q_net1[uid](next_state, new_next_actions),
                                        self.target_soft_q_net2[uid](next_state, new_next_actions),
                                    ) - self.alpha[uid] * new_log_pi

                        q_target = reward + (1 - done) * self.discount * target_q_values

                    # Update Soft Q-Networks
                    q1_pred = self.soft_q_net1[uid](state, action)
                    q2_pred = self.soft_q_net2[uid](state, action)

                    q1_loss = self.soft_q_criterion(q1_pred, q_target)
                    q2_loss = self.soft_q_criterion(q2_pred, q_target)


                    self.soft_q_optimizer1[uid].zero_grad()
                    q1_loss.backward()
                    self.soft_q_optimizer1[uid].step()

                    self.soft_q_optimizer2[uid].zero_grad()
                    q2_loss.backward()
                    self.soft_q_optimizer2[uid].step()

                    # Update Policy    
                    new_actions, log_pi, _ = self.policy_net[uid].sample(state)

                    q_new_actions = torch.min(
                        self.soft_q_net1[uid](state, new_actions),
                        self.soft_q_net2[uid](state, new_actions)
                    )

                    policy_loss = (self.alpha[uid]*log_pi - q_new_actions).mean()

                    self.policy_optimizer[uid].zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer[uid].step()

                    # Soft Updates
                    for target_param, param in zip(self.target_soft_q_net1[uid].parameters(), self.soft_q_net1[uid].parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - self.tau) + param.data * self.tau
                        )

                    for target_param, param in zip(self.target_soft_q_net2[uid].parameters(), self.soft_q_net2[uid].parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - self.tau) + param.data * self.tau
                        )