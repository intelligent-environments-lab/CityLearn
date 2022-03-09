from typing import List
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from citylearn.controller.base import Controller
from citylearn.controller.rbc import RBC, BasicRBC, OptimizedRBC
from citylearn.preprocessing import Encoder, NoNormalization
from citylearn.rl import PolicyNetwork, ReplayBuffer, SoftQNetwork

class RLC(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @property
    def reward(self) -> List[float]:
        return self.__reward

    @reward.setter
    def reward(self, reward: float):
        self.__reward[self.time_step] = reward

    def next_time_step(self):
        super().next_time_step()
        self.__reward.append(np.nan)

    def reset(self):
        super().reset()
        self.__reward = [np.nan]

class SAC(RLC):
    def __init__(
        self, action_spaces: spaces.Box, observation_spaces: spaces.Box, encoders: List[Encoder] = None, hidden_dimension: List[float] = [256, 256], discount: float = 0.99, tau: float = 5e-3, lr: float = 3e-4, batch_size: int = 256,
        replay_buffer_capacity: int = 1e5, start_training_time_step: int = 6000, end_exploration_time_step: int = 7000, 
        deterministic_start_time_step = 7500, action_scaling_coef: float = 0.5, reward_scaling: float = 5.0, 
        update_per_time_step: int = 2, seed: int = 0, **kwargs
    ):
        # user defined
        super().__init__(**kwargs)
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces
        self.encoders = encoders
        self.hidden_dimension = hidden_dimension
        self.discount = discount
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.start_training_time_step = start_training_time_step
        self.end_exploration_time_step = end_exploration_time_step
        self.deterministic_start_time_step = deterministic_start_time_step
        self.action_scaling_coef = action_scaling_coef
        self.reward_scaling = reward_scaling
        self.update_per_time_step = update_per_time_step
        self.seed = seed
        
        # internally defined
        self.__normalized = False
        self.__alpha = 0.2
        self.__soft_q_criterion = nn.SmoothL1Loss()
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__replay_buffer = ReplayBuffer(int(self.replay_buffer_capacity))
        self.__soft_q_net1 = None
        self.__soft_q_net2 = None
        self.__target_soft_q_net1 = None
        self.__target_soft_q_net2 = None
        self.__policy_net = None
        self.__soft_q_optimizer1 = None
        self.__soft_q_optimizer2 = None
        self.__policy_optimizer = None
        self.__target_entropy = None
        self.__norm_mean = None
        self.__norm_std = None
        self.__r_norm_mean = None
        self.__r_norm_std = None
        self.__set_networks()

    @property
    def action_spaces(self) -> spaces.Box:
        return self.__action_spaces

    @property
    def observation_spaces(self) -> spaces.Box:
        return self.__observation_spaces

    @property
    def encoders(self) -> List[Encoder]:
        return self.__encoders

    @property
    def hidden_dimension(self) -> List[float]:
        return self.__hidden_dimension

    @property
    def discount(self) -> float:
        return self.__discount

    @property
    def tau(self) -> float:
        return self.__tau
    
    @property
    def lr(self) -> float:
        return self.__lr

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def replay_buffer_capacity(self) -> int:
        return self.__replay_buffer_capacity

    @property
    def start_training_time_step(self) -> int:
        return self.__start_training_time_step

    @property
    def end_exploration_time_step(self) -> int:
        return self.__end_exploration_time_step

    @property
    def deterministic_start_time_step(self) -> int:
        return self.__deterministic_start_time_step

    @property
    def action_scaling_coef(self) -> float:
        return self.__action_scaling_coef

    @property
    def reward_scaling(self) -> float:
        return self.__reward_scaling

    @property
    def update_per_time_step(self) -> int:
        return self.__update_per_time_step

    @property
    def seed(self) -> int:
        return self.__seed

    @property
    def action_dimension(self) -> int:
        return self.action_spaces.shape[0]

    @property
    def observation_dimension(self) -> int:
        return len([j for j in np.hstack(self.encoders*np.ones(len(self.observation_spaces.low))) if j != None])

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def soft_q_net1(self) -> SoftQNetwork:
        return self.__soft_q_net1

    @property
    def soft_q_net2(self) -> SoftQNetwork:
        return self.__soft_q_net2

    @property
    def policy_net(self) -> PolicyNetwork:
        return self.__policy_net

    @property
    def norm_mean(self) -> List[float]:
        return self.__norm_mean

    @property
    def norm_std(self) -> List[float]:
        return self.__norm_std
    
    @property
    def normalized(self) -> bool:
        return self.__normalized

    @property
    def r_norm_mean(self) -> float:
        return self.__r_norm_mean

    @property
    def r_norm_std(self) -> float:
        return self.__r_norm_std

    @property
    def replay_buffer(self) -> ReplayBuffer:
        return self.__replay_buffer

    @property
    def alpha(self) -> float:
        return self.__alpha

    @property
    def soft_q_criterion(self) -> nn.SmoothL1Loss:
        return self.__soft_q_criterion

    @property
    def target_soft_q_net1(self) -> SoftQNetwork:
        return self.__target_soft_q_net1

    @property
    def target_soft_q_net2(self) -> SoftQNetwork:
        return self.__target_soft_q_net2

    @property
    def soft_q_optimizer1(self) -> optim.Adam:
        return self.__soft_q_optimizer1

    @property
    def soft_q_optimizer2(self) -> optim.Adam:
        return self.__soft_q_optimizer2

    @property
    def policy_optimizer(self) -> optim.Adam:
        return self.__policy_optimizer

    @property
    def target_entropy(self) -> float:
        return self.__target_entropy

    @action_spaces.setter
    def action_spaces(self, action_spaces: spaces.Box):
        self.__action_spaces = action_spaces

    @observation_spaces.setter
    def observation_spaces(self, observation_spaces: spaces.Box):
        self.__observation_spaces = observation_spaces

    @encoders.setter
    def encoders(self, encoders: List[Encoder]):
        if encoders is None:
            self.__encoders = [NoNormalization for _ in self.observation_spaces.shape[0]]
        else:
            self.__encoders = encoders

    @hidden_dimension.setter
    def hidden_dimension(self,  hidden_dimension: List[float]):
        self.__hidden_dimension = hidden_dimension

    @discount.setter
    def discount(self, discount: float):
        self.__discount = discount

    @tau.setter
    def tau(self, tau: float):
        self.__tau = tau
    
    @lr.setter
    def lr(self, lr: float):
        self.__lr = lr

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self.__batch_size = batch_size

    @replay_buffer_capacity.setter
    def replay_buffer_capacity(self, replay_buffer_capacity: int):
        self.__replay_buffer_capacity = replay_buffer_capacity

    @start_training_time_step.setter
    def start_training_time_step(self, start_training_time_step: int):
        self.__start_training_time_step = start_training_time_step

    @end_exploration_time_step.setter
    def end_exploration_time_step(self, end_exploration_time_step: int):
        self.__end_exploration_time_step = end_exploration_time_step

    @deterministic_start_time_step.setter
    def deterministic_start_time_step(self, deterministic_start_time_step: int):
        self.__deterministic_start_time_step = deterministic_start_time_step

    @action_scaling_coef.setter
    def action_scaling_coef(self, action_scaling_coef: float):
        self.__action_scaling_coef = action_scaling_coef

    @reward_scaling.setter
    def reward_scaling(self, reward_scaling: float):
        self.__reward_scaling = reward_scaling

    @update_per_time_step.setter
    def update_per_time_step(self, update_per_time_step: int):
        self.__update_per_time_step = update_per_time_step

    @seed.setter
    def seed(self, seed: int):
        self.__seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def add_to_buffer(self, states: List[float], actions: List[float], reward: float, next_states: List[float], done: bool = False):
        # Run once the regression model has been fitted
        # Normalize all the states using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes states that are not necessary (solar radiation if there are no solar PV panels).
        self.reward = reward
        states = np.array(self.__get_encoded_states(states), dtype = float)
        next_states = np.array(self.__get_encoded_states(next_states), dtype = float)

        if self.normalized:
            states = np.array(self.__get_normalized_states(states), dtype = float)
            next_states = np.array(self.__get_normalized_states(states), dtype = float)
            reward = self.__get_normalized_reward(reward)
        else:
            pass

        self.__replay_buffer.push(states, actions, reward, next_states, done)

        if self.time_step >= self.start_training_time_step and self.batch_size <= len(self.__replay_buffer):
            if not self.normalized:
                X = np.array([j[0] for j in self.__replay_buffer.buffer], dtype = float)
                self.__norm_mean = np.nanmean(X, axis = 0)
                self.__norm_std = np.nanstd(X, axis = 0) + 1e-5
                R = np.array([j[2] for j in self.__replay_buffer.buffer], dtype = float)
                self.__r_norm_mean = np.nanmean(R, dtype = float)
                self.__r_norm_std = np.nanstd(R, dtype = float)/self.reward_scaling + 1e-5
                new_buffer = [(
                    np.hstack((np.array(self.__get_normalized_states(states), dtype = float)).reshape(1,-1)[0]),
                    actions,
                    self.__get_normalized_reward(reward),
                    np.hstack((np.array(self.__get_normalized_states(next_states), dtype = float)).reshape(1,-1)[0]),
                    done
                ) for states, actions, reward, next_states, done in self.__replay_buffer.buffer]
                self.__replay_buffer.buffer = new_buffer
                self.__normalized = True
            else:
                pass

            for _ in range(self.update_per_time_step):
                states, actions, reward, next_states, done = self.__replay_buffer.sample(self.batch_size)
                tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                states = tensor(states).to(self.device)
                next_states = tensor(next_states).to(self.device)
                actions = tensor(actions).to(self.device)
                reward = tensor(reward).unsqueeze(1).to(self.device)
                done = tensor(done).unsqueeze(1).to(self.device)

                with torch.no_grad():
                    # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) state and its associated log probability of occurrence.
                    new_next_actions, new_log_pi, _ = self.__policy_net.sample(next_states)

                    # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
                    target_q_values = torch.min(
                        self.__target_soft_q_net1(next_states, new_next_actions),
                        self.__target_soft_q_net2(next_states, new_next_actions),
                    ) - self.alpha*new_log_pi
                    q_target = reward + (1 - done)*self.discount*target_q_values

                # Update Soft Q-Networks
                q1_pred = self.__soft_q_net1(states, actions)
                q2_pred = self.__soft_q_net2(states, actions)
                q1_loss = self.__soft_q_criterion(q1_pred, q_target)
                q2_loss = self.__soft_q_criterion(q2_pred, q_target)
                self.__soft_q_optimizer1.zero_grad()
                q1_loss.backward()
                self.__soft_q_optimizer1.step()
                self.__soft_q_optimizer2.zero_grad()
                q2_loss.backward()
                self.__soft_q_optimizer2.step()

                # Update Policy
                new_actions, log_pi, _ = self.__policy_net.sample(states)
                q_new_actions = torch.min(
                    self.__soft_q_net1(states, new_actions),
                    self.__soft_q_net2(states, new_actions)
                )
                policy_loss = (self.alpha*log_pi - q_new_actions).mean()
                self.__policy_optimizer.zero_grad()
                policy_loss.backward()
                self.__policy_optimizer.step()

                # Soft Updates
                for target_param, param in zip(self.__target_soft_q_net1.parameters(), self.__soft_q_net1.parameters()):
                    target_param.data.copy_(target_param.data*(1.0 - self.tau) + param.data*self.tau)

                for target_param, param in zip(self.__target_soft_q_net2.parameters(), self.__soft_q_net2.parameters()):
                    target_param.data.copy_(target_param.data*(1.0 - self.tau) + param.data * self.tau)

        else:
            pass

    def select_actions(self, states: List[float], **kwargs):        
        if self.time_step <= self.end_exploration_time_step:
            actions = self.get_exploration_actions(**kwargs)
        
        else:
            actions = self.__get_post_exploration_actions(states)

        self.actions = actions
        self.next_time_step()
        return actions

    def __get_post_exploration_actions(self, states: List[float]) -> List[float]:
        states = np.array(self.__get_encoded_states(states), dtype = float)
        states = np.array(self.__get_normalized_states(states), dtype = float)
        states = torch.FloatTensor(states).unsqueeze(0).to(self.__device)
        result = self.__policy_net.sample(states)
        actions = result[2] if self.time_step >= self.deterministic_start_time_step else result[0]
        actions = actions.detach().cpu().numpy()[0]
        return list(actions)
            
    def get_exploration_actions(self) -> List[float]:
        # random actions
        return list(self.action_scaling_coef*self.action_spaces.sample())

    def __get_normalized_reward(self, reward: float) -> float:
        return (reward - self.r_norm_mean)/self.r_norm_std

    def __get_normalized_states(self, states: List[float]) -> List[float]:
        return ((np.array(states, dtype = float) - self.norm_mean)/self.norm_std).tolist()

    def __get_encoded_states(self, states: List[float]) -> List[float]:
        return np.array([j for j in np.hstack(self.encoders*np.array(states, dtype = float)) if j != None], dtype = float).tolist()

    def __set_networks(self):
        # init networks
        self.__soft_q_net1 = SoftQNetwork(self.observation_dimension, self.action_dimension, self.hidden_dimension).to(self.device)
        self.__soft_q_net2 = SoftQNetwork(self.observation_dimension, self.action_dimension, self.hidden_dimension).to(self.device)
        self.__target_soft_q_net1 = SoftQNetwork(self.observation_dimension, self.action_dimension, self.hidden_dimension).to(self.device)
        self.__target_soft_q_net2 = SoftQNetwork(self.observation_dimension, self.action_dimension, self.hidden_dimension).to(self.device)

        for target_param, param in zip(self.__target_soft_q_net1.parameters(), self.__soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.__target_soft_q_net2.parameters(), self.__soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        # Policy
        self.__policy_net = PolicyNetwork(self.observation_dimension, self.action_dimension, self.action_spaces, self.action_scaling_coef, self.hidden_dimension).to(self.device)
        self.__soft_q_optimizer1 = optim.Adam(self.__soft_q_net1.parameters(), lr = self.lr)
        self.__soft_q_optimizer2 = optim.Adam(self.__soft_q_net2.parameters(), lr = self.lr)
        self.__policy_optimizer = optim.Adam(self.__policy_net.parameters(), lr = self.lr)
        self.__target_entropy = -np.prod(self.action_dimension).item()

class SAC_RBC(SAC):
    def __init__(self, action_spaces: spaces.Box, observation_spaces: spaces.Box, rbc: RBC, **kwargs):
        super().__init__(action_spaces, observation_spaces, **kwargs)
        self.rbc = rbc

    @property
    def rbc(self) -> RBC:
        return self.__rbc

    @rbc.setter
    def rbc(self, rbc: RBC):
        self.__rbc = rbc

    def get_exploration_actions(self, **kwargs) -> List[float]:
        return self.rbc.select_actions(**kwargs)

class SAC_BasicRBC(SAC_RBC):
     def __init__(self, action_spaces: spaces.Box, observation_spaces: spaces.Box, **kwargs):
        super().__init__(action_spaces, observation_spaces, **kwargs)
        self.rbc = BasicRBC(action_dimesion = self.action_dimension)

class SAC_OptimizedRBC(SAC_RBC):
     def __init__(self, action_spaces: spaces.Box, observation_spaces: spaces.Box, **kwargs):
        super().__init__(action_spaces, observation_spaces, **kwargs)
        self.rbc = OptimizedRBC(action_dimesion = self.action_dimension)