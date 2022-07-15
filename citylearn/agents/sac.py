from typing import List
import numpy as np
from citylearn.agents.rbc import RBC, BasicRBC, OptimizedRBC
from citylearn.agents.rlc import RLC
from citylearn.rl import PolicyNetwork, ReplayBuffer, SoftQNetwork

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except (ModuleNotFoundError, ImportError) as e:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")


class SAC(RLC):
    def __init__(self, *args, **kwargs):
        r"""Initialize :class:`SAC`.

        Parameters
        ----------
        *args : tuple
            `RLC` positional arguments.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)

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
    def device(self) -> torch.device:
        """Device; cuda or cpu."""

        return self.__device

    @property
    def soft_q_net1(self) -> SoftQNetwork:
        """soft_q_net1."""

        return self.__soft_q_net1

    @property
    def soft_q_net2(self) -> SoftQNetwork:
        """soft_q_net2."""

        return self.__soft_q_net2

    @property
    def policy_net(self) -> PolicyNetwork:
        """policy_net."""

        return self.__policy_net

    @property
    def norm_mean(self) -> List[float]:
        """norm_mean."""

        return self.__norm_mean

    @property
    def norm_std(self) -> List[float]:
        """norm_std."""
        
        return self.__norm_std
    
    @property
    def normalized(self) -> bool:
        """normalized."""

        return self.__normalized

    @property
    def r_norm_mean(self) -> float:
        """r_norm_mean."""

        return self.__r_norm_mean

    @property
    def r_norm_std(self) -> float:
        """r_norm_std."""

        return self.__r_norm_std

    @property
    def replay_buffer(self) -> ReplayBuffer:
        """replay_buffer."""

        return self.__replay_buffer

    @property
    def alpha(self) -> float:
        """alpha."""

        return self.__alpha

    @property
    def soft_q_criterion(self) -> nn.SmoothL1Loss:
        """soft_q_criterion."""
        
        return self.__soft_q_criterion

    @property
    def target_soft_q_net1(self) -> SoftQNetwork:
        """target_soft_q_net1."""

        return self.__target_soft_q_net1

    @property
    def target_soft_q_net2(self) -> SoftQNetwork:
        """target_soft_q_net2."""

        return self.__target_soft_q_net2

    @property
    def soft_q_optimizer1(self) -> optim.Adam:
        """soft_q_optimizer1."""

        return self.__soft_q_optimizer1

    @property
    def soft_q_optimizer2(self) -> optim.Adam:
        """soft_q_optimizer2."""

        return self.__soft_q_optimizer2

    @property
    def policy_optimizer(self) -> optim.Adam:
        """policy_optimizer."""

        return self.__policy_optimizer

    @property
    def target_entropy(self) -> float:
        """target_entropy."""

        return self.__target_entropy

    def add_to_buffer(self, observations: List[float], actions: List[float], reward: float, next_observations: List[float], done: bool = False):
        r"""Update replay buffer.

        Parameters
        ----------
        observations : List[float]
            Previous time step observations.
        actions : List[float]
            Previous time step actions.
        reward : float
            Current time step reward.
        next_observations : List[float]
            Current time step observations.
        done : bool
            Indication that episode has ended.
        """

        # Run once the regression model has been fitted
        # Normalize all the observations using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes observations that are not necessary (solar irradiance if there are no solar PV panels).
        observations = np.array(self.__get_encoded_observations(observations), dtype = float)
        next_observations = np.array(self.__get_encoded_observations(next_observations), dtype = float)

        if self.normalized:
            observations = np.array(self.__get_normalized_observations(observations), dtype = float)
            next_observations = np.array(self.__get_normalized_observations(observations), dtype = float)
            reward = self.__get_normalized_reward(reward)
        else:
            pass

        self.__replay_buffer.push(observations, actions, reward, next_observations, done)

        if self.time_step >= self.start_training_time_step and self.batch_size <= len(self.__replay_buffer):
            if not self.normalized:
                X = np.array([j[0] for j in self.__replay_buffer.buffer], dtype = float)
                self.__norm_mean = np.nanmean(X, axis = 0)
                self.__norm_std = np.nanstd(X, axis = 0) + 1e-5
                R = np.array([j[2] for j in self.__replay_buffer.buffer], dtype = float)
                self.__r_norm_mean = np.nanmean(R, dtype = float)
                self.__r_norm_std = np.nanstd(R, dtype = float)/self.reward_scaling + 1e-5
                self.__replay_buffer.buffer = [(
                    np.hstack((np.array(self.__get_normalized_observations(observations), dtype = float)).reshape(1,-1)[0]),
                    actions,
                    self.__get_normalized_reward(reward),
                    np.hstack((np.array(self.__get_normalized_observations(next_observations), dtype = float)).reshape(1,-1)[0]),
                    done
                ) for observations, actions, reward, next_observations, done in self.__replay_buffer.buffer]
                self.__normalized = True
            else:
                pass

            for _ in range(self.update_per_time_step):
                observations, actions, reward, next_observations, done = self.__replay_buffer.sample(self.batch_size)
                tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                observations = tensor(observations).to(self.device)
                next_observations = tensor(next_observations).to(self.device)
                actions = tensor(actions).to(self.device)
                reward = tensor(reward).unsqueeze(1).to(self.device)
                done = tensor(done).unsqueeze(1).to(self.device)

                with torch.no_grad():
                    # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) observation and its associated log probability of occurrence.
                    new_next_actions, new_log_pi, _ = self.__policy_net.sample(next_observations)

                    # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
                    target_q_values = torch.min(
                        self.__target_soft_q_net1(next_observations, new_next_actions),
                        self.__target_soft_q_net2(next_observations, new_next_actions),
                    ) - self.alpha*new_log_pi
                    q_target = reward + (1 - done)*self.discount*target_q_values

                # Update Soft Q-Networks
                q1_pred = self.__soft_q_net1(observations, actions)
                q2_pred = self.__soft_q_net2(observations, actions)
                q1_loss = self.__soft_q_criterion(q1_pred, q_target)
                q2_loss = self.__soft_q_criterion(q2_pred, q_target)
                self.__soft_q_optimizer1.zero_grad()
                q1_loss.backward()
                self.__soft_q_optimizer1.step()
                self.__soft_q_optimizer2.zero_grad()
                q2_loss.backward()
                self.__soft_q_optimizer2.step()

                # Update Policy
                new_actions, log_pi, _ = self.__policy_net.sample(observations)
                q_new_actions = torch.min(
                    self.__soft_q_net1(observations, new_actions),
                    self.__soft_q_net2(observations, new_actions)
                )
                policy_loss = (self.alpha*log_pi - q_new_actions).mean()
                self.__policy_optimizer.zero_grad()
                policy_loss.backward()
                self.__policy_optimizer.step()

                # Soft Updates
                for target_param, param in zip(self.__target_soft_q_net1.parameters(), self.__soft_q_net1.parameters()):
                    target_param.data.copy_(target_param.data*(1.0 - self.tau) + param.data*self.tau)

                for target_param, param in zip(self.__target_soft_q_net2.parameters(), self.__soft_q_net2.parameters()):
                    target_param.data.copy_(target_param.data*(1.0 - self.tau) + param.data*self.tau)

        else:
            pass

    def select_actions(self, observations: List[float]):
        r"""Provide actions for current time step.

        Will return randomly sampled actions from `action_space` if :attr:`end_exploration_time_step` >= :attr:`time_step` 
        else will use policy to sample actions.
        
        Returns
        -------
        actions: List[float]
            Action values
        """

        if self.time_step <= self.end_exploration_time_step:
            actions = self.get_exploration_actions(observations)
        
        else:
            actions = self.__get_post_exploration_actions(observations)

        self.actions = actions
        self.next_time_step()
        return actions

    def __get_post_exploration_actions(self, observations: List[float]) -> List[float]:
        """Action sampling using policy, post-exploration time step"""

        observations = np.array(self.__get_encoded_observations(observations), dtype = float)
        observations = np.array(self.__get_normalized_observations(observations), dtype = float)
        observations = torch.FloatTensor(observations).unsqueeze(0).to(self.__device)
        result = self.__policy_net.sample(observations)
        actions = result[2] if self.time_step >= self.deterministic_start_time_step else result[0]
        actions = actions.detach().cpu().numpy()[0]
        return list(actions)
            
    def get_exploration_actions(self, observations: List[float]) -> List[float]:
        """Return randomly sampled actions from `action_space` multiplied by :attr:`action_scaling_coefficient`.
        
        Returns
        -------
        actions: List[float]
            Action values.
        """

        # random actions
        return list(self.action_scaling_coefficient*self.action_space.sample())

    def __get_normalized_reward(self, reward: float) -> float:
        return (reward - self.r_norm_mean)/self.r_norm_std

    def __get_normalized_observations(self, observations: List[float]) -> List[float]:
        return ((np.array(observations, dtype = float) - self.norm_mean)/self.norm_std).tolist()

    def __get_encoded_observations(self, observations: List[float]) -> List[float]:
        return np.array([j for j in np.hstack(self.encoders*np.array(observations, dtype = float)) if j != None], dtype = float).tolist()

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
        self.__policy_net = PolicyNetwork(self.observation_dimension, self.action_dimension, self.action_space, self.action_scaling_coefficient, self.hidden_dimension).to(self.device)
        self.__soft_q_optimizer1 = optim.Adam(self.__soft_q_net1.parameters(), lr = self.lr)
        self.__soft_q_optimizer2 = optim.Adam(self.__soft_q_net2.parameters(), lr = self.lr)
        self.__policy_optimizer = optim.Adam(self.__policy_net.parameters(), lr = self.lr)
        self.__target_entropy = -np.prod(self.action_space.shape).item()

class SACRBC(SAC):
    def __init__(self, *args, **kwargs):
        r"""Initialize `SACRBC`.

        Uses :class:`RBC` to select action during exploration before using :class:`SAC`. 

        Parameters
        ----------
        *args : tuple
            :class:`SAC` positional arguments.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.rbc = RBC(action_space=self.action_space)

    @property
    def rbc(self) -> RBC:
        """:class:`RBC` or child class, used to select actions during exploration."""

        return self.__rbc

    @rbc.setter
    def rbc(self, rbc: RBC):
        self.__rbc = rbc

    def get_exploration_actions(self, states: List[float]) -> List[float]:
        """Return actions using :class:`RBC`.
        
        Returns
        -------
        actions: List[float]
            Action values.
        """

        return self.rbc.select_actions(states)

class SACBasicRBC(SACRBC):
     def __init__(self, *args, hour_index: int = None, **kwargs):
        r"""Initialize `SACRBC`.

        Uses :class:`BasicRBC` to select action during exploration before using :class:`SAC`.

        Parameters
        ----------
        *args : tuple
            :class:`SAC` positional arguments.
        hour_index: int, default: 2
            Expected position of hour observation when `observations` paramater is parsed into `select_actions` method (used in :class:`BasicRBC`).
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.rbc = BasicRBC(action_space=self.action_space, hour_index=hour_index)

class SACOptimizedRBC(SACBasicRBC):
     def __init__(self,*args, hour_index: int = None, **kwargs):
        r"""Initialize `SACOptimizedRBC`.

        Uses :class:`OptimizedRBC` to select action during exploration before using :class:`SAC`.

        Parameters
        ----------
        *args : tuple
            :class:`SAC` positional arguments.
        hour_index: int, default: 2
            Expected position of hour observation when `observations` paramater is parsed into `select_actions` method (used in :class:`OptimizedRBC`).
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.rbc = OptimizedRBC(action_space=self.action_space, hour_index=hour_index)