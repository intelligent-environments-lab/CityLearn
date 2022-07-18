from typing import List
from gym import spaces
import numpy as np
from citylearn.agents.base import Agent
from citylearn.preprocessing import Encoder, NoNormalization

# conditional imports
try:
    import torch
except (ModuleNotFoundError, ImportError) as e:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

class RLC(Agent):
    def __init__(
        self, *args, observation_space: spaces.Box, encoders: List[Encoder] = None, hidden_dimension: List[float] = None, 
        discount: float = None, tau: float = None, lr: float = None, batch_size: int = None,
        replay_buffer_capacity: int = None, start_training_time_step: int = None, end_exploration_time_step: int = None, 
        deterministic_start_time_step: int = None, action_scaling_coefficienct: float = None, reward_scaling: float = None, 
        update_per_time_step: int = None, seed: int = None, **kwargs
    ):
        r"""Initialize `RLC`.

        Base reinforcement learning controller class.

        Parameters
        ----------
        *args : tuple
            `Agent` positional arguments.
        observation_space : spaces.Box
            Format of valid observations.
        encoders : List[Encoder], optional
            Observation value transformers/encoders.
        hidden_dimension : List[float], default: [256, 256]
            Hidden dimension.
        discount : float, default: 0.99
            Discount factor.
        tau : float, default: 5e-3
            Exploration-exploitation trade-off.
        lr : float, default: 3e-4
            Learning rate.
        batch_size : int, default: 256
            Batch size.
        replay_buffer_capacity : int, default: 1e5
            Replay buffer capacity.
        start_training_time_step : int, default: 6000
            Time step to begin training regression model.
        end_exploration_time_step : int, default: 7000
            Time step to stop exploration.
        deterministic_start_time_step : int, default: 7500
            Time step to begin taking deterministic actions.
        action_scaling_coefficient : float, default: 0.5
            Action scaling coefficient.
        reward_scaling : float, default: 5.0
            Reward scaling.
        update_per_time_step : int, default: 2
            Number of updates per time step.
        seed : int
            Pseudorandom number generator seed for repeatable results.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.observation_space = observation_space
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
        self.action_scaling_coefficient = action_scaling_coefficienct
        self.reward_scaling = reward_scaling
        self.update_per_time_step = update_per_time_step
        self.seed = seed

    @property
    def encoders(self) -> List[Encoder]:
        """Observation value transformers/encoders."""

        return self.__encoders

    @property
    def observation_dimension(self) -> int:
        """Number of observations after applying `encoders`."""

        return len([j for j in np.hstack(self.encoders*np.ones(len(self.observation_space.low))) if j != None])

    @property
    def observation_space(self) -> spaces.Box:
        """Format of valid observations."""

        return self.__observation_space

    @property
    def hidden_dimension(self) -> List[float]:
        """Hidden dimension."""

        return self.__hidden_dimension

    @property
    def discount(self) -> float:
        """Discount factor."""

        return self.__discount

    @property
    def tau(self) -> float:
        """Exploration-exploitation trade-off."""

        return self.__tau
    
    @property
    def lr(self) -> float:
        """Learning rate."""

        return self.__lr

    @property
    def batch_size(self) -> int:
        """Batch size."""

        return self.__batch_size

    @property
    def replay_buffer_capacity(self) -> int:
        """Replay buffer capacity."""

        return self.__replay_buffer_capacity

    @property
    def start_training_time_step(self) -> int:
        """Time step to begin training regression model."""

        return self.__start_training_time_step

    @property
    def end_exploration_time_step(self) -> int:
        """Time step to stop exploration."""

        return self.__end_exploration_time_step

    @property
    def deterministic_start_time_step(self) -> int:
        """Time step to begin taking deterministic actions."""

        return self.__deterministic_start_time_step

    @property
    def action_scaling_coefficient(self) -> float:
        """Action scaling coefficient."""

        return self.__action_scaling_coefficient

    @property
    def reward_scaling(self) -> float:
        """Reward scaling."""

        return self.__reward_scaling

    @property
    def update_per_time_step(self) -> int:
        """Number of updates per time step."""

        return self.__update_per_time_step

    @property
    def seed(self) -> int:
        """Pseudorandom number generator seed for repeatable results."""

        return self.__seed

    @encoders.setter
    def encoders(self, encoders: List[Encoder]):
        self.__encoders = [NoNormalization() for _ in range(self.observation_space.shape[0])] if encoders is None else encoders

    @observation_space.setter
    def observation_space(self, observation_space: spaces.Box):
        self.__observation_space = observation_space

    @hidden_dimension.setter
    def hidden_dimension(self,  hidden_dimension: List[float]):
        self.__hidden_dimension = [256, 256] if hidden_dimension is None else hidden_dimension

    @discount.setter
    def discount(self, discount: float):
        self.__discount = 0.99 if discount is None else discount

    @tau.setter
    def tau(self, tau: float):
        self.__tau = 5e-3 if tau is None else tau
    
    @lr.setter
    def lr(self, lr: float):
        self.__lr = 3e-4 if lr is None else lr

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self.__batch_size = 256 if batch_size is None else batch_size

    @replay_buffer_capacity.setter
    def replay_buffer_capacity(self, replay_buffer_capacity: int):
        self.__replay_buffer_capacity = 1e5 if replay_buffer_capacity is None else replay_buffer_capacity

    @start_training_time_step.setter
    def start_training_time_step(self, start_training_time_step: int):
        self.__start_training_time_step = 6000 if start_training_time_step is None else start_training_time_step

    @end_exploration_time_step.setter
    def end_exploration_time_step(self, end_exploration_time_step: int):
        self.__end_exploration_time_step = 7000 if end_exploration_time_step is None else end_exploration_time_step

    @deterministic_start_time_step.setter
    def deterministic_start_time_step(self, deterministic_start_time_step: int):
        self.__deterministic_start_time_step = np.nan if deterministic_start_time_step is None else deterministic_start_time_step

    @action_scaling_coefficient.setter
    def action_scaling_coefficient(self, action_scaling_coefficient: float):
        self.__action_scaling_coefficient = 0.5 if action_scaling_coefficient is None else action_scaling_coefficient

    @reward_scaling.setter
    def reward_scaling(self, reward_scaling: float):
        self.__reward_scaling = 5.0 if reward_scaling is None else reward_scaling

    @update_per_time_step.setter
    def update_per_time_step(self, update_per_time_step: int):
        update_per_time_step = 2 if update_per_time_step is None else update_per_time_step
        assert isinstance(update_per_time_step,int), f'update_per_time_step mut be int type. {update_per_time_step} is of {type(update_per_time_step)} type'
        self.__update_per_time_step = update_per_time_step

    @seed.setter
    def seed(self, seed: int):
        self.__seed = 0 if seed is None else seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)