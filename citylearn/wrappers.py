import itertools
from typing import List, Mapping
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, spaces, Wrapper
import numpy as np
from citylearn.citylearn import CityLearnEnv

class NormalizedObservationWrapper(ObservationWrapper):
    """Wrapper for observations min-max and periodic normalization.
    
    Temporal observations including `hour`, `day_type` and `month` are periodically normalized using sine/cosine 
    transformations and then all observations are min-max normalized between 0 and 1.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv) -> None:
        super().__init__(env)
        self.env: CityLearnEnv
    
    @property
    def observation_space(self) -> List[spaces.Box]:
        """Returns observation space for normalized observations."""

        low_limit = []
        high_limit = []

        if self.env.central_agent:
            shared_observations = []

            for i, b in enumerate(self.env.buildings):
                s = b.estimate_observation_space(normalize=True, periodic_normalization=True)
                o = b.observations(normalize=True, periodic_normalization=True)

                for k, lv, hv in zip(o, s.low, s.high):
                    k_p = k.rstrip('_sin').rstrip('_cos')
                    
                    if i == 0 or k_p not in self.env.shared_observations or k not in shared_observations:
                        low_limit.append(lv)
                        high_limit.append(hv)

                    else:
                        pass

                    if k_p in self.env.shared_observations and k not in shared_observations:
                        shared_observations.append(k)
                    
                    else:
                        pass
            
            observation_space = [spaces.Box(low=np.array(low_limit), high=np.array(high_limit), dtype=np.float32)]

        else:
            observation_space = [b.estimate_observation_space(normalize=True, periodic_normalization=True) for b in self.env.buildings]
        
        return observation_space

    def observation(self, observations: List[List[float]]) -> List[List[float]]:
        """Returns normalized observations."""

        if self.env.central_agent:
            norm_observations = []
            shared_observations = []

            for i, b in enumerate(self.env.buildings):
                for k, v in b.observations(normalize=True, periodic_normalization=True).items():
                    k_p = k.rstrip('_sin').rstrip('_cos')
                    
                    if i==0 or k_p not in self.env.shared_observations or k not in shared_observations:
                        norm_observations.append(v)

                    else:
                        pass

                    if k_p in self.env.shared_observations and k not in shared_observations:
                        shared_observations.append(k)
                    
                    else:
                        pass
            
            norm_observations = [norm_observations]

        else:
            norm_observations = [list(b.observations(normalize=True, periodic_normalization=True).values()) for b in self.env.buildings]
        
        return norm_observations
    
class DiscreteObservationWrapper(ObservationWrapper):
    """Wrapper for observation space discretization.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active observation in each building.
    default_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building observation.
    """

    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None):
        super().__init__(env)
        self.env: CityLearnEnv
        assert bin_sizes is None or len(bin_sizes) == len(self.env.buildings), 'length of bin_size must equal number of buildings.'
        self.bin_sizes = [{} for _ in self.env.buildings] if bin_sizes is None else bin_sizes
        self.default_bin_size = 10 if default_bin_size is None else default_bin_size
        self.bin_sizes = [
            {o: s.get(o, self.default_bin_size) for o in b.active_observations} 
            for b, s in zip(self.env.buildings, self.bin_sizes)
        ]
        
    @property
    def observation_space(self) -> List[spaces.MultiDiscrete]:
        """Returns observation space for discretized observations."""

        if self.env.central_agent:
            bin_sizes = []
            shared_observations = []

            for i, b in enumerate(self.bin_sizes):
                for k, v in b.items():
                    
                    k_p = k.rstrip('_sin').rstrip('_cos')
                    
                    if i == 0 or k_p not in self.env.shared_observations or k not in shared_observations:
                        bin_sizes.append(v)

                    else:
                        pass

                    if k_p in self.env.shared_observations and k not in shared_observations:
                        shared_observations.append(k)

                    else:
                        pass
            
            observation_space = [spaces.MultiDiscrete(bin_sizes)]

        else:
            observation_space = [spaces.MultiDiscrete(list(b.values())) for b in self.bin_sizes]
        
        return observation_space
    
    def observation(self, observations: List[List[float]]) -> np.ndarray:
        """Returns discretized observations."""

        transformed_observations = []

        for i, (cs, ds) in enumerate(zip(self.env.unwrapped.observation_space, self.observation_space)):
            transformed_observations_ = []

            for j, (ll, hl, b) in enumerate(zip(cs.low, cs.high, ds)):
                o = np.digitize(observations[i][j], np.linspace(ll, hl, b.n), right=True)
                transformed_observations_.append(o)

            transformed_observations.append(transformed_observations_)
                
        return transformed_observations
    
class DiscreteActionWrapper(ActionWrapper):
    """Wrapper for action space discretization.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active action in each building.
    default_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building action.
    """

    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None):
        super().__init__(env)
        self.env: CityLearnEnv
        assert bin_sizes is None or len(bin_sizes) == len(self.env.buildings), 'length of bin_size must equal number of buildings.'
        self.bin_sizes = [{} for _ in self.env.buildings] if bin_sizes is None else bin_sizes
        self.default_bin_size = 10 if default_bin_size is None else default_bin_size
        self.bin_sizes = [
            {a: s.get(a, self.default_bin_size) for a in b.active_actions} 
            for b, s in zip(self.env.buildings, self.bin_sizes)
        ]
        
    @property
    def action_space(self) -> List[spaces.MultiDiscrete]:
        """Returns action space for discretized actions."""

        if self.env.central_agent:
            bin_sizes = []

            for b in self.bin_sizes:
                for _, v in b.items():
                    bin_sizes.append(v)
            
            action_space = [spaces.MultiDiscrete(bin_sizes)]

        else:
            action_space = [spaces.MultiDiscrete(list(b.values())) for b in self.bin_sizes]

        return action_space

    def action(self, actions: List[float]) -> List[List[float]]:
        """Returns discretized actions."""

        transformed_actions = []

        for i, (cs, ds) in enumerate(zip(self.env.unwrapped.action_space, self.action_space)):
            transformed_actions_ = []
            
            for j, (ll, hl, b) in enumerate(zip(cs.low, cs.high, ds)):
                a = np.linspace(ll, hl, b.n)[actions[i][j]]
                transformed_actions_.append(a)
            
            transformed_actions.append(transformed_actions_)

        return transformed_actions
    
class DiscreteSpaceWrapper(Wrapper):
    """Wrapper for observation and action spaces discretization.

    Wraps `env` in :py:class:`citylearn.wrappers.DiscreteObservationWrapper` and :py:class:`citylearn.wrappers.DiscreteActionWrapper`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    observation_bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active observation in each building.
    action_bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active action in each building.
    default_observation_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building observation.
    default_action_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building action.
    """

    def __init__(self, env: CityLearnEnv, observation_bin_sizes: List[Mapping[str, int]] = None, action_bin_sizes: List[Mapping[str, int]] = None, default_observation_bin_size: int = None, default_action_bin_size: int = None):
        env = DiscreteObservationWrapper(env, bin_sizes=observation_bin_sizes, default_bin_size=default_observation_bin_size)
        env = DiscreteActionWrapper(env, bin_sizes=action_bin_sizes, default_bin_size=default_action_bin_size)
        super().__init__(env)
        self.env: CityLearnEnv

class TabularQLearningObservationWrapper(ObservationWrapper):
    """Observation wrapper for :py:class:`citylearn.agents.q_learning.TabularQLearning` agent.

    Wraps `env` in :py:class:`citylearn.wrappers.DiscreteObservationWrapper`.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active observation in each building.
    default_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building observation.
    """

    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None) -> None:
        env = DiscreteObservationWrapper(env, bin_sizes=bin_sizes, default_bin_size=default_bin_size)
        super().__init__(env)
        self.env: CityLearnEnv
        self.combinations = self.set_combinations()

    @property
    def observation_space(self) -> List[spaces.Discrete]:
        """Returns observation space for discretized observations."""

        observation_space = []

        for c in self.combinations:
            observation_space.append(spaces.Discrete(len(c) - 1))
        
        return observation_space
    
    def observation(self, observations: List[List[int]]) -> List[List[int]]:
        """Returns discretized observations."""

        return [[c.index(tuple(o))] for o, c in zip(observations, self.combinations)]
    
    def set_combinations(self) -> List[List[int]]:
        """Returns all combinations of discrete observations."""

        combs_list = []

        for s in self.env.observation_space:
            options = [list(range(d.n + 1)) for d in s]
            combs = list(itertools.product(*options))
            combs_list.append(combs)

        return combs_list
    
class TabularQLearningActionWrapper(ActionWrapper):
    """Action wrapper for :py:class:`citylearn.agents.q_learning.TabularQLearning` agent.

    Wraps `env` in :py:class:`citylearn.wrappers.DiscreteActionWrapper`.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active action in each building.
    default_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building action.
    """

    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None) -> None:
        env = DiscreteActionWrapper(env, bin_sizes=bin_sizes, default_bin_size=default_bin_size)
        super().__init__(env)
        self.env: CityLearnEnv
        self.combinations = self.set_combinations()

    @property
    def action_space(self) -> List[spaces.Discrete]:
        """Returns action space for discretized actions."""

        action_space = []

        for c in self.combinations:
            action_space.append(spaces.Discrete(len(c)))
        
        return action_space
    
    def action(self, actions: List[float]) -> List[List[int]]:
        """Returns discretized actions."""

        return [list(c[a[0]]) for a, c in zip(actions, self.combinations)]
    
    def set_combinations(self) -> List[List[int]]:
        """Returns all combinations of discrete actions."""

        combs_list = []

        for s in self.env.action_space:
            options = [list(range(d.n)) for d in s]
            combs = list(itertools.product(*options))
            combs_list.append(combs)

        return combs_list
    
class TabularQLearningWrapper(Wrapper):
    """Wrapper for :py:class:`citylearn.agents.q_learning.TabularQLearning` agent.

    Wraps `env` in :py:class:`citylearn.wrappers.DiscreteObservationWrapper` and :py:class:`citylearn.wrappers.DiscreteActionWrapper`.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    observation_bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active observation in each building.
    action_bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active action in each building.
    default_observation_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building observation.
    default_action_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building action.
    """

    def __init__(self, env: CityLearnEnv, observation_bin_sizes: List[Mapping[str, int]] = None, action_bin_sizes: List[Mapping[str, int]] = None, default_observation_bin_size: int = None, default_action_bin_size: int = None):
        env = TabularQLearningObservationWrapper(env, bin_sizes=observation_bin_sizes, default_bin_size=default_observation_bin_size)
        env = TabularQLearningActionWrapper(env, bin_sizes=action_bin_sizes, default_bin_size=default_action_bin_size)
        super().__init__(env)
        self.env: CityLearnEnv

class StableBaselines3ObservationWrapper(ObservationWrapper):
    """Observation wrapper for :code:`stable-baselines3` algorithms.

    Wraps observations so that they are returned in a 1-dimensional numpy array.
    This wrapper is only compatible when the environment is controlled by a central agent
    i.e., :py:attr:`citylearn.citylearn.CityLearnEnv.central_agent` = True.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv):
        assert env.central_agent, 'StableBaselines3ObservationWrapper is compatible only when env.central_agent = True.'\
            ' First set env.central_agent = True to use this wrapper.'
        
        super().__init__(env)
        self.env: CityLearnEnv
        
    @property
    def observation_space(self) -> spaces.Box:
        """Returns single spaces Box object."""

        return self.env.observation_space[0]
    
    def observation(self, observations: List[List[float]]) -> np.ndarray:
        """Returns observations as 1-dimensional numpy array."""

        return np.array(observations[0], dtype='float32')

class StableBaselines3ActionWrapper(ActionWrapper):
    """Action wrapper for :code:`stable-baselines3` algorithms.

    Wraps actions so that they are returned in a 1-dimensional numpy array.
    This wrapper is only compatible when the environment is controlled by a central agent
    i.e., :py:attr:`citylearn.citylearn.CityLearnEnv.central_agent` = True.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv):
        assert env.central_agent, 'StableBaselines3ActionWrapper is compatible only when env.central_agent = True.'\
            ' First set env.central_agent = True to use this wrapper.'
        
        super().__init__(env)
        self.env: CityLearnEnv

    @property
    def action_space(self) -> spaces.Box:
        """Returns single spaces Box object."""

        return self.env.action_space[0]

    def action(self, actions: List[float]) -> List[List[float]]:
        """Returns actions as 1-dimensional numpy array."""

        return [actions]
    
class StableBaselines3RewardWrapper(RewardWrapper):
    """Reward wrapper for :code:`stable-baselines3` algorithms.

    Wraps rewards so that it is returned as float value.
    This wrapper is only compatible when the environment is controlled by a central agent
    i.e., :py:attr:`citylearn.citylearn.CityLearnEnv.central_agent` = True.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv):
        assert env.central_agent, 'StableBaselines3RewardWrapper is compatible only when env.central_agent = True.'\
            ' First set env.central_agent = True to use this wrapper.'
        
        super().__init__(env)
        self.env: CityLearnEnv

    def reward(self, reward: List[float]) -> float:
        """Returns reward as float value."""

        return reward[0]

class StableBaselines3Wrapper(Wrapper):
    """Wrapper for :code:`stable-baselines3` algorithms.

    Wraps `env` in :py:class:`citylearn.wrappers.StableBaselines3ObservationWrapper`,
    :py:class:`citylearn.wrappers.StableBaselines3ActionWrapper`
    and :py:class:`citylearn.wrappers.StableBaselines3RewardWrapper`.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    """

    def __init__(self, env: CityLearnEnv):
        env = StableBaselines3ActionWrapper(env)
        env = StableBaselines3RewardWrapper(env)
        env = StableBaselines3ObservationWrapper(env)
        super().__init__(env)
        self.env: CityLearnEnv