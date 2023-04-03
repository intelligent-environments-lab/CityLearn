import itertools
from typing import List, Mapping
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, spaces, Wrapper
import numpy as np
from citylearn.citylearn import CityLearnEnv

class NormalizedObservationWrapper(ObservationWrapper):
    def __init__(self, env: CityLearnEnv) -> None:
        super().__init__(env)
    
    @property
    def observation_space(self) -> List[spaces.Box]:
        low_limit = []
        high_limit = []

        if self.env.central_agent:
            for i, b in enumerate(self.env.buildings):
                s = b.estimate_observation_space(normalize=True)
                l, _ = b.normalized_observation_space_limits

                for k, lv, hv in zip(l, s.low, s.high):
                    if i == 0 or k.rstrip('_sin').rstrip('_cos') not in self.env.shared_observations:
                        low_limit.append(lv)
                        high_limit.append(hv)

                    else:
                        pass
            
            observation_space = [spaces.Box(low=np.array(low_limit), high=np.array(high_limit), dtype=np.float32)]

        else:
            observation_space = [b.estimate_observation_space(normalize=True) for b in self.env.buildings]
        
        return observation_space

    def observation(self, observations: List[List[float]]) -> List[List[float]]:
        return [[
            v for i, b in enumerate(self.env.buildings) for k, v in b.observations(normalize=True).items() 
            if i == 0 or k.rstrip('_sin').rstrip('_cos') not in self.env.shared_observations
        ]] if self.env.central_agent else [list(b.observations(normalize=True).values()) for b in self.env.buildings]
    
class DiscreteObservationWrapper(ObservationWrapper):
    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None):
        super().__init__(env)
        assert bin_sizes is None or len(bin_sizes) == len(self.env.buildings), 'length of bin_size must equal number of buildings.'
        self.bin_sizes = [{} for _ in self.env.buildings] if bin_sizes is None else bin_sizes
        self.default_bin_size = 10 if default_bin_size is None else default_bin_size
        self.bin_sizes = [
            {o: s.get(o, self.default_bin_size) for o in b.active_observations} 
            for b, s in zip(self.env.buildings, self.bin_sizes)
        ]
        
    @property
    def observation_space(self) -> List[spaces.MultiDiscrete]:
        if self.env.central_agent:
            bin_sizes = []

            for i, b in enumerate(self.bin_sizes):
                for k, v in b.items():
                    if i == 0 or k.rstrip('_sin').rstrip('_cos') not in self.env.shared_observations:
                        bin_sizes.append(v)

                    else:
                        pass
            
            observation_space = [spaces.MultiDiscrete(bin_sizes)]

        else:
            observation_space = [spaces.MultiDiscrete(list(b.values())) for b in self.bin_sizes]
        
        return observation_space
    
    def observation(self, observations: List[List[float]]) -> np.ndarray:
        transformed_observations = []

        for i, (cs, ds) in enumerate(zip(self.env.unwrapped.observation_space, self.observation_space)):
            transformed_observations_ = []

            for j, (ll, hl, b) in enumerate(zip(cs.low, cs.high, ds)):
                o = np.digitize(observations[i][j], np.linspace(ll, hl, b.n), right=True)
                transformed_observations_.append(o)

            transformed_observations.append(transformed_observations_)
                
        return transformed_observations
    
class DiscreteActionWrapper(ActionWrapper):
    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None):
        super().__init__(env)
        assert bin_sizes is None or len(bin_sizes) == len(self.env.buildings), 'length of bin_size must equal number of buildings.'
        self.bin_sizes = [{} for _ in self.env.buildings] if bin_sizes is None else bin_sizes
        self.default_bin_size = 10 if default_bin_size is None else default_bin_size
        self.bin_sizes = [
            {a: s.get(a, self.default_bin_size) for a in b.active_actions} 
            for b, s in zip(self.env.buildings, self.bin_sizes)
        ]
        
    @property
    def action_space(self) -> List[spaces.MultiDiscrete]:
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
        transformed_actions = []

        for i, (cs, ds) in enumerate(zip(self.env.unwrapped.action_space, self.action_space)):
            transformed_actions_ = []
            
            for j, (ll, hl, b) in enumerate(zip(cs.low, cs.high, ds)):
                a = np.linspace(ll, hl, b.n)[actions[i][j]]
                transformed_actions_.append(a)
            
            transformed_actions.append(transformed_actions_)

        return transformed_actions
    
class DiscreteSpaceWrapper(Wrapper):
    def __init__(self, env: CityLearnEnv, observation_bin_sizes: List[Mapping[str, int]] = None, action_bin_sizes: List[Mapping[str, int]] = None):
        env = DiscreteObservationWrapper(env, bin_sizes=observation_bin_sizes)
        env = DiscreteActionWrapper(env, bin_sizes=action_bin_sizes)
        super().__init__(env)

class TabularQLearningObservationWrapper(ObservationWrapper):
    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None) -> None:
        env = DiscreteObservationWrapper(env, bin_sizes=bin_sizes)
        super().__init__(env)
        self.combinations = self.set_combinations()

    @property
    def observation_space(self) -> List[spaces.Discrete]:
        observation_space = []

        for c in self.combinations:
            observation_space.append(spaces.Discrete(len(c) - 1))
        
        return observation_space
    
    def observation(self, observations: List[List[int]]) -> List[List[int]]:
        return [[c.index(tuple(o))] for o, c in zip(observations, self.combinations)]
    
    def set_combinations(self) -> List[List[int]]:
        combs_list = []

        for s in self.env.observation_space:
            options = [list(range(d.n + 1)) for d in s]
            combs = list(itertools.product(*options))
            combs_list.append(combs)

        return combs_list
    
class TabularQLearningActionWrapper(ActionWrapper):
    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None) -> None:
        env = DiscreteActionWrapper(env, bin_sizes=bin_sizes)
        super().__init__(env)
        self.combinations = self.set_combinations()

    @property
    def action_space(self) -> List[spaces.Discrete]:
        action_space = []

        for c in self.combinations:
            action_space.append(spaces.Discrete(len(c)))
        
        return action_space
    
    def action(self, actions: List[float]) -> List[List[int]]:
        return [list(c[a[0]]) for a, c in zip(actions, self.combinations)]
    
    def set_combinations(self) -> List[List[int]]:
        combs_list = []

        for s in self.env.action_space:
            options = [list(range(d.n)) for d in s]
            combs = list(itertools.product(*options))
            combs_list.append(combs)

        return combs_list
    
class TabularQLearningWrapper(Wrapper):
    def __init__(self, env: CityLearnEnv, observation_bin_sizes: List[Mapping[str, int]] = None, action_bin_sizes: List[Mapping[str, int]] = None):
        env = TabularQLearningObservationWrapper(env, bin_sizes=observation_bin_sizes)
        env = TabularQLearningActionWrapper(env, bin_sizes=action_bin_sizes)
        super().__init__(env)

class StableBaselines3ActionWrapper(ActionWrapper):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    @property
    def action_space(self) -> spaces.Box:
        return self.env.action_space[0]

    def action(self, actions: List[float]) -> List[List[float]]:
        return [actions]
    
class StableBaselines3RewardWrapper(RewardWrapper):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def reward(self, reward: List[float]) -> float:
        return reward[0]
    
class StableBaselines3ObservationWrapper(ObservationWrapper):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        
    @property
    def observation_space(self) -> spaces.Box:
        return self.env.observation_space[0]
    
    def observation(self, observations: List[List[float]]) -> np.ndarray:
        return np.array(observations[0], dtype='float32')

class StableBaselines3Wrapper(Wrapper):
    def __init__(self, env: CityLearnEnv):
        assert env.central_agent, 'The StableBaselines3Wrapper wrapper is compatible only when env.central_agent = True.'\
            ' First set env.central_agent = True to use this wrapper.'

        env = StableBaselines3ActionWrapper(env)
        env = StableBaselines3RewardWrapper(env)
        env = StableBaselines3ObservationWrapper(env)
        super().__init__(env)