"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will. This reward_function takes all the electrical demands and carbon intensity of all the buildings and turns them into one or multiple rewards for the agent(s)
"""
import numpy as np

# Reward used in the CityLearn Challenge. Reward function for the multi-agent (decentralized) agents.
class reward_function_ma:
    def __init__(self, n_agents, building_info):
        self.n_agents = n_agents
        self.building_info = building_info

    @classmethod
    def get_styles(cls):
        return list(cls(None,None).__get_style_functions().keys())

    # electricity_demand contains negative values when the building consumes more electricity than it generates
    def get_rewards(self, electricity_demand, carbon_intensity, style='default', **kwargs):
        try:
            func = self.__get_style_functions()[style]
        except KeyError as e:
            raise Exception(f'Invalid stlye parsed. Choose a style from {self.styles}')

        kwargs['electricity_demand'] = electricity_demand
        kwargs['carbon_intensity'] = carbon_intensity
        return func(**kwargs)

    def __get_style_functions(self):
        return {
            'default': self.__get_default_reward,
            'marlisa': self.__get_marlisa_reward,
            'ramping_square': self.__get_ramping_square_reward,
            'exponential': self.__get_exponential_reward,
            'mixed': self.__get_mixed_reward
        }

    def __get_default_reward(self,electricity_demand,carbon_intensity,**kwargs):
        # You can edit what comes next and customize it for The CityLearn Challenge
        electricity_demand = np.float32(electricity_demand)
        total_electricity_demand = 0
        for e in electricity_demand:
            total_electricity_demand += -e
            
        electricity_demand = np.array(electricity_demand)
        
        using_marlisa = False
        # Use this reward function when running the MARLISA example with information_sharing = True. The reward sent to each agent will have an individual and a collective component.
        if using_marlisa:
            return list(np.sign(electricity_demand)*0.01*(np.array(np.abs(electricity_demand))**2 * max(0, total_electricity_demand)))
        
        else:
            
            # Use this reward when running the SAC example. It assumes that the building-agents act independently of each other, without sharing information through the reward.
            reward_ = np.array(electricity_demand)**3.0
            reward_[reward_>0] = 0
            return list(reward_)

    def __get_marlisa_reward(self,**kwargs):
        default_reward = self.__get_default_reward(kwargs['electricity_demand'],kwargs['carbon_intensity'])
        reward = list(np.sign(default_reward)*0.01*(np.array(np.abs(default_reward))**2 * max(0, -np.sum(default_reward))))
        return reward

    def __get_ramping_square_reward(self,**kwargs):
        assert 'previous_electricity_demand' in kwargs.keys(), 'ramping_square reward requires previous_electricity_demand keyword argument.'
        assert 'previous_carbon_intensity' in kwargs.keys(), 'ramping_square reward requires previous_carbon_intensity keyword argument.'
        default_reward = self.__get_default_reward(kwargs['electricity_demand'],kwargs['carbon_intensity'])
        previous_default_reward = self.__get_default_reward(kwargs['previous_electricity_demand'],kwargs['previous_carbon_intensity'])\
            if kwargs['previous_electricity_demand'] and kwargs['previous_carbon_intensity'] is not None else None
        reward = -np.square(np.sum(default_reward) - np.sum(previous_default_reward)) if previous_default_reward is not None else 0
        reward = [reward/9. for _ in range(9)]
        return reward

    def __get_exponential_reward(self,**kwargs):
        scaling_factor = kwargs.get('exponential_scaling_factor',1)
        default_reward = self.__get_default_reward(kwargs['electricity_demand'],kwargs['carbon_intensity'])
        reward = -np.exp(-np.sum(default_reward)*scaling_factor/50.)
        reward = [reward/9. for _ in range(9)]
        return reward

    def __get_mixed_reward(self,**kwargs):
        assert 'previous_electricity_demand' in kwargs.keys(), 'mixed reward requires previous_electricity_demand keyword argument.'
        assert 'previous_carbon_intensity' in kwargs.keys(), 'mixed reward requires previous_carbon_intensity keyword argument.'
        scaling_factor = kwargs.get('exponential_scaling_factor',1)
        default_reward = self.__get_default_reward(kwargs['electricity_demand'],kwargs['carbon_intensity'])
        previous_default_reward = self.__get_default_reward(kwargs['previous_electricity_demand'],kwargs['previous_carbon_intensity'])\
            if kwargs['previous_electricity_demand'] and kwargs['previous_carbon_intensity'] is not None else None
        reward_1 = -np.square(np.sum(default_reward) - np.sum(previous_default_reward)) if previous_default_reward is not None else 0
        reward_2 = -np.exp(-np.sum(default_reward)*scaling_factor/50.)
        reward = [(reward_1 + reward_2)/9. for _ in range(9)]
        return reward

# Do not use or delete
# Reward function for the centralized agent. To be used only if all the buildings receive the same reward.
def reward_function_sa(electricity_demand):
    reward_ = -np.array(electricity_demand).sum()
    reward_ = max(0, reward_)
    reward_ = reward_**3.0
    
    return reward_