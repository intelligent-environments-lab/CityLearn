#***** HOW TO DEFINE A CUSTOM REWARD FUNCTION *****
#
# This module provides an example of how a custom
# reward function class can be defined by 
# inheriting the base RewardFunction class. Point 
# to this module in reward_function:type object in 
# schema.json to use this custom reward function
# during simulation.
#
#********************** END ***********************

from typing import List
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction

class CustomReward(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        
    def calculate(self) -> List[float]:
        """Calculates custom user-defined multi-agent reward.
        
        Reward is the :py:atter:`net_electricity_consumption_emission` for entire district if central agent setup 
        otherwise it is the :py:atter:`net_electricity_consumption_emission` each building.
        """

        if self.env.central_agent:
            reward = [self.env.net_electricity_consumption_emission[-1]]

        else:
            reward = [b.net_electricity_consumption_emission[-1] for b in self.env.buildings]

        return reward