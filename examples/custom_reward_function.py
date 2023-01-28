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
from citylearn.reward_function import RewardFunction

class CustomReward(RewardFunction):
    def __init__(self, agent_count: int, electricity_consumption: List[float], carbon_emission: List[float], electricity_price: List[float]):
        super().__init__(agent_count, electricity_consumption=electricity_consumption, carbon_emission=carbon_emission, electricity_price=electricity_price)
        
    def calculate(self) -> List[float]:
        """Calculates custom user-defined multi-agent reward.
        
        Reward is the `carbon_emission` for each building.
        """

        return list(self.carbon_emission)