from typing import List
from citylearn.reward_function import RewardFunction

class ChallengeReward(RewardFunction):
    def __init__(self, agent_count: int, electricity_consumption: List[float], carbon_emission: List[float], electricity_price: List[float]):
        super().__init__(agent_count, electricity_consumption=electricity_consumption, carbon_emission=carbon_emission, electricity_price=electricity_price)
        
    def calculate(self) -> float:
        """CityLearn Challenge reward calculation.

        Notes
        -----
        The placeholder reward value is calculated as :math:`[\textrm{max}(-e_0, 0), \dots, \textrm{max}(-e_n, 0)]` 
        where :math:`e` is `electricity_consumption` and :math:`n` is the number of agents. Use the available properties
        to design a custom reward function as needed. The available properties include `self.electricity_consumption`,
        `self.carbon_emission` and `self.electricity_price`.
        """

        # *********** BEGIN EDIT ***********
        # Provide custom reward calculation
        # .....
        # ************** END ***************


        return super().calculate()