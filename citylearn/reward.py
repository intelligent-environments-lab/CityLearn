"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will. This reward_function takes all the electrical demands and carbon intensity of all the buildings and turns them into one or multiple rewards for the agent(s)
"""
import numpy as np

class Reward:
    @staticmethod
    def marl(building_net_electric_consumption: float, district_net_electric_consumption) -> float:
        return np.sign(building_net_electric_consumption)*0.01*(building_net_electric_consumption**2)*max(0, -district_net_electric_consumption)

    @staticmethod
    def ramping_square(current_net_electricity_consumption: float, previous_net_electricity_consumption: float) -> float:
        return -np.square(current_net_electricity_consumption - previous_net_electricity_consumption)

    @staticmethod
    def exponential(net_electricity_consumption: float, scaling_factor: float = 1.0) -> float:
        return -np.exp(net_electricity_consumption*scaling_factor)

    @staticmethod
    def ramping_square_and_exponential(current_net_electricity_consumption: float, previous_net_electricity_consumption: float) -> float:
        return Reward.ramping_square(current_net_electricity_consumption, previous_net_electricity_consumption)\
            + Reward.exponential(current_net_electricity_consumption)

    @staticmethod
    def central_agent_exponential(district_net_electricity_consumption: float) -> float:
        return max(0, -district_net_electricity_consumption)**3