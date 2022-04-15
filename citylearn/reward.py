"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will. This reward_function takes all the electrical demands and carbon intensity of all the buildings and turns them into one or multiple rewards for the agent(s)
"""
from typing import List
import numpy as np
from citylearn.citylearn import District

class Reward:
    def __init__(self):
        pass

    def get(self, index: int, district: District) -> List[float]:
        return None

class MARL(Reward):
    def __init__(self):
        super().__init__()

    def get(self, index: int, district: District) -> float:
        rewards = np.sign(district.buildings[index].net_electricity_consumption)*0.01\
            *(np.array(district.buildings[index].net_electricity_consumption, dtype = float)**2)\
                *np.nanmax([np.zeros(len(district.net_electricity_consumption)), -np.array(district.net_electricity_consumption, dtype = float)])
        return list(rewards)

class CentralAgentExponentialElectricityConsumption(Reward):
    def __init__(self):
        super().__init__()

    def get(self, index: int, district: District) -> float:
        rewards = np.nanmax([np.zeros(len(district.net_electricity_consumption)), -district.net_electricity_consumption])**3
        return list(rewards)