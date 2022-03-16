"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will. This reward_function takes all the electrical demands and carbon intensity of all the buildings and turns them into one or multiple rewards for the agent(s)
"""
from typing import List
import numpy as np
from citylearn.citylearn import Building, District

class Reward:
    def __init__(self, index: int, district: District):
        self.__index = index
        self.__district = district

    @property
    def building(self) -> Building:
        return self.district.buildings[self.index]

    @property 
    def index(self) -> int:
        return self.__index

    @property
    def district(self) -> District:
        return self.__district

    def get(self) -> List[float]:
        raise NotImplementedError

class MARL(Reward):
    def __init__(self, *args):
        super().__init__(*args)

    def get(self) -> float:
        rewards = np.sign(self.building.net_electricity_consumption)*0.01\
            *(np.array(self.building.net_electricity_consumption, dtype = float)**2)\
                *np.nanmax([np.zeros(len(self.district.net_electricity_consumption)), -np.array(self.district.net_electricity_consumption, dtype = float)])
        return list(rewards)

class CentralAgentExponentialElectricityConsumption(Reward):
    def __init__(self, *args):
        super().__init__(*args)

    def get(self) -> float:
        rewards = np.nanmax([np.zeros(len(self.district.net_electricity_consumption)), -self.district.net_electricity_consumption])**3
        return list(rewards)