import random
from citylearn.citylearn import District
from typing import Any, List, Mapping, Tuple

class Agent:
    def __init__(self, index: int, actions_dimension: int, district: District):
        '''Initialize the class and define any hyperparameters of the controller.'''
        self.__index = index
        self.__actions_dimension = actions_dimension
        self.__district = district

    @property
    def index(self) -> int:
        return self.__index

    @property
    def actions_dimension(self) -> int:
        return self.__actions_dimension

    @property
    def district(self) -> District:
        return self.__district

    def select_actions(self, states: List[float]) -> List[float]:
        '''Action selection algorithm'''
        actions = [random.randrange(-1, 1) for _ in range(self.actions_dimension)]
        return actions

    def add_to_buffer(self, states: List[float], actions: List[float], reward: float, next_states: List[float], done: bool, **kwargs):
        '''Make any updates to your policy, you don't have to use all the variables above (you can leave the coordination
        variables empty if you wish, or use them to share information among your different agents). You can add a counter
        within this function to compute the time-step of the simulation, since it will be called once per time-step.'''
        pass

    #**************** write other supporting functions below ****************