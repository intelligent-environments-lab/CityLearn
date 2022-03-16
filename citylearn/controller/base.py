from typing import Any, List
import numpy as np
from citylearn.base import Environment
from citylearn.citylearn import District
from citylearn.reward import Reward

class Controller(Environment):
    def __init__(self, index: int, district: District, reward_function: Reward, **kwargs):
        super().__init__(**kwargs)
        self.__index = index
        self.__district = district
        self.__reward_function = reward_function

    @property
    def index(self) -> int:
        return self.__index

    @property
    def district(self) -> District:
        return self.__district

    @property
    def reward_function(self) -> Reward:
        return self.__reward_function

    @property
    def action_dimension(self) -> int:
        return self.district.action_spaces[self.index].shape[0]

    @property
    def actions(self) -> List[List[Any]]:
        return self.__actions

    @property
    def rewards(self) -> List[float]:
        return self.reward_function(self.index, self.district).get()

    @actions.setter
    def actions(self, actions: List[Any]):
        self.__actions[self.time_step] = actions

    def select_actions(self):
        raise NotImplementedError

    def add_to_buffer(self, *args, **kwargs):
        pass

    def next_time_step(self):
        super().next_time_step()
        self.__actions.append([])

    def reset(self):
        super().reset()
        self.__actions = [[]]