from typing import Any, List
from citylearn.base import Environment

class Controller(Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def actions(self) -> List[List[Any]]:
        return self.__actions

    @actions.setter
    def actions(self, actions: List[Any]):
        self.__actions[self.time_step] = actions

    def select_actions(self):
        raise NotImplementedError

    def next_time_step(self):
        super().next_time_step()
        self.__actions.append([])

    def reset(self):
        super().reset()
        self.__actions = [[]]