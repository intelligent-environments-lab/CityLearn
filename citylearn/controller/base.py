import inspect
from typing import Any, List
from gym import spaces
from citylearn.base import Environment

class Controller(Environment):
    def __init__(self, action_spaces: spaces.Box, **kwargs):
        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key:value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))
        }
        super().__init__(**kwargs)
        self.action_spaces = action_spaces

    @property
    def action_spaces(self) -> spaces.Box:
        return self.__action_spaces

    @property
    def action_dimension(self) -> int:
        return self.action_spaces.shape[0]

    @property
    def actions(self) -> List[List[Any]]:
        return self.__actions

    @action_spaces.setter
    def action_spaces(self, action_spaces: spaces.Box):
        self.__action_spaces = action_spaces

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