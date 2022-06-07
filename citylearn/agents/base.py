import inspect
from typing import Any, List
from gym import spaces
from citylearn.base import Environment

class Agent(Environment):
    def __init__(self, action_space: spaces.Box, **kwargs):
        r"""Initialize `Agent`.

        Parameters
        ----------
        action_space : spaces.Box
            Format of valid actions.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        arg_spec = inspect.getfullargspec(super().__init__)
        kwargs = {
            key:value for (key, value) in kwargs.items()
            if (key in arg_spec.args or (arg_spec.varkw is not None))
        }
        super().__init__(**kwargs)
        self.action_space = action_space

    @property
    def action_space(self) -> spaces.Box:
        """Format of valid actions."""

        return self.__action_space

    @property
    def action_dimension(self) -> int:
        """Number of returned actions."""

        return self.action_space.shape[0]

    @property
    def actions(self) -> List[List[Any]]:
        """Action history/time series."""

        return self.__actions

    @action_space.setter
    def action_space(self, action_space: spaces.Box):
        self.__action_space = action_space

    @actions.setter
    def actions(self, actions: List[Any]):
        self.__actions[self.time_step] = actions

    def select_actions(self) -> List[float]:
        """Provide actions for current time step.

        Return randomly sampled actions from `action_space`.
        
        Returns
        -------
        actions: List[float]
            Action values
        """

        actions = list(self.action_space.sample())
        self.actions = actions
        self.next_time_step()
        return actions

    def add_to_buffer(self, *args, **kwargs):
        """Update replay buffer
        
        Notes
        -----
        This implementation does nothing but is kept to keep the API for all agents similar during simulation.
        """

        pass

    def next_time_step(self):
        super().next_time_step()
        self.__actions.append([])

    def reset(self):
        super().reset()
        self.__actions = [[]]