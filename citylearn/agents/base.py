import inspect
from typing import Any, List, Mapping
from gym import spaces
from citylearn.base import Environment
from citylearn.preprocessing import Encoder, PeriodicNormalization, Normalize, OnehotEncoding

class Agent(Environment):
    def __init__(self, observation_names: List[List[str]], observation_space: List[spaces.Box], action_space: List[spaces.Box], building_information: List[Mapping[str, Any]], **kwargs):
        r"""Initialize `Agent`.

        Parameters
        ----------
        observation_names: List[List[str]]
            Names of active observations that can be used to map observation values.
        observation_space : List[spaces.Box]
            Format of valid observations.
        action_space : List[spaces.Box]
            Format of valid actions.
         building_information : List[Mapping[str, Any]]
            Building metadata.

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
        self.observation_names = observation_names
        self.observation_space = observation_space
        self.action_space = action_space
        self.building_information = building_information
        self.encoders = self.set_encoders()
        super().__init__(**kwargs)

    @property
    def observation_names(self) -> List[List[str]]:
        """Names of active observations that can be used to map observation values."""

        return self.__observation_names

    @property
    def observation_space(self) -> List[spaces.Box]:
        """Format of valid observations."""

        return self.__observation_space

    @property
    def action_space(self) -> List[spaces.Box]:
        """Format of valid actions."""

        return self.__action_space

    @property
    def building_information(self) -> List[Mapping[str, Any]]:
        """Building metadata."""

        return self.__building_information

    @property
    def action_dimension(self) -> List[int]:
        """Number of returned actions."""

        return [s.shape[0] for s in self.action_space]

    @property
    def actions(self) -> List[List[List[Any]]]:
        """Action history/time series."""

        return self.__actions

    @observation_names.setter
    def observation_names(self, observation_names: List[List[str]]):
        self.__observation_names = observation_names

    @observation_space.setter
    def observation_space(self, observation_space: List[spaces.Box]):
        self.__observation_space = observation_space

    @action_space.setter
    def action_space(self, action_space: List[spaces.Box]):
        self.__action_space = action_space

    @building_information.setter
    def building_information(self, building_information: List[Mapping[str, Any]]):
        self.__building_information = building_information

    @actions.setter
    def actions(self, actions: List[List[Any]]):
        for i in range(len(self.action_space)):
            self.__actions[i][self.time_step] = actions[i]

    def select_actions(self,  observations: List[List[float]]) -> List[List[float]]:
        """Provide actions for current time step.

        Return randomly sampled actions from `action_space`.
        
        Returns
        -------
        actions: List[List[float]]
            Action values
        """

        actions = [list(s.sample()) for s in self.action_space]
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

    def set_encoders(self) -> List[List[Encoder]]:
        r"""Get observation value transformers/encoders for use in agent algorithm.

        The encoder classes are defined in the `preprocessing.py` module and include `PeriodicNormalization` for cyclic observations,
        `OnehotEncoding` for categorical obeservations, `RemoveFeature` for non-applicable observations given available storage systems and devices
        and `Normalize` for observations with known minimum and maximum boundaries.
        
        Returns
        -------
        encoders : List[List[Encoder]]
            Encoder classes for observations ordered with respect to `active_observations`.
        """

        encoders = []

        for o, s in zip(self.observation_names, self.observation_space):
            e = []

            for i, n in enumerate(o):
                if n in ['month', 'hour']:
                    e.append(PeriodicNormalization(s.high[i]))
            
                elif n == 'day_type':
                    e.append(OnehotEncoding([1, 2, 3, 4, 5, 6, 7, 8]))
            
                elif n == "daylight_savings_status":
                    e.append(OnehotEncoding([0, 1]))
            
                else:
                    e.append(Normalize(s.low[i], s.high[i]))

            encoders.append(e)

        return encoders

    def next_time_step(self):
        super().next_time_step()

        for i in range(len(self.action_space)):
            self.__actions[i].append([])

    def reset(self):
        super().reset()
        self.__actions = [[[]] for _ in self.action_space]