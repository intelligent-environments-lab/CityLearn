from typing import List
from gym import spaces
import numpy as np
from citylearn.agents.base import Agent
from citylearn.preprocessing import Encoder

class RLC(Agent):
    def __init__(self, *args, encoders: List[Encoder], observation_space: spaces.Box, **kwargs):
        r"""Initialize `RLC`.

        Base reinforcement learning controller class.

        Parameters
        ----------
        *args : tuple
            `Agent` positional arguments.
        encoders : List[Encoder]
            Observation value transformers/encoders.
        observation_space : spaces.Box
            Format of valid observations.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize `Agent` super class.
        """

        super().__init__(*args, **kwargs)
        self.encoders = encoders
        self.observation_space = observation_space

    @property
    def encoders(self) -> List[Encoder]:
        """Observation value transformers/encoders."""

        return self.__encoders

    @property
    def observation_dimension(self) -> int:
        """Number of observations after applying `encoders`."""

        return len([j for j in np.hstack(self.encoders*np.ones(len(self.observation_space.low))) if j != None])

    @property
    def observation_space(self) -> spaces.Box:
        """Format of valid observations."""
        
        return self.__observation_space

    @encoders.setter
    def encoders(self, encoders: List[Encoder]):
        self.__encoders = encoders

    @observation_space.setter
    def observation_space(self, observation_space: spaces.Box):
        self.__observation_space = observation_space