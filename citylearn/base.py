import random
from typing import Any, Mapping
import uuid

class Environment:
    """Base class for all `citylearn` classes that have a spatio-temporal dimension.

    Parameters
    ----------
    seconds_per_time_step: float, default: 3600.0
        Number of seconds in 1 `time_step` and must be set to >= 1.
    random_seed : int, optional
        Pseudorandom number generator seed for repeatable results.
    """
    
    def __init__(self, seconds_per_time_step: float = None, random_seed: int = None):
        self.seconds_per_time_step = seconds_per_time_step
        self.__uid = uuid.uuid4().hex
        self.random_seed = random_seed
        self.__time_step = None
        self.reset()

    @property
    def uid(self) -> str:
        r"""Unique environment ID."""

        return self.__uid
    
    @property
    def random_seed(self) -> int:
        """Pseudorandom number generator seed for repeatable results."""

        return self.__random_seed

    @property
    def time_step(self) -> int:
        r"""Current environment time step."""

        return self.__time_step

    @property
    def seconds_per_time_step(self) -> float:
        r"""Number of seconds in 1 time step."""

        return self.__seconds_per_time_step
    
    @random_seed.setter
    def random_seed(self, random_seed: int):
        random_seed = random.randint(0, 100_000_000) if random_seed is None else random_seed
        self.__random_seed = random_seed

    @seconds_per_time_step.setter
    def seconds_per_time_step(self, seconds_per_time_step: float):
        if seconds_per_time_step is None:
            self.seconds_per_time_step = 3600.0
        else:
            assert seconds_per_time_step >= 1, 'seconds_per_time_step >= 1'
            self.__seconds_per_time_step = seconds_per_time_step

    def get_metadata(self) -> Mapping[str, Any]:
        """Returns general static information."""

        return {
            'uid': self.uid,
            'random_seed': self.random_seed,
            'seconds_per_time_step': self.seconds_per_time_step
        }

    def next_time_step(self):
        r"""Advance to next `time_step` value.

        Notes
        -----
        Override in subclass for custom implementation when advancing to next `time_step`.
        """

        self.__time_step += 1

    def reset(self):
        r"""Reset environment to initial state.

        Calls `reset_time_step`.

        Notes
        -----
        Override in subclass for custom implementation when reseting environment.
        """

        self.reset_time_step()

    def reset_time_step(self):
        r"""Reset `time_step` to initial state.

        Sets `time_step` to 0.
        """

        self.__time_step = 0