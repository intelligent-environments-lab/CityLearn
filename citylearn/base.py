import uuid

class Environment:
    def __init__(self, seconds_per_timestep: float = 3600.0):
        self.__uid = uuid.uuid4().hex
        self.__time_step = None
        self.__seconds_per_time_step = seconds_per_timestep
        self.reset()

    @property
    def uid(self) -> str:
        return self.__uid

    @property
    def time_step(self) -> int:
        return self.__time_step

    @property
    def seconds_per_time_step(self) -> float:
        return self.__seconds_per_time_step

    @seconds_per_time_step.setter
    def seconds_per_time_step(self, seconds_per_time_step: float):
        self.__seconds_per_time_step = seconds_per_time_step

    def next_time_step(self):
        self.__time_step += 1

    def reset(self):
        self.reset_time_step()

    def reset_time_step(self):
        self.__time_step = 0