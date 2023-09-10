from typing import List
import numpy as np
from citylearn.base import Environment
from citylearn.data import Weather

class PowerOutage:
    def __init__(self, random_seed: int = None, seconds_per_time_step: float = None):
        self.random_seed = random_seed
        self.seconds_per_time_step = seconds_per_time_step

    @property
    def random_seed(self) -> int:
        return self.__random_seed
    
    @property
    def seconds_per_time_step(self) -> float:
        return self.__seconds_per_time_step
    
    @random_seed.setter
    def random_seed(self, value: int):
        self.__random_seed = np.random.randint(*Environment.DEFAULT_RANDOM_SEED_RANGE) if value is None else value

    @seconds_per_time_step.setter
    def seconds_per_time_step(self, value: float):
        self.__seconds_per_time_step = Environment.DEFAULT_SECONDS_PER_TIME_STEP if value is None else value

    def get_signals(self, time_steps: int, weather: Weather = None) -> np.ndarray:
        np.random.seed(self.random_seed)
        signals = np.random.choice([0, 1], size=time_steps) 

        return signals  

class ReliabilityMetricsPowerOutage(PowerOutage):
    def __init__(self, *args, saifi: float = None, saidi: float = None, caidi: float = None, start_time_steps: List[int] = None):
        super().__init__(*args)
        self.saifi = saifi
        self.saidi = saidi
        self.caidi = caidi
        self.start_time_steps = start_time_steps

    @property
    def saifi(self) -> float:
        return self.__saifi
    
    @property
    def saidi(self) -> float:
        return self.__saidi
    
    @property
    def caidi(self) -> float:
        return self.__caidi
    
    @property
    def start_time_steps(self) -> List[int]:
        return self.__start_time_steps
    
    @saifi.setter
    def saifi(self, value: float):
        self.__saifi = 1.436 if value is None else value

    @saidi.setter
    def saidi(self, value: float):
        self.__saidi = 475.8 if value is None else value

    @caidi.setter
    def caidi(self, value: float):
        self.__caidi = 331.2 if value is None else value

    @start_time_steps.setter
    def start_time_steps(self, value: List[float]):
        self.__start_time_steps = value

    def get_signals(self, time_steps: int, weather: Weather = None) -> np.ndarray:
        np.random.seed(self.random_seed)
        days_per_year = 365.0
        seconds_per_day = 86400.0
        seconds_per_minute = 60.0
        time_steps_per_day = seconds_per_day/self.seconds_per_time_step
        time_steps_per_minute = seconds_per_minute/self.seconds_per_time_step
        day_count = time_steps/time_steps_per_day
        daily_outage_probability = self.saifi/days_per_year
        outage_days = np.random.binomial(n=1, p=daily_outage_probability, size=day_count)
        outage_day_ixs = outage_days*np.arange(day_count)
        outage_day_ixs = outage_day_ixs[outage_day_ixs != 0]
        outage_day_count = outage_days[outage_days == 1].shape[0]
        start_time_steps = list(range(time_steps_per_day)) if self.start_time_steps is None else self.start_time_steps
        outage_start_time_steps = np.random.choice(start_time_steps, size=outage_day_count)
        outage_durations = np.random.exponential(scale=self.caidi, size=outage_day_count) # [mins]
        outage_duration_time_steps = outage_durations*time_steps_per_minute
        signals = np.zeros(time_steps, dtype=int)

        for i, j , k in zip(outage_day_ixs, outage_start_time_steps, outage_duration_time_steps):
            start_ix = i*time_steps_per_day + j
            end_ix = start_ix + k
            start_ix = int(start_ix)
            end_ix = int(end_ix)
            signals[start_ix:end_ix] = 1
        
        return signals