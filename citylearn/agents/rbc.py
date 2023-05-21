from typing import Any, Mapping, List
from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from citylearn.building import Building

class RBC(Agent):
    r"""Base rule based controller class.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    
    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)

class HourRBC(RBC):
    r"""A time-of-use rule-based controller.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    action_map: Mapping[int, float], optional
        A 24-hour action map where the key is the hour between 1-24 and the value is the action.
        For storage systems, the value is negative for discharge and positive for charge. Will
        return random actions if no map is provided.
    
    Other Parameters
    ----------------
    **kwargs: Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, env: CityLearnEnv, action_map: Mapping[int, float] = None, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.action_map = action_map

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for current time step.

        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[List[float]]
            Action values
        """

        actions = []

        if self.action_map is None:
            actions = super().predict(observations, deterministic=deterministic)
        
        else:
            for n, o, d in zip(self.observation_names, observations, self.action_dimension):
                hour = o[n.index('hour')]
                a = [self.action_map[hour] for _ in range(d)]
                actions.append(a)

            self.actions = actions
            self.next_time_step()
        
        return actions

class BasicRBC(HourRBC):
    r"""A time-of-use rule-based controller for heat-pump charged thermal energy storage systems that charges when COP is high.

    The actions are designed such that the agent charges the controlled storage system(s) by 9.1% of its maximum capacity every
    hour between 10:00 PM and 08:00 AM, and discharges 8.0% of its maximum capacity at every other hour.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    hour_index: int, default: 2
        Expected position of hour observation when `observations` paramater is parsed into `select_actions` method.
    
    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        action_map = {}

        for hour in Building.get_periodic_observation_metadata()['hour']:
            if 9 <= hour <= 21:
                value = -0.08
            elif (1 <= hour <= 8) or (22 <= hour <= 24):
                value = 0.091
            else:
                value = 0.0

            action_map[hour] = value

        self.action_map = action_map

class OptimizedRBC(BasicRBC):
    r"""A time-of-use rule-based controller that is an optimized version of :py:class:`citylearn.agents.rbc.BasicRBC`
    where control actions have been selected through a search grid.

    The actions are designed such that the agent discharges the controlled storage system(s) by 2.0% of its 
    maximum capacity every hour between 07:00 AM and 03:00 PM, discharges by 4.4% of its maximum capacity 
    between 04:00 PM and 06:00 PM, discharges by 2.4% of its maximum capacity between 07:00 PM and 10:00 PM, 
    charges by 3.4% of its maximum capacity between 11:00 PM to midnight and charges by 5.532% of its maximum 
    capacity at every other hour.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    hour_index: int, default: 2
        Expected position of hour observation when `observations` paramater is parsed into `select_actions` method.
    
    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        action_map = {}

        for hour in Building.get_periodic_observation_metadata()['hour']:
            if 7 <= hour <= 15:
                value = -0.02
            elif 16 <= hour <= 18:
                value = -0.0044
            elif 19 <= hour <= 22:
                value = -0.024
            elif 23 <= hour <= 24:
                value = 0.034
            elif 1 <= hour <= 6:
                value = 0.05532
            else:
                value = 0.0

            action_map[hour] = value

        self.action_map = action_map

class BasicBatteryRBC(BasicRBC):
    r"""A time-of-use rule-based controller that is designed to take advantage of solar generation for charging.

    The actions are optimized for electrical storage (battery) such that the agent charges the controlled
    storage system(s) by 11.0% of its maximum capacity every hour between 06:00 AM and 02:00 PM, 
    and discharges 6.7% of its maximum capacity at every other hour.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    hour_index: int, default: 2
        Expected position of hour observation when `observations` paramater is parsed into `select_actions` method.
    
    Other Parameters
    ----------------
    **kwargs: Any
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        action_map = {}

        for hour in Building.get_periodic_observation_metadata()['hour']:
            if 6 <= hour <= 14:
                value = 0.11
            else:
                value = -0.067

            action_map[hour] = value
        
        self.action_map = action_map