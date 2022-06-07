from typing import List
from citylearn.agents.base import Agent

class RBC(Agent):
    def __init__(self, *args, **kwargs):
        r"""Initialize `RBC`.

        Base rule based controller class.

        Parameters
        ----------
        *args : tuple
            `Agent` positional arguments.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)

class BasicRBC(RBC):
    def __init__(self, *args, hour_index: int = None, **kwargs):
        r"""Initialize `RBC`.

        Base rule based controller class.

        Parameters
        ----------
        *args : tuple
            `Agent` positional arguments.
        hour_index: int, default: 2
            Expected position of hour observation when `observations` paramater is parsed into `select_actions` method.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)
        self.hour_index = hour_index

    @property
    def hour_index(self) -> int:
        """Expected position of hour observation when `observations` paramater is parsed into `select_actions` method."""

        return self.__hour_index

    @hour_index.setter
    def hour_index(self, hour_index: int):
        self.__hour_index = 2 if hour_index is None else hour_index

    def select_actions(self, observations: List[float]) -> List[float]:
        """Provide actions for current time step.

        Notes
        -----
        The actions are designed such that the agent charges the controlled storage system(s) by 9.1% of its maximum capacity every hour between 10:00 PM and 08:00 AM, and discharges 8.0% of its maximum capacity at every other hour.
        
        Returns
        -------
        actions: List[float]
            Action values
        """

        hour = observations[self.hour_index]

        # Daytime: release stored energy
        if hour >= 9 and hour <= 21:
            actions = [-0.08 for _ in range(self.action_dimension)]
        
        # Early nightime: store DHW and/or cooling energy
        elif (hour >= 1 and hour <= 8) or (hour >= 22 and hour <= 24):
            actions = [0.091 for _ in range(self.action_dimension)]
        
        else:
            actions = [0.0 for _ in range(self.action_dimension)]

        self.actions = actions
        self.next_time_step()
        return actions

class OptimizedRBC(BasicRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, observations: List[float]) -> List[float]:
        """Provide actions for current time step.

        Notes
        -----
        The actions are designed such that the agent discharges the controlled storage system(s) by 2.0% of its maximum capacity every hour between 07:00 AM and 03:00 PM, discharges by 4.4% of its maximum capacity between 04:00 PM and 06:00 PM, discharges by 2.4% of its maximum capacity between 07:00 PM and 10:00 PM, charges by 3.4% of its maximum capacity between 11:00 PM to midnight and charges by 5.532% of its maximum capacity at every other hour.
        
        Returns
        -------
        actions: List[float]
            Action values
        """

        hour = observations[self.hour_index]
        
        # Daytime: release stored energy  2*0.08 + 0.1*7 + 0.09
        if hour >= 7 and hour <= 15:
            actions = [-0.02 for _ in range(self.action_dimension)]
            
        elif hour >= 16 and hour <= 18:
            actions = [-0.0044 for _ in range(self.action_dimension)]
            
        elif hour >= 19 and hour <= 22:
            actions = [-0.024 for _ in range(self.action_dimension)]
            
        # Early nightime: store DHW and/or cooling energy
        elif hour >= 23 and hour <= 24:
            actions = [0.034 for _ in range(self.action_dimension)]
            
        elif hour >= 1 and hour <= 6:
            actions = [0.05532 for _ in range(self.action_dimension)]
            
        else:
            actions = [0.0 for _ in range(self.action_dimension)]

        self.actions = actions
        self.next_time_step()
        return actions

class BasicBatteryRBC(BasicRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, observations: List[float]) -> List[float]:
        """Provide actions for current time step.

        Notes
        -----
        The actions are optimized for electrical storage (battery) such that the agent charges the controlled storage system(s) by 11.0% of its maximum capacity every hour between 06:00 AM and 02:00 PM, and discharges 6.7% of its maximum capacity at every other hour.
        
        Returns
        -------
        actions: List[float]
            Action values
        """

        hour = observations[self.hour_index]

        # Late morning and early evening: store energy
        if hour >= 6 and hour <= 14:
            actions = [0.11 for _ in range(self.action_dimension)]
        
        # Early morning and late evening: release energy
        else:
            actions = [-0.067 for _ in range(self.action_dimension)]

        self.actions = actions
        self.next_time_step()
        return actions