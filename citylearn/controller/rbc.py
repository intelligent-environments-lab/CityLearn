from typing import List
from citylearn.controller.base import Controller

class RBC(Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class BasicRBC(RBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, hour: int) -> List[float]:
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

    def select_actions(self, hour: int) -> List[float]:
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