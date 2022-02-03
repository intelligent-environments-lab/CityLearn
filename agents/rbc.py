import numpy as np

class RBC:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []

    def select_action(self):
        raise NotImplementedError

class BasicRBC(RBC):
    def __init__(self, actions_spaces):
        super().__init__(actions_spaces)

    def select_action(self,hour):
        # Daytime: release stored energy
        if hour >= 9 and hour <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        # Early nightime: store DHW and/or cooling energy
        elif (hour >= 1 and hour <= 8) or (hour >= 22 and hour <= 24):
            a = [[0.091 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        else:
            a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
            
        self.action_tracker.append(a)
        return np.array(a, dtype = 'object')

class OptimizedRBC(RBC):
    def __init__(self, actions_spaces):
        super().__init__(actions_spaces)
        
    def select_action(self,hour,multiplier=0.4):
        # Daytime: release stored energy  2*0.08 + 0.1*7 + 0.09
        if hour >= 7 and hour <= 11:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour >= 12 and hour <= 15:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour >= 16 and hour <= 18:
            a = [[-0.11 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour >= 19 and hour <= 22:
            a = [[-0.06 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        # Early nightime: store DHW and/or cooling energy
        elif hour >= 23 and hour <= 24:
            a = [[0.085 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour >= 1 and hour <= 6:
            a = [[0.1383 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        else:
            a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]

        self.action_tracker.append(a)
        return np.array(a, dtype = 'object')
