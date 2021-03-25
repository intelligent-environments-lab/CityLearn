import numpy as np

class RBC:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day = states[0][0]
        multiplier = 0.4
        # Daytime: release stored energy  2*0.08 + 0.1*7 + 0.09
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 7 and hour_day <= 11:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 12 and hour_day <= 15:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 16 and hour_day <= 18:
            a = [[-0.11 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 19 and hour_day <= 22:
            a = [[-0.06 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        # Early nightime: store DHW and/or cooling energy
        if hour_day >= 23 and hour_day <= 24:
            a = [[0.085 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 1 and hour_day <= 6:
            a = [[0.1383 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]

        self.action_tracker.append(a)
        return np.array(a, dtype = 'object')
