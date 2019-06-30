import numpy as np

#RULE BASED CONTROLLER
class AgentRBC:
    def __init__(self):
        self.hour = 3500
    
    def select_action(self, states):
        self.hour += 1
        hour_day = states[0]
        #DAYTIME
        action1 = 0.0
        action2 = 0.0
        if hour_day >= 15 and hour_day <= 21:
            #WINTER (STORE HEAT)
            if self.hour > 7000 or self.hour < 2800:
                action1 = 1.0
            #SUMMER (RELEASE COOLING)
            if self.hour >= 2800 and self.hour <= 7000:
                action2 = -1.0
        #NIGHTTIME       
        elif hour_day >= 4 and hour_day <= 10:
            #WINTER (RELEASE HEAT)
            if self.hour > 7000 or self.hour < 2800:
                action1 = -1.0
            #SUMMER (STORE COOLING)
            if self.hour >= 2800 and self.hour <= 7000:
                action2 = 1.0
            
        return np.array([action1,action2])