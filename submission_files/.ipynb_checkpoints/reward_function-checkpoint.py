'''Import any packages here'''

class reward_function_ma:
    def __init__(self, n_agents, building_info):
        
        '''Initialize the class'''
        

    # Electricity_demand contains negative values when the building consumes more electricity than it generates
    def get_rewards(self, electricity_demand, carbon_intensity):
        
        '''Write your reward function here'''
        
        return rewards
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      
# Do not use or delete, it isn't used in the CityLearn Challenge
# Reward function for the centralized agent. To be used only if all the buildings receive the same reward.
def reward_function_sa(electricity_demand):

    reward_ = -np.array(electricity_demand).sum()
    reward_ = max(0, reward_)
    reward_ = reward_**3.0
    
    return reward_