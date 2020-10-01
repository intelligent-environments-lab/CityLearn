"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified at will.
This reward_function takes all the electrical demands of all the buildings and turns them into one or multiple rewards for the agent(s)

The current code of reward_functioin_ma computes a reward multiplying net electricity consumption of the whole district and of each individual building. Therefore, each agent (building) is incentivized to not only
minimize its own electricity consumption but also the electricity consumption of the whole district of buildings. The reward function is non-linear, and the penalty it returns increases polinomially with the net electriicty consumption.
This incentivizes not just reducing the net electricity consumption, but also flattening the curve of net electrical demand, as higher values for demand contribute a lot more to the penalty than lower values for demand.
"""
import numpy as np

# Reward function for the multi-agent (decentralized) agents
class reward_function_ma:
    def __init__(self, n_agents, building_info):
        self.n_agents = n_agents
        self.building_info = building_info

    # Electricity_demand contains negative values when the building consumes more electricity than it generates
    def get_rewards(self, electricity_demand):  
        electricity_demand = np.float32(electricity_demand)
        total_electricity_demand = 0
        for e in electricity_demand:
            total_electricity_demand += -e
            
        electricity_demand = np.array(electricity_demand)
        
        return list(np.sign(electricity_demand)*0.01*(np.array(np.abs(electricity_demand))**2 * max(0, total_electricity_demand)))
        
        # Single-agent reward
        # reward_ = np.array(electricity_demand)**3.0
        # reward_[reward_>0] = 0
        # return list(reward_)
    
      
# Reward function for the single-agent (centralized) agent
def reward_function_sa(electricity_demand):

    reward_ = -np.array(electricity_demand).sum()
    reward_ = max(0, reward_)
    reward_ = reward_**3.0
    
    return reward_