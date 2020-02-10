"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified by the participants of The CityLearn Challenge.
CityLearn returns the energy consumption of each building as a reward. 
This reward_function takes all the electrical demands of all the buildings and turns them into one or multiple rewards for the agent(s)

The current code computes a virtual (made-up) electricity price proportional to the total demand for electricity in the neighborhood, and multiplies every
reward by that price. Then it returns the new rewards, which should be used by the agent. Participants of the CityLearn Challenge are encouraged to completely modify this function
in order to minimize the 5 proposed metrics.
"""

# Reward function for the multi-agent (decentralized) agents
def reward_function_ma(electricity_demand):
    total_energy_demand = 0
    for e in electricity_demand:
        total_energy_demand += -e
        
    price = max(total_energy_demand*0.01, 0)
    
    for i in range(len(electricity_demand)):
        electricity_demand[i] = min(price*electricity_demand[i], 0)
    
    return electricity_demand

# Reward function for the single-agent (centralized) agent
def reward_function_sa(electricity_demand):
    total_energy_demand = 0
    for e in electricity_demand:
        total_energy_demand += -e
        
    price = max(total_energy_demand*0.01, 0)
    
    for i in range(len(electricity_demand)):
        electricity_demand[i] = min(price*electricity_demand[i], 0)
    
    return sum(electricity_demand)

