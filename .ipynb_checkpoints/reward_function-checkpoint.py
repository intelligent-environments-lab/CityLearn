"""
This function is intended to wrap the rewards returned by the CityLearn RL environment.
CityLearn returns the energy consumption of each building as a reward. This reward function
takes all the electrical demands of all the buildings, computer an electricity price which is
proportional to the total demand for electricity in the neighborhood, and multiplies every
reward by that price. Then it returns the new rewards, which should be used by the agent.
"""
def reward_function(rewards):
    total_energy_demand = 0
    for r in rewards:
        total_energy_demand += -r
        
    price = total_energy_demand*3e-5 # + 0.045
    
    for i in range(len(rewards)):
        rewards[i] = price*rewards[i]
    
    return rewards

