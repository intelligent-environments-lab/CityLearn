"""
This function is intended to wrap the rewards returned by the CityLearn RL environment, and is meant to 
be modified by the participants of The CityLearn Challenge.
CityLearn returns the energy consumption of each building as a reward. 
This reward_function takes all the electrical demands of all the buildings and turns them into one or multiple rewards for the agent(s)

The current code computes a virtual (made-up) electricity price proportional to the total demand for electricity in the neighborhood, and multiplies every
reward by that price. Then it returns the new rewards, which should be used by the agent. Participants of the CityLearn Challenge are encouraged to completely modify this function
in order to minimize the 5 proposed metrics.
"""

def reward_function(rewards):
    total_energy_demand = 0
    for r in rewards:
        total_energy_demand += -r
        
    price = total_energy_demand*0.01
    
    for i in range(len(rewards)):
        rewards[i] = price*rewards[i]
    
    return rewards

