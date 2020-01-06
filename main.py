from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Load environment
data_folder = Path("data/")
building_attributes = data_folder / 'building_attributes.json'
solar_profile = data_folder / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]

env = CityLearn(building_attributes, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = ['ramping','1-load_factor','peak_to_valley_ratio','peak_demand','net_electricity_consumption','quadratic'])
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

# RL CONTROLLER
#Instantiating the control agent(s)
agents = RL_Agents(building_info, observations_spaces,actions_spaces)
episodes = 300

k, c = 0, 0
cost, cum_reward = {}, {}

# The number of episodes can be replaces by a stopping criterion (i.e. convergence of the average reward)
for e in range(episodes):     
    cum_reward[e] = 0
    rewards = []
    state = env.reset()
    done = False
    while not done:
        if k%(40000*4)==0:
            print('hour: '+str(k)+' of '+str(8760*episodes))
            
        action = agents.select_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward_function(reward) #See comments in reward_function.py
        agents.add_to_buffer(state, action, reward, next_state, done)
        state = next_state
        
        cum_reward[e] += reward[0]
        rewards.append(reward)
        k+=1
        
    cost[e] = env.cost()
    if c%20==0:
        print(cost[e])
    c+=1
