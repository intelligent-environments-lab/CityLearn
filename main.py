#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt

from agent import RL_Agents, RBC_Agent
import numpy as np
from csv import DictWriter
import json
from pathlib import Path
import time
import warnings

from citylearn import  CityLearn

# Extra packages for postprocessing
import matplotlib.dates as dates
import pandas as pd

from algo_utils import graph_total, graph_building, tabulate_table

# Ignore the casting to float32 warnings
warnings.simplefilter("ignore", UserWarning)
    
# In[2]:

# Load environment
climate_zone = 4
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ["Building_3"]
# building_ids = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
objective_function = ['1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = True)

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()


# In[ ]:

# Set seeds (TO DO: CHECK PERFORMANCE SAME FOR TWO RUNS WITH SAME SEED)
seed = 101
np.random.seed(seed)
env.seed(seed)

# Initialise agent
agents = RL_Agents(building_info, observations_spaces, actions_spaces, env, seed)

# Play a saved ckpt of actor network in the environment

# Select many episodes for training. In the final run we will set this value to 1 (the buildings run for one year)
episodes = 1

k, c = 0, 0
cost, cum_reward = {}, {}

# Measure the time taken for training
start_timer = time.time()

for e in range(episodes):
    cum_reward[e] = 0
    rewards = []
    state = env.reset()
    done = False
            
    while not done:
        
        # Add batch dimension to single state input, and remove batch dimension from single action output
        action = agents.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agents.add_to_buffer(state, action, reward, next_state, done, env.net_electric_consumption[-1])
        state = next_state
        cum_reward[e] += agents.reward_tracker[-1]
        rewards.append(agents.reward_tracker[-1])
        k+=1
            
    cost[e] = env.cost()
            
    if c%1==0:
        print(f"Episode {e} |", "Cost:", cost[e]['total'], "Reward:", cum_reward[e], '\n')
    c+=1
                    
env.close() 

timer = time.time() - start_timer

# In[ ]:
## POSTPROCESSING

# Plotting winter operation
interval = range(5000,5200)
plt.figure(figsize=(16,5))
plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
plt.plot(env.net_electric_consumption_no_storage[interval])
plt.plot(env.net_electric_consumption[interval], '--')
plt.xlabel('time (hours)')
plt.ylabel('kW')
plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using SAC for storage(kW)'])
plt.savefig("test.jpg", bbox_inches='tight', dpi = 300)

tabulate_table(env=env, timer=timer, algo="SAC", agent = agents, climate_zone=climate_zone, building_ids=building_ids, 
               building_attributes=building_attributes, parent_dir=agents.load_path, num_episodes=episodes, episode_scores=rewards)

