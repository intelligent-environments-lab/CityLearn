# Run this again after editing submodules so Colab uses the updated versions
from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
from agent import RL_Agents_Coord
import numpy as np                                                                                                                                                                                      
import csv
import time
import re
import pandas as pd
import torch
from joblib import dump, load

# Load environment
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_id = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','quadratic']

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_id, buildings_states_actions = building_state_actions, cost_function = objective_function, verbose = 0, simulation_period=(0,8760-1))
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

# Hyperparameters
bs = 256
tau = 0.005
gamma = 0.99
lr = 0.0003
hid = [256,256]

n_episodes = 12

# Instantiating the control agent(s)
agents = RL_Agents_Coord(building_id, building_state_actions, building_info, observations_spaces, actions_spaces, discount = gamma, batch_size = bs, replay_buffer_capacity = 1e5, regression_buffer_capacity = 12*8760, tau=tau, lr=lr, hidden_dim=hid, start_training=8760*3, exploration_period = 8760*3+1,  start_regression=8760, information_sharing = True, pca_compression = .95, action_scaling_coef=0.5, reward_scaling = 5., update_per_step = 1, iterations_as = 2)

# The number of episodes can be replaces by a stopping criterion (i.e. convergence of the average reward)
start = time.time()
for e in range(n_episodes): 
    is_evaluating = (e > 7) # Evaluate deterministic policy after 7 epochs
    rewards = []
    state = env.reset()
    done = False

    j = 0
    action, coordination_vars = agents.select_action(state, deterministic=is_evaluating)    
    while not done:
        next_state, reward, done, _ = env.step(action)
        action_next, coordination_vars_next = agents.select_action(next_state, deterministic=is_evaluating)
        agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)

        state = next_state
        coordination_vars = coordination_vars_next
        action = action_next

    print('Loss -',env.cost(), 'Simulation time (min) -',(time.time()-start)/60.0)