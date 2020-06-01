#!/usr/bin/env python
# coding: utf-8

"""
Implementation of Deep Deterministic Policy Gradients (DDPG) network
using PyTorch.
See https://arxiv.org/abs/1509.02971 for algorithm details.

@author: Anjukan Kathirgamanathan 2020 (k.anjukan@gmail.com) 

Project for CityLearn Competition
"""

# In[1]:
# Import Packages
from agent_DDPG import Agent
import numpy as np
from pathlib import Path

import torch

from citylearn import  CityLearn

# Extra packages for postprocessing
import matplotlib.dates as dates
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import os
from csv import DictWriter


# In[2]:

# Load environment
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
#building_ids = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
building_ids = ["Building_1","Building_2"]
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = False, verbose = 1)

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()


# In[ ]:

"""
###################################
STEP 1: Set the Training Parameters
======
        num_episodes (int): maximum number of training episodes
        episode_scores (float): list to record the scores obtained from each episode
        scores_average_window (int): the window size employed for calculating the average score (e.g. 100)
"""
num_episodes=10
episode_scores = []
scores_average_window = 5

"""
#############################################
STEP 2: Determine the size of the Action and State Spaces and the Number of Agents

The observation space consists of various variables corresponding to the 
building_state_action_json file. See https://github.com/intelligent-environments-lab/CityLearn
for more information about the states. Each agent receives all observations of all buildings
(communication between buildings). 
Up to two continuous actions are available, corresponding to whether to charge or discharge
the cooling storage and DHW storage tanks.

"""

# Get number of agents in Environment
num_agents = env.n_buildings
print('\nNumber of Agents: ', num_agents)

# Set the size of state observations or state size
print('\nSize of State: ', observations_spaces)

"""
###################################
STEP 3: Create DDPG Agents from the Agent Class in ddpg_agent.py
A DDPG agent initialized with the following parameters.
======
building_info: Dictionary with building information as described above
state_size (list): List of lists with observation spaces of all buildings selected
action_size (list): List of lists with action spaces of all buildings selected
seed (int): random seed for initializing training point (default = 0)

The invididual agents are defined within the Agent call for each building
"""

agent = Agent(building_info, state_size=observations_spaces, action_size=actions_spaces, random_seed=0)

"""
###################################
STEP 4: Run the DDPG Training Sequence
The DDPG Training Process involves the agent learning from repeated episodes of behaviour 
to map states to actions the maximize rewards received via environmental interaction.

The agent training process involves the following:
(1) Reset the environment at the beginning of each episode.
(2) Obtain (observe) current state, s, of the environment at time t
(3) Perform an action, a(t), in the environment given s(t)
(4) Observe the result of the action in terms of the reward received and 
	the state of the environment at time t+1 (i.e., s(t+1))
(5) Update agent memory and learn from experience (i.e, agent.step)
(6) Update episode score (total reward received) and set s(t) -> s(t+1).
(7) If episode is done, break and repeat from (1), otherwise repeat from (3).
"""

# Measure the time taken for training
start_timer = time.time()

# Store the weights and scores in a new directory
parent_dir = "alg/ddpg{}/".format(time.strftime("%Y%m%d-%H%M%S")) # apprends the timedate
os.makedirs(parent_dir, exist_ok=True)

# loop from num_episodes
for i_episode in range(1, num_episodes+1):

	# reset the environment at the beginning of each episode
	states = env.reset()
	
	# set the initial episode score to zero.
	agent_scores = np.zeros(num_agents)

	# Run the episode training loop;
	# At each loop step take an action as a function of the current state observations
	# Based on the resultant environmental state (next_state) and reward received update the Agents Actor and Critic networks
	# If environment episode is done, exit loop.
	# Otherwise repeat until done == true 
	while True:
		# determine actions for the agents from current sate, using noise for exploration
		action = agent.select_action(states, add_noise=True)
		#print(actions)
		# send the actions to the agents in the environment and receive resultant environment information
		next_states, reward, done, _ = env.step(action)
		
		#Send (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
		agent.step(states, action, reward, next_states, done)
		
		# set new states to current states for determining next actions
		states = next_states
		# Update episode score for each agent
		agent_scores += reward
		
		# If any agent indicates that the episode is done, 
		# then exit episode loop, to begin new episode
		if np.any(done):
			break
	
	timer = time.time() - start_timer
	
	# Add episode score to Scores and
	# Calculate mean score over averaging window 
	episode_scores.append(np.sum(agent_scores))
	average_score = np.mean(episode_scores[i_episode-min(i_episode,scores_average_window):i_episode+1])
	
	#Print current and average score
	print('\nEpisode {}\tCumulated Reward: {:.2f}\tAverage Reward: {:.2f}\n'.format(i_episode, episode_scores[i_episode-1], average_score), end="")
	
	# Check to see if the training is completed
	# If yes, save the network weights and scores and end training.
	if i_episode > num_episodes - 1:
		print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\n'.format(i_episode, average_score))
		
		# Save trained  Actor and Critic network weights for each agent
		for building in range(0,num_agents):
			an_filename = "ddpgActor{0}_Model.pth".format(building)
			torch.save(agent.actor_local[building].state_dict(), parent_dir + an_filename)
			cn_filename = "ddpgCritic{0}_Model.pth".format(building)
			torch.save(agent.critic_local[building].state_dict(), parent_dir + cn_filename)
		
		# Save the recorded Scores data
		scores_filename = "ddpgAgent_Scores_new.csv"
		np.savetxt(parent_dir + scores_filename, episode_scores, delimiter=",")
		break

"""
###################################
STEP 5: Everything is Finished -> Close the Environment.
"""

env.close()

"""
###################################
STEP 6: POSTPROCESSING
"""

# Building to plot results for
building_number = 'Building_1'

# Convert output to dataframes for easy plotting
time_periods = pd.date_range('2017-01-01 T01:00', '2017-12-31 T23:00', freq='1H')
output = pd.DataFrame(index = time_periods)

# Extract building behaviour
output['Electricity demand without storage or generation (kW)'] = env.net_electric_consumption_no_pv_no_storage[-8759:]
output['Electricity demand with PV generation and without storage(kW)'] = env.net_electric_consumption_no_storage[-8759:]
output['Electricity demand with PV generation and using DDPG for storage(kW)'] = env.net_electric_consumption[-8759:]
# Cooling Storage
output['Cooling Demand (kWh)'] = env.buildings[building_number].cooling_demand_building[-8759:]
output['Energy Storage State of Charge - SOC (kWh)'] = env.buildings[building_number].cooling_storage_soc[-8759:]
output['Heat Pump Total Cooling Supply (kW)'] = env.buildings[building_number].cooling_device_to_building[-8759:] + env.buildings[building_number].cooling_device_to_storage[-8759:]
output['Cooling Controller Action - Increase or Decrease of SOC (kW)'] = [k[0][0]*env.buildings[building_number].cooling_storage.capacity for k in [j for j in np.array(agent.action_tracker[-8759:])]]
# DHW
output['DHW Demand (kWh)'] = env.buildings[building_number].dhw_demand_building[-8759:]
#output['Energy Balance of DHW Tank (kWh)'] = -env.buildings[building_number].dhw_storage.energy_balance[-8759:]
output['Energy Balance of DHW Tank (kWh)'] = env.buildings[building_number].dhw_storage_soc[-8759:]
output['DHW Heater Total Heating Supply (kWh)'] = env.buildings[building_number].dhw_heating_device.heat_supply[-8759:]
output['DHW Controller Action - Increase or Decrease of SOC (kW)'] = [k[0][1]*env.buildings[building_number].dhw_storage.capacity for k in [j for j in np.array(agent.action_tracker[-8759:])]]
output['DHW Heater Electricity Consumption (kWh)'] = env.buildings[building_number].electric_consumption_dhw[-8759:]

output_filtered = output.loc['2017-12-30':'2017-12-31']

# Create plot showing electricity demand profile with RL agent, cooling storage behaviour and DHW storage behaviour
fig, ax = plt.subplots(nrows = 3, figsize=(20,12), sharex = True)
output_filtered['Electricity demand without storage or generation (kW)'].plot(ax = ax[0], color='blue', label='Electricity demand without storage or generation (kW)', x_compat=True)
output_filtered['Electricity demand with PV generation and without storage(kW)'].plot(ax = ax[0], color='orange', label='Electricity demand with PV generation and without storage(kW)')
output_filtered['Electricity demand with PV generation and using DDPG for storage(kW)'].plot(ax = ax[0], color = 'green', ls = '--', label='Electricity demand with PV generation and using DDPG for storage(kW)')
ax[0].set_title('(a) - Electricity Demand')
ax[0].set(ylabel="Demand [kW]")
ax[0].legend(loc="upper right")
ax[0].xaxis.set_major_locator(dates.DayLocator())
ax[0].xaxis.set_major_formatter(dates.DateFormatter('\n%d/%m'))
ax[0].xaxis.set_minor_locator(dates.HourLocator(interval=6))
ax[0].xaxis.set_minor_formatter(dates.DateFormatter('%H'))
output_filtered['Cooling Demand (kWh)'].plot(ax = ax[1], color='blue', label='Cooling Demand (kWh)', x_compat=True)
output_filtered['Energy Storage State of Charge - SOC (kWh)'].plot(ax = ax[1], color='orange', label='Energy Storage State of Charge - SOC (kWh)')
output_filtered['Heat Pump Total Cooling Supply (kW)'].plot(ax = ax[1], color = 'green', label='Heat Pump Total Cooling Supply (kW)')
output_filtered['Cooling Controller Action - Increase or Decrease of SOC (kW)'].plot(ax = ax[1], color = 'red', label='Controller Action - Increase or Decrease of SOC (kW)')
ax[1].set_title('(b) - Cooling Storage Utilisation')
ax[1].set(ylabel="Power [kW]")
ax[1].legend(loc="upper right")
output_filtered['DHW Demand (kWh)'].plot(ax = ax[2], color='blue', label='DHW Demand (kWh)', x_compat=True)
output_filtered['Energy Balance of DHW Tank (kWh)'].plot(ax = ax[2], color='orange', label='Energy Balance of DHW Tank (kWh)')
output_filtered['DHW Heater Total Heating Supply (kWh)'].plot(ax = ax[2], color = 'green', label='DHW Heater Total Heating Supply (kWh)')
output_filtered['DHW Controller Action - Increase or Decrease of SOC (kW)'].plot(ax = ax[2], color = 'red', label='Controller Action - Increase or Decrease of SOC (kW)')
output_filtered['DHW Heater Electricity Consumption (kWh)'].plot(ax = ax[2], color = 'purple', ls = '--', label='DHW Heater Electricity Consumption (kWh)')
ax[2].set_title('(c) - DWH Storage Utilisation')
ax[2].set(ylabel="Power [kW]")
ax[2].legend(loc="upper right")
# Set minor grid lines
ax[0].xaxis.grid(False) # Just x
ax[0].yaxis.grid(False) # Just x
for j in range(3):
	for xmin in ax[j].xaxis.get_minorticklocs():
		ax[j].axvline(x=xmin, ls='-', color = 'lightgrey')
ax[0].tick_params(direction='out', length=6, width=2, colors='black', top=0, right=0)
plt.setp( ax[0].xaxis.get_minorticklabels(), rotation=0, ha="center" )
plt.setp( ax[0].xaxis.get_majorticklabels(), rotation=0, ha="center" )
# Export Figure
plt.savefig(parent_dir + r"train.jpg", bbox_inches='tight', dpi = 300)
plt.close()

# SET UP RESULTS TABLE

def append_dict_as_row(file_name, dict_of_elem, field_names):
	# Open file in append mode
	with open(file_name, 'a+', newline='') as write_obj:
		# Create a writer object from csv module
		dict_writer = DictWriter(write_obj, fieldnames=field_names)
		# Add dictionary as wor in the csv
		dict_writer.writerow(dict_of_elem)

run_results = {}

## TABULATE KEY VALIDATION RESULTS---------------------------------------------
run_results['Time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
run_results['Time_Training'] = timer
run_results['Time_Training_per_Step'] = timer / (num_episodes*8759)
run_results['Climate'] = climate_zone
run_results['Building'] = building_ids
run_results['Building_Attributes'] = json.load(open(building_attributes))
if env.central_agent == True:
	run_results['Reward_Function'] = 'reward_function_sa'
else:
	run_results['Reward_Function'] = 'reward_function_ma'
run_results['Central_Agent'] = env.central_agent
run_results['Model'] = ''
run_results['Algorithm'] = 'DDPG'
run_results['Directory'] = parent_dir
run_results['Train_Episodes'] = num_episodes
run_results['Ramping'] = env.cost()['ramping']
run_results['1-Load_Factor'] = env.cost()['1-load_factor']
run_results['Average_Daily_Peak'] = env.cost()['average_daily_peak']
run_results['Peak_Demand'] = env.cost()['peak_demand']
run_results['Net_Electricity_Consumption'] = env.cost()['net_electricity_consumption']
run_results['Total'] = env.cost()['total']
run_results['Reward'] = episode_scores[-1]
	
field_names = ['Time','Time_Training','Time_Training_per_Step','Climate','Building','Building_Attributes',
			'Reward_Function','Central_Agent','Model', 'Algorithm', 'Directory',
			'Train_Episodes','Ramping','1-Load_Factor','Average_Daily_Peak','Peak_Demand',
			'Net_Electricity_Consumption','Total','Reward']

append_dict_as_row('test_results.csv', run_results, field_names)