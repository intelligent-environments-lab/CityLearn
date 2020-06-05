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
from torch.utils.tensorboard import SummaryWriter

from citylearn import  CityLearn
from algo_utils import graph_building, tabulate_table

# Extra packages for postprocessing
import matplotlib.dates as dates
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import os
import pprint as pp
from csv import DictWriter

import warnings
warnings.simplefilter("ignore", UserWarning) # Ignore casting to float32 warnings


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
num_episodes=1
episode_scores = []
scores_average_window = 5
checkpoint_interval = 8760

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

# Summary Writer setup
# Writer will output to ./runs/ directory by default
writer = SummaryWriter(log_dir=parent_dir+"tensorboard/")
print("Saving TB to {}".format(parent_dir+"tensorboard/"))

# Crate the final dir
final_dir = parent_dir + "final/"

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
	iteration_step = 0
	iteration_interval = 100
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

		if iteration_step % iteration_interval == 0:

			buildings_reward_dict = {}
			building_idx=1
			for building in reward:
				buildings_reward_dict["Building {}".format(building_idx)] = building
				building_idx += 1
			# Building reward
			writer.add_scalars("Reward/Buildings", buildings_reward_dict, iteration_step)

			agent_scores_dict = {}
			agent_idx=1
			for agentS in agent_scores:
				agent_scores_dict["Agent {}".format(agent_idx)] = agentS
				agent_idx += 1
			# Agent scores
			writer.add_scalars("Scores/Agents", agent_scores_dict, iteration_step)

			# Plot losses for critic and actor
			if agent.critic_loss is not None:
				writer.add_scalar("Losses/Critic Loss", agent.critic_loss, iteration_step)
			if agent.actor_loss is not None:
				writer.add_scalar("Losses/Actor Loss", agent.actor_loss, iteration_step)

			# Action choices
			writer.add_histogram("Action Tracker", np.array(agent.action_tracker), iteration_step)


		# Save trained Actor and Critic network weights for each agent periodically 
		if iteration_step % checkpoint_interval == 0:
			os.makedirs(parent_dir+"chk/step_{}".format(iteration_step), exist_ok=True)
			for building in range(0,num_agents):
				an_filename = "/ddpgActor{0}_Model.pth".format(building)
				torch.save(agent.actor_local[building].state_dict(), parent_dir + "chk/step_{}".format(iteration_step) + an_filename)
				cn_filename = "/ddpgCritic{0}_Model.pth".format(building)
				torch.save(agent.critic_local[building].state_dict(), parent_dir + "chk/step_{}".format(iteration_step) + cn_filename)
			print("Saving a checkpoint to {}".format(parent_dir + "chk/step_{}".format(iteration_step)))

		# If any agent indicates that the episode is done, 
		# then exit episode loop, to begin new episode
		if np.any(done):
			break

		iteration_step += 1
	
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
		os.makedirs(final_dir, exist_ok=True)

		print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\n'.format(i_episode, average_score))
		
		# Save trained  Actor and Critic network weights for each agent
		for building in range(0,num_agents):
			an_filename = "ddpgActor{0}_Model.pth".format(building)
			torch.save(agent.actor_local[building].state_dict(), final_dir + an_filename)
			cn_filename = "ddpgCritic{0}_Model.pth".format(building)
			torch.save(agent.critic_local[building].state_dict(), final_dir + cn_filename)
		
		# Save the recorded Scores data
		scores_filename = "ddpgAgent_Scores_new.csv"
		np.savetxt(final_dir + scores_filename, episode_scores, delimiter=",")
		break

"""
###################################
STEP 5: Everything is Finished -> Close the Environment.
"""

env.close()
writer.close()

"""
###################################
STEP 6: POSTPROCESSING
"""

# Building to plot results for
building_number = 'Building_1'

graph_building(building_number=building_number, env=env, agent=agent, parent_dir=final_dir)

# SET UP RESULTS TABLE

tabulate_table(env=env, timer=timer, algo="DDPG", climate_zone=climate_zone, building_ids=building_ids,
	building_attributes=building_attributes, parent_dir=final_dir, num_episodes=num_episodes, episode_scores=episode_scores)
