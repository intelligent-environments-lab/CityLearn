#!/usr/bin/env python
# coding: utf-8

"""
Implementation of Soft Actor Critic (SAC) network
using PyTorch.
See https://arxiv.org/pdf/1801.01290.pdf for algorithm details.

@author: Anjukan Kathirgamanathan 2020 (k.anjukan@gmail.com) 

Project for CityLearn Competition
"""

# Import Packages
import argparse
import numpy as np
import itertools
import torch
from agent_SAC import SAC
from torch.utils.tensorboard import SummaryWriter
from citylearn import  CityLearn
from pathlib import Path
import os, time, warnings

from algo_utils import graph_building, tabulate_table

# Ignore the casting to float32 warnings
warnings.simplefilter("ignore", UserWarning)

"""
###################################
STEP 1: Set the Training Parameters
======
        To complete
"""

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_episodes', type=int, default=200, metavar='N',
                    help='Number of episodes to train for (default: 1000000)')
parser.add_argument('--start_steps', type=int, default=8760, metavar='N',
                    help='Steps sampling random actions (default: 8760)')
parser.add_argument('--update_interval', type=int, default=8760, metavar='N',
                    help='Update network parameters every n steps')
parser.add_argument('--checkpoint_interval', type=int, default=10, metavar='N',
                    help='Saves a checkpoint with actor/critic weights every n episodes')
args = parser.parse_args()

# Environment
# Central agent controlling the buildings using the OpenAI Stable Baselines
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ['Building_1',"Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
#building_ids = ['Building_1']
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = True, verbose = 0)

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

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

# Get number of buildings and agents in Environment
num_buildings = env.n_buildings
print('\nNumber of Buildings: ', num_buildings)

print('\nCentral Agent: ', env.central_agent)

# Set the size of state observations or state size
print('\nSize of State: ', observations_spaces)
print('\n')

# Store the weights and scores in a new directory
parent_dir = "alg/sac_{}/".format(time.strftime("%Y%m%d-%H%M%S")) # apprends the timedate
os.makedirs(parent_dir, exist_ok=True)

# Create the final dir
final_dir = parent_dir+"final/"
os.makedirs(final_dir, exist_ok=True)

#Tensorboard
writer = SummaryWriter(log_dir=parent_dir+'tensorboard/')
print("Logging to {}\n".format(parent_dir+'tensorboard/'))

torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

"""
###################################
STEP 3: Create SAC Agent from the Agent Class in sac.py
A SAC agent initialized with the following parameters.
======
To be completed
"""

# Agent
agent = SAC(env, env.observation_space.shape[0], env.action_space, args)

"""
###################################
STEP 4: Run the SAC Training Sequence
The SAC Training Process involves the agent learning from repeated episodes of behaviour 
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

# Training Loop
total_numsteps = 0
updates = 0

# Measure the time taken for training
start_timer = time.time()

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    # Start logging after the first timestep to avoid a sudden jump in the graph
    log_step = 0 if i_episode != 1 else 1

    while not done:
        if args.start_steps > total_numsteps:
            # Sample action from policy
            action = env.action_space.sample()
            agent.action_tracker.append(action)
        else:
            # Sample action from policy
            action = agent.select_action(state)

        # Action choices
        cooling = action[0]
        dhw = action[1]

        if episode_steps == log_step:
            writer.add_histogram("Action/Cooling", cooling, total_numsteps)
            writer.add_histogram("Action/DHW", dhw, total_numsteps)
            writer.add_histogram("Action/Tracker", np.array(agent.action_tracker), total_numsteps)

        if len(agent.memory) > agent.batch_size:
            if total_numsteps % args.update_interval == 0:
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(total_numsteps)

                writer.add_scalar('loss/critic_1', critic_1_loss,total_numsteps)
                writer.add_scalar('loss/critic_2', critic_2_loss, total_numsteps)
                writer.add_scalar('loss/policy', policy_loss, total_numsteps)
                writer.add_scalar('loss/entropy_loss', ent_loss, total_numsteps)
                writer.add_scalar('entropy_temprature/alpha', alpha, total_numsteps)
        
        next_state, reward, done, _ = env.step(action) # Step

        # Append transition to memory
        reward = agent.append(state, action, reward, next_state, done) 

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        state = next_state
        #if total_numsteps == 10:
        #    sys.exit()

    writer.add_scalar('Reward/Buildings', episode_reward, total_numsteps)
	
    #print(env.cost())
    writer.add_scalar("Scores/ramping", env.cost()['ramping'], total_numsteps)
    writer.add_scalar("Scores/1-load_factor", env.cost()['1-load_factor'], total_numsteps)
    writer.add_scalar("Scores/average_daily_peak", env.cost()['average_daily_peak'], total_numsteps)
    writer.add_scalar("Scores/peak_demand", env.cost()['peak_demand'], total_numsteps)
    writer.add_scalar("Scores/net_electricity_consumption", env.cost()['net_electricity_consumption'], total_numsteps)
    writer.add_scalar("Scores/total", env.cost()['total'], total_numsteps)

    print("Episode: {}, total numsteps: {}, total cost: {}, reward: {}".format(i_episode, total_numsteps, round(env.cost()['total'],5), round(episode_reward, 2)))

    # Save trained Actor and Critic network weights for each agent periodically 
    if i_episode % args.checkpoint_interval == 0:
        agent.save_model(parent_dir)

    if i_episode > args.num_episodes - 1:
        break

env.close()

timer = time.time() - start_timer

"""
###################################
STEP 7: POSTPROCESSING
"""

# Building to plot results for
building_number = 'Building_1'

# TO DO: THIS FUNCTION IS NOT COMPATIBLE WITH THIS SAC IMPLEMENTATION YET!
graph_building(building_number=building_number, env=env, agent=agent, parent_dir=final_dir)

tabulate_table(env=env, timer=timer, algo="SAC", climate_zone=climate_zone, building_ids=building_ids, 
               building_attributes=building_attributes, parent_dir=final_dir, num_episodes=i_episode, episode_scores=[episode_reward])