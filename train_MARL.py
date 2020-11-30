#!/usr/bin/env python
# coding: utf-8

"""
Implementation of Decentralised Soft Actor Critic (SAC) network
using PyTorch.
See https://arxiv.org/pdf/1801.01290.pdf for algorithm details.

@author: Anjukan Kathirgamanathan 2020 (k.anjukan@gmail.com)

"""

# Import Packages
import argparse
import numpy as np
import itertools
import torch
from agent_MARL import RL_Agents, RBC_Agent
from torch.utils.tensorboard import SummaryWriter
from citylearn import  CityLearn
from pathlib import Path
import os, time, warnings
from PIL import Image
from torchvision.transforms import ToTensor

from algo_utils import graph_total, graph_building, tabulate_table

import sklearn
import sklearn.preprocessing

# Ignore the casting to float32 warnings
warnings.simplefilter("ignore", UserWarning)

#%%
"""
###################################
STEP 1: Set the Training Parameters
======
        To complete
"""

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--seed', type=int, default=101, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_episodes', type=int, default=25, metavar='N',
                    help='Number of episodes to train for (default: 1000000)')
parser.add_argument('--start_steps', type=int, default=2208*1, metavar='N',
                    help='Steps sampling random actions (default: 8760)')
parser.add_argument('--checkpoint_interval', type=int, default=10, metavar='N',
                    help='Saves a checkpoint with actor/critic weights every n episodes')
parser.add_argument('--control_step', type=int, default=4, metavar='N',
                    help='how often to apply control setpoint')

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
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, simulation_period=(3624,5832), central_agent = False, verbose = 0)
RBC_env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, simulation_period=(3624,5832), central_agent = False, normalise = True, verbose = 0)

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
observations_spaces, actions_spaces = env.get_state_action_spaces()
observations_spacesRBC, actions_spacesRBC = RBC_env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

#%%
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
parent_dir = "alg/MARL/sac_{}/".format(time.strftime("%Y%m%d-%H%M%S")) # apprends the timedate
os.makedirs(parent_dir, exist_ok=True)

# Create the final dir
final_dir = parent_dir+"final/"
os.makedirs(final_dir, exist_ok=True)

# Tensorboard writer object
writer = SummaryWriter(log_dir=parent_dir+'tensorboard/')
print("Logging to {}\n".format(parent_dir+'tensorboard/'))

# Set seeds (TO DO: CHECK PERFORMANCE SAME FOR TWO RUNS WITH SAME SEED)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Get the Rule Base Controller baseline actions
RBC_agent = RBC_Agent(actions_spacesRBC)
state = RBC_env.reset()
state_list = []
action_list = []
doneRBC = False
while not doneRBC:
    action_RBC = RBC_agent.select_action([list(RBC_env.buildings.values())[0].sim_results['hour'][RBC_env.time_step]])
    action_list.append(action_RBC)
    state_list.append(state)
    next_stateRBC, rewardsRBC, doneRBC, _ = RBC_env.step(action_RBC)
    state = next_stateRBC
RBC_action_base = np.array(action_list)
RBC_state_base = np.array(state_list)

# Sample from state space for state normalization
#scaler = []
#for uid in building_ids:
#    scaler[uid] = sklearn.preprocessing.StandardScaler()
#    scaler[uid].fit(RBC_state_base)

#function to normalize states
#def scale_state(state):                 #requires input shape=(2,)
#	scaled = scaler.transform([state])

#	return scaled.transpose().reshape((len(state),))                       #returns shape =(1,2)

#%%
"""
###################################
STEP 3: Create SAC Agent from the Agent Class in sac.py
A SAC agent initialized with the following parameters.
======
To be completed
"""

# Agent

agents = RL_Agents(building_ids, building_info, observations_spaces, actions_spaces, env, args, evaluate = False)

#%%
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

best_reward = 1.0

# Measure the time taken for training
start_timer = time.time()

for i_episode in itertools.count(1):
    # Initialise episode rewards
    episode_reward = [0] * len(building_ids)
    episode_steps = 0
    done = False
    state = env.reset()

    # For every step
    while not done:

        if episode_steps == 0 or episode_steps % args.control_step == 0:
            action = [0] * len(building_ids)
            for uid in building_ids:
                action[int(uid[-1])-1] = agents.select_action(state[int(uid[-1])-1], uid)
        else:
            action = action

        agents.action_tracker.append(np.array(action))

        if len(agents.memory) > agents.batch_size:
            # Update parameters of all the networks
            if total_numsteps % agents.update_interval == 0:
                
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agents.update_parameters(total_numsteps)

                # Tensorboard log policy metrics
                writer.add_scalar('loss/critic_1', critic_1_loss,total_numsteps)
                writer.add_scalar('loss/critic_2', critic_2_loss, total_numsteps)
                writer.add_scalar('loss/policy', policy_loss, total_numsteps)
                writer.add_scalar('loss/entropy_loss', ent_loss, total_numsteps)
                writer.add_scalar('entropy_temprature/alpha', alpha, total_numsteps)
        
        # Step
        next_state, reward, done, _ = env.step(action)

        # Append transition to memory
        agents.add_to_buffer(state, action, reward, next_state, done) 

        episode_steps += 1
        total_numsteps += 1
        for uid in building_ids:
            episode_reward[int(uid[-1])-1] += agents.reward_tracker[-1][int(uid[-1])-1]

        state = next_state
        #if total_numsteps == 1:
        #    sys.exit()

    # Tensorboard log reward values
    writer.add_scalar("Reward/Total", sum(episode_reward), total_numsteps)
    writer.add_scalar("Reward/Building_1", episode_reward[0], total_numsteps)
    writer.add_scalar("Reward/Building_2", episode_reward[1], total_numsteps)
    writer.add_scalar("Reward/Building_3", episode_reward[2], total_numsteps)
    writer.add_scalar("Reward/Building_4", episode_reward[3], total_numsteps)
    writer.add_scalar("Reward/Building_5", episode_reward[4], total_numsteps)
    writer.add_scalar("Reward/Building_6", episode_reward[5], total_numsteps)
    writer.add_scalar("Reward/Building_7", episode_reward[6], total_numsteps)
    writer.add_scalar("Reward/Building_8", episode_reward[7], total_numsteps)
    writer.add_scalar("Reward/Building_9", episode_reward[8], total_numsteps)
	
    # Tensorboard log citylearn cost function
    writer.add_scalar("Scores/ramping", env.cost()['ramping'], total_numsteps)
    writer.add_scalar("Scores/1-load_factor", env.cost()['1-load_factor'], total_numsteps)
    writer.add_scalar("Scores/average_daily_peak", env.cost()['average_daily_peak'], total_numsteps)
    writer.add_scalar("Scores/peak_demand", env.cost()['peak_demand'], total_numsteps)
    writer.add_scalar("Scores/net_electricity_consumption", env.cost()['net_electricity_consumption'], total_numsteps)
    writer.add_scalar("Scores/total", env.cost()['total'], total_numsteps)
            
    print("Episode: {}, total numsteps: {}, total cost: {}, reward: {}".format(i_episode, total_numsteps, round(env.cost()['total'],5), round(sum(episode_reward), 2)))

    # Save trained Actor and Critic network periodically as a checkpoint if it's the best model achieved
    if i_episode % args.checkpoint_interval == 0:
        if env.cost()['total'] < best_reward:
            best_reward = env.cost()['total']
            print("Saving new best model to {}".format(parent_dir))
            agents.save_model(parent_dir)

    # If training episodes completed
    if i_episode > args.num_episodes - 1:
        break

env.close()

timer = time.time() - start_timer

"""
###################################
STEP 5: POSTPROCESSING
"""

# Plot District level power consumption
graph_total(env=env, RBC_env = RBC_env, agent=agents, parent_dir=final_dir, start_date = '2017-08-01', end_date = '2017-08-10')

action_index = 0
i = 0

# Plot individual building power consumption and agent actions
for building in building_ids:
    # Graph district energy consumption and agent behaviour
    graph_building(building_number=building, env=env, RBC_env = RBC_env, agent=agents, parent_dir=final_dir, start_date = '2017-08-01', end_date = '2017-08-10', action_index = action_index)
    
    if env.central_agent == True:
        action_index = action_index + agents.act_size[i]
    i = i + 1

# Tabulate run parameters in training log
tabulate_table(env=env, timer=timer, algo="SAC", agent = agents, climate_zone=climate_zone, building_ids=building_ids, 
               building_attributes=building_attributes, parent_dir=final_dir, num_episodes=i_episode, episode_scores=[episode_reward])
