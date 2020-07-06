#!/usr/bin/env python
# coding: utf-8

"""
Implementation of Soft Actor Critic (SAC) network
using PyTorch.
See https://arxiv.org/pdf/1801.01290.pdf for algorithm details.

@author: Anjukan Kathirgamanathan 2020 (k.anjukan@gmail.com) and Kacper
Twardowski (kanexer@gmail.com) 

Project for CityLearn Competition
"""

# Import Packages
import argparse
import numpy as np
import itertools
import torch
from agent_SAC import SAC, RBC_Agent
from torch.utils.tensorboard import SummaryWriter
from citylearn import  CityLearn
from pathlib import Path
import os, time, warnings
from PIL import Image
from torchvision.transforms import ToTensor

from algo_utils import graph_total, graph_building, tabulate_table
from autoencoder import Autoencoder

# Ignore the casting to float32 warnings
warnings.simplefilter("ignore", UserWarning)

"""
###################################
STEP 1: Set the Training Parameters
======
        To complete
"""

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_episodes', type=int, default=100, metavar='N',
                    help='Number of episodes to train for (default: 1000000)')
parser.add_argument('--start_steps', type=int, default=8760*1, metavar='N',
                    help='Steps sampling random actions (default: 8760)')
parser.add_argument('--checkpoint_interval', type=int, default=10, metavar='N',
                    help='Saves a checkpoint with actor/critic weights every n episodes')
args = parser.parse_args()

# Autoencoder parameters
NEW_STATE_SIZE = 8
AE_EPOCHS = 5000

# Environment
# Central agent controlling the buildings using the OpenAI Stable Baselines
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ['Building_1',"Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
# building_ids = ['Building_3']
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = True, verbose = 0)
RBC_env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = False, verbose = 0)

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
observations_spaces, actions_spaces = env.get_state_action_spaces()
observations_spacesRBC, actions_spacesRBC = RBC_env.get_state_action_spaces()

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

# Tensorboard writer object
writer = SummaryWriter(log_dir=parent_dir+'tensorboard/')
print("Logging to {}\n".format(parent_dir+'tensorboard/'))

# Set seeds (TO DO: CHECK PERFORMANCE SAME FOR TWO RUNS WITH SAME SEED)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Get the Rule Base Controller baseline actions
agent = RBC_Agent(actions_spacesRBC)
state = RBC_env.reset()
state_list = []
action_list = []
doneRBC = False
while not doneRBC:
    action = agent.select_action([list(RBC_env.buildings.values())[0].sim_results['hour'][RBC_env.time_step]])
    action_list.append(action)
    state_list.append(state)
    next_stateRBC, rewardsRBC, doneRBC, _ = RBC_env.step(action)
    state = next_stateRBC
RBC_action_base = np.array(action_list)
RBC_state_base = np.array(state_list)
RBC_24h_peak = [day.max() for day in np.append(RBC_env.net_electric_consumption,0).reshape(-1, 24)]

"""
###################################
STEP 3: Create SAC Agent from the Agent Class in sac.py
A SAC agent initialized with the following parameters.
======
To be completed
"""

# Agent
agent = SAC(env, env.observation_space.shape[0], env.action_space, args, constrain_action_space=False and env.central_agent, smooth_action_space = True, evaluate = False)#, continue_training = True)

# Sample a year of random actions to feed into the AE
device = ('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Autoencoder(in_shape=env.observation_space.shape[0]-3, enc_shape=NEW_STATE_SIZE-3).double().to(device)
action_list_AE = []
state_list_AE = []
state_AE = env.reset()
done_AE = False
while not done_AE:
    action_AE = env.action_space.sample()
    action_list_AE.append(action_AE)
    state_list_AE.append(state_AE[3:])
    state_AE, reward_AE, done_AE, _ = env.step(action_AE)
state_array_AE = np.array(state_list_AE)
state_scaled_AE = encoder.scaler.fit_transform(state_array_AE)
state_tensor_AE = torch.from_numpy(state_scaled_AE).to(device)
encoder.train_model(AE_EPOCHS, state_tensor_AE)
agent = SAC(env, NEW_STATE_SIZE, env.action_space, args, constrain_action_space=False and env.central_agent, smooth_action_space = True, evaluate = False)

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

best_reward = 0.95

# Measure the time taken for training
start_timer = time.time()

for i_episode in itertools.count(1):

    # Initialise episode rewards
    episode_reward = 0
    episode_peak_reward = 0
    episode_day_reward = 0
    episode_night_reward = 0
    episode_smooth_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    temporal_state = state[:3]
    state = state[3:]
    state = encoder.encode_min(state)
    state = temporal_state.tolist() + state

    grads_G1_daily = []
    grads_G1_weekly = []

    grads_G2_daily = []
    grads_G2_weekly = []

    # For every step
    while not done:
        # If learning hasn't started yet, sample random action
        if args.start_steps > total_numsteps:
            # 
            action = env.action_space.sample()
            agent.action_tracker.append(action)
        # Else sample action from policy
        else:
            # state = [1] * NEW_STATE_SIZE
            action = agent.select_action(state)

        if len(agent.memory) > agent.batch_size:
            # Update parameters of all the networks
            if total_numsteps % agent.update_interval == 0:
                
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(total_numsteps)

                # Tensorboard log policy metrics
                writer.add_scalar('loss/critic_1', critic_1_loss,total_numsteps)
                writer.add_scalar('loss/critic_2', critic_2_loss, total_numsteps)
                writer.add_scalar('loss/policy', policy_loss, total_numsteps)
                writer.add_scalar('loss/entropy_loss', ent_loss, total_numsteps)
                writer.add_scalar('entropy_temprature/alpha', alpha, total_numsteps)
        
        # Step
        next_state, reward, done, _ = env.step(action)
        temporal_state = next_state[:3]
        next_state = encoder.encode_min(next_state[3:])
        next_state = temporal_state.tolist() + next_state

        # Net Energy Consumption
        #this_netRBC = env.net_electric_consumption_no_storage[episode_steps]
        #this_netSAC = env.net_electric_consumption[episode_steps]
        #reward = (this_netRBC * (1 - this_netSAC/RBC_24h_peak[episode_steps%24]))
        # print(reward)

        grads_G1_daily.append(env.net_electric_consumption[-1]/env.net_electric_consumption[-2] if episode_steps != 0 else 1)
        grads_G1_weekly.append(env.net_electric_consumption[-1]/env.net_electric_consumption[-2] if episode_steps != 0 else 1)
        grads_G2_daily.append(grads_G1_daily[-1]/grads_G1_daily[-2] if episode_steps > 1 else 1)
        grads_G2_weekly.append(grads_G1_weekly[-1]/grads_G1_weekly[-2] if episode_steps > 1 else 1)
        writer.add_scalar('Gradients/G0/Net', env.net_electric_consumption[-1], total_numsteps)
        if episode_steps % 24 == 0 and episode_steps > 0:
            grad_daily = sum(grads_G1_daily)/len(grads_G1_daily)
            writer.add_scalar('Gradients/G1/Daily Average', grad_daily, total_numsteps/24)
            grads_daily = []
        if episode_steps % (7*24) == 0 and episode_steps > 0:
            grad_weekly = sum(grads_G1_weekly)/len(grads_G1_weekly)
            writer.add_scalar('Gradients/G1/Weekly Average', grad_weekly, total_numsteps/(7*24))
            grads_daily = []
        if episode_steps % 24 == 0 and episode_steps > 1:
            grad_daily = sum(grads_G2_daily)/len(grads_G2_daily)
            writer.add_scalar('Gradients/G2/Daily Average', grad_daily, total_numsteps/24)
            grads_daily = []
        if episode_steps % (7*24) == 0 and episode_steps > 1:
            grad_weekly = sum(grads_G2_weekly)/len(grads_G2_weekly)
            writer.add_scalar('Gradients/G2/Weekly Average', grad_weekly, total_numsteps/(7*24))
            grads_daily = []

        # Append transition to memory
        reward, r_peak, r_day, r_night, r_smooth = agent.add_to_buffer(state, action, reward, next_state, done) 

        # writer.add_scalars('RBC vs SAC/Net Energy Consumption', {'RBC':this_netRBC, 'SAC':this_netSAC}, total_numsteps)

        # Tensorboard net electric consumption gradients
        # writer.add_scalars('Gradients/First Gradient/Net Electric Consumption',
        #     {'Net': env.net_electric_consumption[-1]/env.net_electric_consumption[-2] if episode_steps != 0 else 1,
        #     'No Str, No PV': env.net_electric_consumption_no_pv_no_storage[-1]/env.net_electric_consumption_no_pv_no_storage[-2] if episode_steps != 0 else 1,
        #     'No Str': env.net_electric_consumption_no_storage[-1]/env.net_electric_consumption_no_storage[-2] if episode_steps != 0 else 1}, episode_steps)

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        episode_peak_reward += r_peak
        episode_day_reward += r_day
        episode_night_reward += r_night
        episode_smooth_reward += r_smooth

        state = next_state

    # Tensorboard log reward values
    writer.add_scalar('Reward/Total', episode_reward, total_numsteps)
    writer.add_scalar('Reward/Peak', episode_peak_reward, total_numsteps)
    writer.add_scalar('Reward/Day_Charging', episode_day_reward, total_numsteps)
    writer.add_scalar('Reward/Night_Charging', episode_night_reward, total_numsteps)
    writer.add_scalar('Reward/Smooth_Actions', episode_smooth_reward, total_numsteps)
	
    # Tensorboard log citylearn cost function
    writer.add_scalar("Scores/ramping", env.cost()['ramping'], total_numsteps)
    writer.add_scalar("Scores/1-load_factor", env.cost()['1-load_factor'], total_numsteps)
    writer.add_scalar("Scores/average_daily_peak", env.cost()['average_daily_peak'], total_numsteps)
    writer.add_scalar("Scores/peak_demand", env.cost()['peak_demand'], total_numsteps)
    writer.add_scalar("Scores/net_electricity_consumption", env.cost()['net_electricity_consumption'], total_numsteps)
    writer.add_scalar("Scores/total", env.cost()['total'], total_numsteps)

    # Log how much storage is utilised by calculating abs sum of actions (CHECK IF WORKS WITH MULTIPLE BUILDINGS!!!)
    episode_actions = np.array(agent.action_tracker[-8759:])
    cooling = sum(abs(episode_actions[:,0]))
    writer.add_scalar("Action/Cooling", cooling, total_numsteps)
    if agent.act_size[0] == 2:
        dhw = sum(abs(episode_actions[:,1]))
        writer.add_scalar("Action/DHW", dhw, total_numsteps)
    writer.add_histogram("Action/Tracker", np.array(agent.action_tracker), total_numsteps)
            
    print("Episode: {}, total numsteps: {}, total cost: {}, reward: {}".format(i_episode, total_numsteps, round(env.cost()['total'],5), round(episode_reward, 2)))

    # Save trained Actor and Critic network periodically as a checkpoint if it's the best model achieved
    if i_episode % args.checkpoint_interval == 0:
        if env.cost()['total'] < best_reward:
            best_reward = env.cost()['total']
            print("Saving new best model to {}".format(parent_dir))
            agent.save_model(parent_dir)

    # If training episodes completed
    if i_episode > args.num_episodes - 1:
        break

env.close()

timer = time.time() - start_timer

"""
###################################
STEP 5: POSTPROCESSING
"""

# Building to plot results for
building_number = building_ids[0]

# Plot District level power consumption
graph_total(env=env, RBC_env = RBC_env, agent=agent, parent_dir=final_dir, start_date = '2017-09-01', end_date = '2017-09-10')

divide_lambda = lambda x: int(x/4)
district_graph = Image.open(parent_dir+"final/"+r"district.jpg")
district_graph = district_graph.resize(tuple(map(divide_lambda,district_graph.size)))
writer.add_image("Graph for District/Elec_Consumption", ToTensor()(district_graph))
district_RBC_comp_graph = Image.open(parent_dir+"final/"+r"district_RBC_comp_daily_peak.jpg")
district_RBC_comp_graph = district_graph.resize(tuple(map(divide_lambda,district_graph.size)))
writer.add_image("Graph for District/Daily_Peak", ToTensor()(district_RBC_comp_graph))

action_index = 0
i = 0

# Plot individual building power consumption and agent actions
for building in building_ids:
    # Graph district energy consumption and agent behaviour
    graph_building(building_number=building, env=env, RBC_env = RBC_env, agent=agent, parent_dir=final_dir, start_date = '2017-09-01', end_date = '2017-09-10', action_index = action_index)

    # Add these graphs to the tensorboard
    train_graph = Image.open(parent_dir+"final/"+r"train"+"{}.jpg".format(building[-1]))
    train_graph = train_graph.resize(tuple(map(divide_lambda,train_graph.size)))
    action_graph = Image.open(parent_dir+"final/"+r"actions"+"{}.jpg".format(building[-1]))
    action_graph = action_graph.resize(tuple(map(divide_lambda,action_graph.size)))
    writer.add_image("Graph for {}/Train".format(building), ToTensor()(train_graph))
    writer.add_image("Graph for {}/Actions".format(building), ToTensor()(action_graph))
    
    action_index = action_index + agent.act_size[i]
    i = i + 1

# Tabulate run parameters in training log
tabulate_table(env=env, timer=timer, algo="SAC", agent = agent, climate_zone=climate_zone, building_ids=building_ids, 
               building_attributes=building_attributes, parent_dir=final_dir, num_episodes=i_episode, episode_scores=[episode_reward])