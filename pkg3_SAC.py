#!/usr/bin/env python
# coding: utf-8

"""
Implementation of Soft Actor Critic (SAC) network
using PyTorch and Stable Baselines 3.
See https://stable-baselines3.readthedocs.io/en/master/modules/sac.html for algorithm details.

@author: Kacper Twardowski 2020 (kanexer@gmail.com) 

Project for CityLearn Competition
"""

import torch
import numpy as np
import gym
from stable_baselines3.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import make_vec_env, results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from citylearn import  CityLearn
from pkg3_utils import available_cpu_count as cpu_cores, SaveOnBestTrainingRewardCallback, TensorboardCallback
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time, sys, os
import pprint as pp
from torch.utils.tensorboard import writer

# Ignore the float32 bound precision warning
warnings.simplefilter("ignore", UserWarning)

# Central agent controlling the buildings using the OpenAI Stable Baselines
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ['Building_1',"Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
# building_ids = ['Building_1', 'Building_2']
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions,
                cost_function = objective_function, central_agent = True, verbose = 1)

# Store the weights and scores in a new directory
parent_dir = "alg/sac_{}/".format(time.strftime("%Y%m%d-%H%M%S")) # apprends the timedate
os.makedirs(parent_dir, exist_ok=True)

# Create log dir
log_dir = parent_dir+"monitor"
os.makedirs(log_dir, exist_ok=True)

# Set the interval and their count
interval = 8760
icount = int(sys.argv[1]) if len(sys.argv) > 1 else 10
log_interval = 1
check_interval = 1

# Policy kwargs
policy_kwargs = dict(
    net_arch=[128,128]
)

# Make VecEnv + Wrap in Monitor
env = Monitor(env, filename=log_dir)
callbackBest = SaveOnBestTrainingRewardCallback(check_freq=check_interval*interval, log_dir=log_dir)
callbackTB = TensorboardCallback()

# Add callbacks to the callback list
callbackList = []
useBestCallback = True
useTensorboardCallback = True # Not working yet

if useBestCallback:
    callbackList.append(callbackBest)

if useTensorboardCallback:
    callbackList.append(callbackTB)

model = SAC(MlpPolicy_SAC, env, verbose=1, learning_rate=0.005, gamma=0.99, tau=3e-4, batch_size=2048, train_freq=25,
    target_update_interval=25, policy_kwargs=policy_kwargs, learning_starts=interval-1, tensorboard_log=parent_dir+"tensorboard/")
print()

model.learn(total_timesteps=interval*icount, log_interval=log_interval, tb_log_name="", callback=callbackList)
# Summary Writer setup
# Writer will output to ./runs/ directory by default
writer = writer.SummaryWriter(log_dir=parent_dir+"tensorboard/_1")
print("Saving TB to {}".format(parent_dir+"tensorboard/_1"))

iteration_step = 0
obs = env.reset()
dones = False
counter = []

# One episode
while dones==False:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    counter.append(rewards)

    # Logging
    if iteration_step % interval:

		# Building reward
        writer.add_scalar("Reward/Buildings", rewards, iteration_step)

    iteration_step += 1

# Costs
writer.add_scalars("Scores", env.cost(), iteration_step)
# writer.add_scalar("Scores/ramping", env.cost()['ramping'], iteration_step)
# writer.add_scalar("Scores/1-load_factor", env.cost()['1-load_factor'], iteration_step)
# writer.add_scalar("Scores/average_daily_peak", env.cost()['average_daily_peak'], iteration_step)
# writer.add_scalar("Scores/peak_demand", env.cost()['peak_demand'], iteration_step)
# writer.add_scalar("Scores/net_electricity_consumption", env.cost()['net_electricity_consumption'], iteration_step)
# writer.add_scalar("Scores/total", env.cost()['total'], iteration_step)

env.close()

print("\nFinal rewards:")
pp.pprint(env.cost())

# Plot the reward graph
# plot_results([log_dir], interval*icount, results_plotter.X_TIMESTEPS, "SAC CityLearn")
# plt.savefig(log_dir+"/rewards.pdf")