#!/usr/bin/env python
# coding: utf-8

"""
Implementation of Twin Delayed DDPG (TD3) network
using PyTorch and Stable Baselines 3.
See https://stable-baselines3.readthedocs.io/en/master/modules/td3.html for algorithm details.

@author: Kacper Twardowski 2020 (kanexer@gmail.com) 

Project for CityLearn Competition
"""

import torch
import numpy as np
import gym
from stable_baselines3.td3.policies import MlpPolicy as MlpPolicy_TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import make_vec_env, results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from citylearn import  CityLearn
from pkg3_utils import available_cpu_count as cpu_cores, SaveOnBestTrainingRewardCallback
import matplotlib.pyplot as plt
from pathlib import Path
import time, sys, os
import pprint as pp

# Ignore the float32 bound precision warning
import warnings
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
parent_dir = "alg/td3_{}/".format(time.strftime("%Y%m%d-%H%M%S")) # apprends the timedate
os.makedirs(parent_dir, exist_ok=True)

# Create log dir
log_dir = parent_dir+"monitor"
os.makedirs(log_dir, exist_ok=True)

# Set the interval and their count
interval = 8760
icount = int(sys.argv[1]) if sys.argv is not None else 10
log_interval = 1
check_interval = 1

# the noise objects for DDPG
_, actions_spaces = env.get_state_action_spaces()

n_actions = 0
for action in actions_spaces:
    n_actions += action.shape[-1]

# Make VecEnv + Wrap in Monitor
env = Monitor(env, filename=log_dir)
callbackBest = SaveOnBestTrainingRewardCallback(check_freq=check_interval*interval, log_dir=log_dir)

# Add callbacks to the callback list
callbackList = []
useBestCallback = True

if useBestCallback:
    callbackList.append(callbackBest)

# Algo setup
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

policy_kwargs = dict(
    # net_arch=[128,128]
)

model = TD3(policy=MlpPolicy_TD3, policy_kwargs=policy_kwargs, env=env, verbose=0, action_noise=action_noise, learning_starts=interval, tensorboard_log=parent_dir+"tensorboard/")
print()

model.learn(total_timesteps=interval*icount, log_interval=log_interval, tb_log_name="TD3_{}".format(time.strftime("%Y%m%d")), callback=callbackList)

obs = env.reset()
dones = False
counter = []
while dones==False:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    counter.append(rewards)

env.close()

print("\nFinal rewards:")
pp.pprint(env.cost())

# Plot the reward graph
if useBestCallback:
    plot_results([log_dir], interval*icount, results_plotter.X_TIMESTEPS, "TD3 CityLearn")
    plt.savefig(log_dir+"/rewards.pdf")