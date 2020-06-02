import torch
import numpy as np
import gym
from stable_baselines3.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines3 import SAC
from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import time
import pprint as pp

# Ignote TF Deprecation warnings (for TF < 2)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Central agent controlling the buildings using the OpenAI Stable Baselines
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
# building_ids = ['Building_1',"Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
building_ids = ['Building_1', 'Building_2']
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = True, verbose = 1)

# Set the interval and their count
interval = 8760
icount = 10

model = SAC(MlpPolicy_SAC, env, verbose=1, learning_rate=0.001, gamma=0.99, batch_size=1024, learning_starts=interval-1)

model.learn(total_timesteps=interval*icount, log_interval=1000)

obs = env.reset()
dones = False
counter = []
while dones==False:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    counter.append(rewards)
pp.pprint(env.cost())