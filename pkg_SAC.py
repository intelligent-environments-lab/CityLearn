from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines import SAC
from citylearn import  CityLearn
from pathlib import Path
import pprint as pp

# Extra packages for postprocessing
import time
import os
from algo_utils import graph_building, tabulate_table

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
building_ids = ['Building_1',"Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
#building_ids = ['Building_1']
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = True, verbose = 1)

# Set the interval and their count
interval = 8760
icount = 20

# Measure the time taken for training
start_timer = time.time()

# Store the weights and scores in a new directory
parent_dir = "alg/SAC{}/".format(time.strftime("%Y%m%d-%H%M%S")) # apprends the timedate
os.makedirs(parent_dir, exist_ok=True)

model = SAC(MlpPolicy_SAC, env, verbose=1, learning_rate=0.005, tau = 3e-3, gamma=0.99, batch_size=2048, learning_starts=interval-1, 
            train_freq=25, target_update_interval=25, buffer_size=500000, tensorboard_log=parent_dir)

# Crate the final dir
final_dir = parent_dir + "final/"

model.learn(total_timesteps=interval*icount, log_interval=1000)

timer = time.time() - start_timer

obs = env.reset()
dones = False
counter = []
while dones==False:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    counter.append(rewards)

env.close()

pp.pprint(env.cost())

"""
###################################
STEP 7: POSTPROCESSING
"""

# Building to plot results for
building_number = 'Building_1'

# TO DO: THIS FUNCTION IS NOT COMPATIBLE WITH THIS SAC IMPLEMENTATION YET!
#graph_building(building_number=building_number, env=env, agent=model, parent_dir=final_dir)

tabulate_table(env=env, timer=timer, algo="SAC", climate_zone=climate_zone, building_ids=building_ids, 
               building_attributes=building_attributes, parent_dir=final_dir, num_episodes=icount, episode_scores=[sum(counter)])