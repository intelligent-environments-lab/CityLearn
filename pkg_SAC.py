# Ignote TF Deprecation warnings (for TF < 2) & User Warnings
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import numpy as np
import gym
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_SAC
from stable_baselines import SAC
from stable_baselines.common.callbacks import BaseCallback
from citylearn import  CityLearn
from pathlib import Path
import pprint as pp
import sys, multiprocessing
import os
from stable_baselines.results_plotter import ts2xy
from stable_baselines.bench.monitor import Monitor, load_results
from torch.utils.tensorboard import writer


# Callback class for saving the best reward episode, FOR SB 2.10
# Ref: https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
class SaveOnBestTrainingRewardCallback2_10(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, save_freq: int, log_dir: str, verbose=1, interval=8760):
        super(SaveOnBestTrainingRewardCallback2_10, self).__init__(verbose)
        self.save_freq = save_freq
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.log_dir+"/checkpoints")

    def _on_step(self) -> bool:

        if self.n_calls % interval == 0:
            pp.pprint( self.model.get_env() )

        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

            print()

        if self.n_calls % self.save_freq == 0:
            path = log_dir+'/checkpoints/chk_{}'.format(int(self.n_calls/interval))
            if self.verbose > 0:
                print("Saving checkpoint to {}".format(path))
            self.model.save(path)

            print()

        return True


        

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
save_interval = 1

# Policy kwargs
policy_kwargs = dict(
    layers=[128,128]
)
# Measure the time taken for training
start_timer = time.time()

# Make VecEnv + Wrap in Monitor
env = Monitor(env, filename=log_dir)
callbackBest = SaveOnBestTrainingRewardCallback2_10(check_freq=check_interval*interval, log_dir=log_dir, save_freq=interval*save_interval)

timer = time.time() - start_timer

# Add callbacks to the callback list
callbackList = []
useBestCallback = True

if useBestCallback:
    callbackList.append(callbackBest)

model = SAC(LnMlpPolicy_SAC, env, verbose=1, learning_rate=0.005, gamma=0.99, tau=3e-4, batch_size=2048, train_freq=25,
    target_update_interval=25, policy_kwargs=policy_kwargs, learning_starts=interval-1, n_cpu_tf_sess=multiprocessing.cpu_count(), tensorboard_log=parent_dir+"tensorboard/")
print()

model.learn(total_timesteps=interval*icount, log_interval=interval, tb_log_name="", callback=callbackList)

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

        print(rewards)
		# Building reward
        writer.add_scalar("Reward/Buildings", rewards, iteration_step)

    iteration_step += 1

# Costs
writer.add_scalars("Scores", env.cost(), iteration_step)

env.close()

print("\nFinal rewards:")
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