# Ignote TF Deprecation warnings (for TF < 2) & User Warnings
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import gym
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG, results_plotter
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import time
import pprint as pp
import os, sys
from stable_baselines.bench.monitor import Monitor, load_results
from stable_baselines.results_plotter import plot_results, ts2xy

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
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback2_10, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
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

        return True

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

# Store the weights and scores in a new directory
parent_dir = "alg/ddpg_{}/".format(time.strftime("%Y%m%d-%H%M%S")) # apprends the timedate
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
callbackBest = SaveOnBestTrainingRewardCallback2_10(check_freq=check_interval*interval, log_dir=log_dir)

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

model = DDPG(policy=MlpPolicy, policy_kwargs=policy_kwargs, env=env, verbose=0, param_noise=param_noise, action_noise=action_noise, tensorboard_log=parent_dir+"tensorboard/")

model.learn(total_timesteps=interval*icount, log_interval=interval, tb_log_name="DDPG_{}".format(time.strftime("%Y%m%d")), callback=callbackList)

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
    plot_results([log_dir], interval*icount, results_plotter.X_TIMESTEPS, "DDPG CityLearn")
    plt.savefig(log_dir+"/rewards.pdf")