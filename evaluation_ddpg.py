import gym
# import Chargym_Charging_Station
import gym
import numpy as np
import os
import argparse
# from Solvers.RBC.RBC import RBC
import time
import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CityLearn")
parser.add_argument("--reset_flag", default=1, type=int)
args = parser.parse_args()
env = gym.make(args.env)
path=os.getcwd()
print(path)
models_path=path+'\RESULTS'
ac1 = torch.load(models_path+'\ actor50.pt')
ac2 = torch.load(models_path+'\ actor100.pt')
episodes = 100
final_reward_DDPG_1 = [0] * episodes
final_reward_DDPG_2 = [0] * episodes
final_reward_rbc = [0] * episodes
for ep in range(episodes):
    rewards_list_DDPG_1 = []
    rewards_list_DDPG_2 = []
    rewards_list_rbc = []
    # Note that reset_flag=0 means that the environment simulates/emulates a new day and reset_flag=1 means that simulates the same day.
    # This way, in order to compare two RL algorithms with the RBC we need to specify reset_flag=0 at the start of each episode (right before) the DDPG
    # and change to reset_flag=1 for the other two methods within the same episode. This way it will simulate the same day for all three approaches at each episode,
    # but diverse days across episodes.

    ##########obs = env.reset(0)##########

    #DDPG_1
    obs = env.reset(reset_flag=0)
    done = False
    while not done:
        action = ac1.act(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward_DDPG_1, done, info = env.step(action)
        rewards_list_DDPG_1.append(reward_DDPG_1)

    final_reward_DDPG_1[ep] = sum(rewards_list_DDPG_1)
    #DDPG_2
    obs = env.reset(reset_flag=1)
    done = False
    while not done:
        action = ac2.act(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward_DDPG_2, done, info = env.step(action)
        rewards_list_DDPG_2.append(reward_DDPG_2)


    final_reward_DDPG_2[ep] = sum(rewards_list_DDPG_2)
    # RBC case
    ##########obs = env.reset(1)##########
    obs = env.reset(reset_flag=1)
    done = False
    while not done:
        # state = obs
        action_rbc = RBC.select_action(env.env, obs)
        obs, rewards_rbc, done, _ = env.step(action_rbc)
        # print(rewards)
        # obs = next_state_rbc
        rewards_list_rbc.append(rewards_rbc)
    final_reward_rbc[ep] = sum(rewards_list_rbc)
env.close


Mean_reward_DDPG = np.mean(final_reward_DDPG_1)
Mean_reward_PPO = np.mean(final_reward_DDPG_2)
Mean_reward_RBC = np.mean(final_reward_rbc)

print(Mean_reward_DDPG)
print(Mean_reward_PPO)
print(Mean_reward_RBC)
plt.rcParams.update({'font.size': 18})
plt.plot(final_reward_DDPG_1)
plt.plot(final_reward_DDPG_2)
plt.plot(final_reward_rbc)
plt.xlabel('Evaluation episodes')
plt.ylabel('Reward')
plt.legend(['DDPG_1', 'DDPG_2', 'RBC'])

plt.show()

a = 1




