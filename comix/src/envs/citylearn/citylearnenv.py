import csv
import json
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import torch

from .citylearn import CityLearn
from envs.multiagentenv import MultiAgentEnv
from gym.spaces import Box
from joblib import dump, load
from pathlib import Path

################################################################################
#
# Some preprocessing utils from citylearn
#
################################################################################

class no_normalization:
    def __init__(self):
        pass
    def __mul__(self, x):
        return x
        
    def __rmul__(self, x):
        return x
        
class periodic_normalization:
    def __init__(self, x_max):
        self.x_max = x_max
    def __mul__(self, x):
        #x = 2 * np.pi * x / self.x_max
        #x_sin = np.sin(x)
        #x_cos = np.cos(x)
        #return np.array([(x_sin+1)/2.0, (x_cos+1)/2.0])
        return np.array([x, x])
    def __rmul__(self, x):
        #x = 2 * np.pi * x / self.x_max
        #x_sin = np.sin(x)
        #x_cos = np.cos(x)
        #return np.array([(x_sin+1)/2.0, (x_cos+1)/2.0])
        return np.array([x, x])

class onehot_encoding:
    def __init__(self, classes):
        self.classes = classes
    def __mul__(self, x):
        identity_mat = np.eye(len(self.classes))
        return identity_mat[np.array(self.classes) == x][0]
    def __rmul__(self, x):
        identity_mat = np.eye(len(self.classes))
        return identity_mat[np.array(self.classes) == x][0]
    
class normalize:
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
    def __mul__(self, x):
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min)/(self.x_max - self.x_min)
    def __rmul__(self, x):
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min)/(self.x_max - self.x_min)

class remove_feature:
    def __init__(self):
        pass
    def __mul__(self, x):
        return None
    def __rmul__(self, x):
        return None

################################################################################
#
# The gym-like environment
#
################################################################################

class CityLearnEnv(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        # Load environment
        climate_zone = 5
        parent = Path(__file__).parent.absolute()
        data_path = Path(os.path.join(parent, f"data/Climate_Zone_{climate_zone}"))
        #building_ids = ['Building_1', 'Building_2']
        building_ids = ["Building_"+str(i) for i in [1,2,3,4,5,6,7,8,9]]

        buildings_states_actions_file = os.path.join(parent, 'buildings_state_action_space.json')

        params = {'data_path': data_path,
                  'building_attributes':'building_attributes.json', 
                  'weather_file':'weather_data.csv', 
                  'solar_profile':'solar_generation_1kW.csv', 
                  'carbon_intensity':'carbon_intensity.csv',
                  'building_ids': building_ids,
                  'buildings_states_actions':buildings_states_actions_file,
                  'simulation_period': (0, 8760-1),
                  'cost_function': [
                        'ramping',
                        '1-load_factor',
                        'average_daily_peak',
                        'peak_demand',
                        'net_electricity_consumption',
                        'carbon_emissions'], 
                  'central_agent': False,
                  'save_memory': False }

        with open(buildings_states_actions_file) as json_file:
            self.buildings_states_actions = json.load(json_file)

        self.n_agents = len(building_ids)
        self.agent_embedding = np.eye(self.n_agents)

        # Contain the lower and upper bounds of the states and actions, to be
        # provided to the agent to normalize the variables between 0 and 1.
        # Can be obtained using observations_spaces[i].low or .high
        self.env = CityLearn(**params)
        self.original_observation_space, self.original_action_space= \
                self.env.get_state_action_spaces() # the action space are not consistent in shape

        self.n_actions = max([len(a.high) for a in self.original_action_space])

        self.action_space_low = np.zeros((self.n_agents, self.n_actions))
        self.action_space_high = np.zeros((self.n_agents, self.n_actions))
        self.action_mask = np.zeros((self.n_agents, self.n_actions), dtype=bool)

        for i, act_sp in enumerate(self.original_action_space):
            low = act_sp.low
            high = act_sp.high
            x = len(low)
            self.action_space_low[i][:x] = low
            self.action_space_high[i][:x] = high
            self.action_mask[i][:x] = True

        self.action_space = tuple([Box(self.action_space_low[a],
                                       self.action_space_high[a])
                                   for a in range(self.n_agents)])

        # Provides information on Building type, Climate Zone, Annual DHW demand,
        # Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and
        # correllations among buildings
        self.building_info = self.env.get_building_information() # temporarily not used

        self.episode_limit = 8759
        self.n_episode = 0

        self.encoder = {}
        self.encoder_names = {}
        self.encoder_mask = {}
        self.encoder_nums = {}

        self.max_state_dim = 0
        for i, uid in enumerate(building_ids):
            self.encoder_names[uid] = []
            self.encoder[uid] = []
            self.encoder_nums[uid] = []
            self.encoder_mask[uid] = []

            state_n = 0
            pos = 0
            for s_name, s in self.buildings_states_actions[uid]['states'].items():
                if not s:
                    self.encoder_names[uid].append(s_name+"0")
                    self.encoder[uid].append(0)
                    self.encoder_nums[uid].append([])
                    if s_name in ["month", "hour"]:
                        pos += 2
                    elif s_name == "day":
                        pos += 8
                    elif s_name == "net_electricity_comsumption":
                        pos += 1
                    else:
                        pos += 1
                elif s_name in ["month", "hour"]:
                    self.encoder_names[uid].append(s_name+"1")
                    self.encoder[uid].append(periodic_normalization(self.original_observation_space[i].high[state_n]))
                    self.encoder_mask[uid].extend([pos, pos+1])
                    self.encoder_nums[uid].append([pos, pos+1])
                    pos += 2
                    state_n += 1
                elif s_name == "day":
                    self.encoder_names[uid].append(s_name+"1")
                    self.encoder[uid].append(onehot_encoding([1,2,3,4,5,6,7,8]))
                    self.encoder_mask[uid].extend([pos, pos+1, pos+2, pos+3, pos+4, pos+5, pos+6, pos+7])
                    self.encoder_nums[uid].append([pos, pos+1, pos+2, pos+3, pos+4, pos+5, pos+6, pos+7])
                    pos += 8
                    state_n += 1
                elif s_name == "daylight_savings_status":
                    self.encoder_names[uid].append(s_name+"1")
                    self.encoder[uid].append(onehot_encoding([0,1]))
                    self.encoder_mask[uid].extend([pos, pos+1])
                    self.encoder_nums[uid].append([pos, pos+1])
                    pos += 2
                    state_n += 1
                elif s_name == "net_electricity_consumption":
                    self.encoder_names[uid].append(s_name+"1")
                    self.encoder[uid].append(remove_feature())
                    #self.encoder_mask[uid].append(pos)
                    #self.encoder_nums[uid].append([pos])
                    self.encoder_nums[uid].append([])
                    pos += 1
                    state_n += 1
                else:
                    self.encoder_names[uid].append(s_name+"1")
                    self.encoder[uid].append(normalize(self.original_observation_space[i].low[state_n], self.original_observation_space[i].high[state_n]))
                    self.encoder_mask[uid].append(pos)
                    self.encoder_nums[uid].append([pos])
                    pos += 1
                    state_n += 1  

            self.encoder[uid] = np.array(self.encoder[uid])
            self.encoder_names[uid] = np.array(self.encoder_names[uid])

            # If there is no solar PV installed, remove solar radiation variables 
            if self.building_info[uid]['solar_power_capacity (kW)'] == 0:
                for k in range(12,20):
                    if self.encoder[uid][k] != 0:
                        self.encoder[uid][k] = -1
                        self.encoder_names[uid][k] = "none"
                        for kk in self.encoder_nums[uid][k]:
                            if kk in self.encoder_mask[uid]:
                                self.encoder_mask[uid].remove(kk)
                if self.encoder[uid][24] != 0:
                    self.encoder[uid][24] = -1
                    self.encoder_names[uid][24] = "none"
                    for kk in self.encoder_nums[uid][24]:
                        if kk in self.encoder_mask[uid]:
                            self.encoder_mask[uid].remove(kk)

            if self.building_info[uid]['Annual_DHW_demand (kWh)'] == 0 and self.encoder[uid][26] != 0:
                self.encoder[uid][26] = -1
                self.encoder_names[uid][26] = "none"
                for kk in self.encoder_nums[uid][26]:
                    if kk in self.encoder_mask[uid]:
                        self.encoder_mask[uid].remove(kk)
            if self.building_info[uid]['Annual_cooling_demand (kWh)'] == 0 and self.encoder[uid][25] != 0:
                self.encoder[uid][25] = -1
                self.encoder_names[uid][25] = "none"
                for kk in self.encoder_nums[uid][25]:
                    if kk in self.encoder_mask[uid]:
                        self.encoder_mask[uid].remove(kk)
            if self.building_info[uid]['Annual_nonshiftable_electrical_demand (kWh)'] == 0 and self.encoder[uid][23] != 0:
                self.encoder[uid][23] = -1
                self.encoder_names[uid][23] = "none"
                for kk in self.encoder_nums[uid][23]:
                    if kk in self.encoder_mask[uid]:
                        self.encoder_mask[uid].remove(kk)

            #if uid == "Building_3" or uid == "Building_2":
            #    print("="*5)
            #    print(uid)
            #    print('uid ', self.encoder[uid])
            #    print('names ', self.encoder_names[uid])
            #    print('mask ', self.encoder_mask[uid])

            self.encoder_names[uid] = self.encoder_names[uid][self.encoder[uid]!=0]
            self.encoder[uid] = self.encoder[uid][self.encoder[uid]!=0]
            self.encoder_names[uid][self.encoder[uid]==-1] = "removed"

            self.encoder[uid][self.encoder[uid]==-1] = remove_feature()
            self.encoder_mask[uid] = np.array(self.encoder_mask[uid]).astype(np.int)
            self.max_state_dim = max(self.max_state_dim, max(self.encoder_mask[uid])+1)
        
        self.max_state_dim += self.n_agents # append the agent encoding
        self.building_ids = building_ids

        obs_space_low = np.zeros((self.n_agents, self.max_state_dim))
        obs_space_high = np.zeros((self.n_agents, self.max_state_dim))

        for i, uid in enumerate(building_ids):
            j1 = 0; j2 = 0
            for sp in self.encoder[uid]:
                if isinstance(sp, periodic_normalization):
                    n_sp = 2
                elif isinstance(sp, remove_feature):
                    j2 += 1
                    continue
                elif isinstance(sp, onehot_encoding):
                    n_sp = 8
                else:
                    n_sp = 1
                obs_space_low[i, j1:j1+n_sp] = self.original_observation_space[i].low[j2]
                obs_space_high[i, j1:j1+n_sp] = self.original_observation_space[i].high[j2]
                j1 += n_sp
                j2 += 1
            obs_space_high[i, -self.n_agents:] = 1.

        self.observation_space = tuple([Box(obs_space_low[a], obs_space_high[a])
                                        for a in range(self.n_agents)])

        self.raw_state = self.env.reset()
        self.state = self.convert_state(self.raw_state)

    def convert_state(self, raw_states):
        states = []
        for i in range(self.n_agents):
            uid = self.building_ids[i]
            state = raw_states[i]
            state_ = np.array([j for j in np.hstack(self.encoder[uid]*state) if j != None])
            empty = np.zeros(self.max_state_dim)
            try:
                empty[self.encoder_mask[uid]] = state_.copy()
                empty[-self.n_agents:] = self.agent_embedding[i]
            except:
                import pdb; pdb.set_trace()
            states.append(empty)
        states = np.array(states)
        return states

    def step(self, actions):
        """ Returns reward, terminated, info """
        original_actions = [actions[i][self.action_mask[i]]*0.5 for i in range(self.n_agents)]

        self.raw_state, reward, done, _ = self.env.step(original_actions)
        self.state = self.convert_state(self.raw_state)

        reward = (sum(reward)+500000) / 1000000.

        self.t += 1
        info = {}
        if done:
            info['episode_limit'] = False
            info['cost'] = self.env.cost()["total"]
        return reward, done, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.state[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return int(self.max_state_dim)

    def get_state(self):
        return np.array(self.state).reshape(-1)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return int(np.array(self.state).reshape(-1).shape[0])

    def get_avail_actions(self): # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(shape=(self.n_actions,))

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions # CAREFUL! - for continuous dims, this is action space dim rather

    def reset(self):
        """ Returns initial observations and states"""
        self.t = 0
        state = self.env.reset()
        self.state = self.convert_state(state)
        self.n_episode += 1
        pass

    def render(self):
        return

    def close(self):
        self.env.close()

    def seed(self):
        return

    def save_replay(self):
        return

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False}
        return env_info
