import sys
from .citylearn import CityLearn
from pathlib import Path
import numpy as np                                                                                                                                                                                      
import csv
import time
import os
import re
import pandas as pd
import torch
from joblib import dump, load

class CityLearnEnv(object):
    def __init__(self, seed=0):
        # Load environment
        climate_zone = 1
        parent = Path(__file__).parent.absolute()
        data_path = Path(os.path.join(parent, f"data/Climate_Zone_{climate_zone}"))
        building_attributes = data_path / 'building_attributes.json'
        weather_file = data_path / 'weather_data.csv'
        solar_profile = data_path / 'solar_generation_1kW.csv'
        building_state_actions = os.path.join(parent, "buildings_state_action_space.json")
        building_id = ["Building_1","Building_2","Building_3","Building_4",
                       "Building_5","Building_6","Building_7","Building_8",
                       "Building_9"]
        objective_function = ['ramping','1-load_factor','average_daily_peak',
                              'peak_demand','net_electricity_consumption',
                              'quadratic']
        self.n_agents = len(building_id)
        self.n_actions = 25

        # Contain the lower and upper bounds of the states and actions, to be
        # provided to the agent to normalize the variables between 0 and 1.
        # Can be obtained using observations_spaces[i].low or .high
        self.env = CityLearn(data_path,
                             building_attributes,
                             weather_file,
                             solar_profile,
                             building_id,
                             buildings_states_actions=building_state_actions,
                             cost_function=objective_function,
                             verbose=0,
                             simulation_period=(0,8760-1))

        self.observations_spaces, self.actions_spaces = \
                self.env.get_state_action_spaces()

        self.actions_spaces_low = np.zeros((self.n_agents, 2))
        self.actions_spaces_high = np.zeros((self.n_agents, 2))
        self.actions_mask = np.zeros((self.n_agents, 2))

        for i, act_sp in enumerate(self.actions_spaces):
            low = act_sp.low
            high = act_sp.high
            x = len(low) # TODO this should be map instead
            self.actions_spaces_low[i][:x] = low
            self.actions_spaces_high[i][:x] = low
            self.actions_mask[i][:x] = 1

        self.action_maps = [
        ]
        act_sp = self.actions_spaces
        for x in [-0.2,-0.1,0.,0.1,0.2]:
            for y in [-0.2,-0.1,0.,0.1,0.2]:
                self.action_maps.append(np.array([x,y]))

        # Provides information on Building type, Climate Zone, Annual DHW demand,
        # Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and
        # correllations among buildings
        self.building_info = self.env.get_building_information()
        self.episode_limit = 8759

        state = self.env.reset()
        self.state = self._normalize_state(state)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = self._convert_to_continuous_actions(actions)
        state, reward, done, _ = self.env.step(actions)
        reward = sum(reward) / 10000
        self.state = self._normalize_state(state)
        self.t += 1
        return reward, done, {'episode_limit': self.episode_limit}

    def _convert_to_continuous_actions(self, actions):
        cont_actions = []
        for i, a in enumerate(actions):
            cont = self.action_maps[a]
            tmp = []
            for j, x in enumerate(self.actions_mask[i]):
                if x > 0:
                    tmp.append(cont[j])
            tmp = np.array(tmp)
            cont_actions.append(tmp)
        return cont_actions

    def _normalize_state(self, state):
        normalized = []
        obs_sp = self.observations_spaces
        ns = obs_sp[0].low.shape[0]
        for i, s in enumerate(state):
            x = np.zeros((ns,))
            xx = (s - obs_sp[i].low) / (obs_sp[i].high - obs_sp[i].low+1e-40)
            x[:xx.shape[0]] = xx
            normalized.append(x)
        return normalized

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.state[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.state[0].shape[0]

    def get_state(self):
        return np.array(self.state).reshape(-1)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return sum([len(x) for x in self.state])

    def get_avail_actions(self):
        avails = []
        for i in range(self.n_agents):
            avails.append(self.get_avail_agent_actions(i))
        return avails

    def get_avail_agent_actions(self, agent_id):
        avail = [1] * 25
        act_space = self.actions_spaces
        counter = 0
        for x in self.action_maps:
            flag = True
            for i in range(len(act_space[agent_id].low)):
                if act_space[agent_id].low[i] <= x[i] and x[i] <= act_space[agent_id].high[i]:
                    flag = True
                else:
                    flag = False
                    break
            avail[counter] = 1 if flag else 0
            counter += 1
        return avail

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self.t = 0
        state = self.env.reset()
        self.state = self._normalize_state(state)
        return self.get_obs(), self.get_state()

    def render(self):
        return

    def close(self):
        self.env.close()

    def seed(self):
        return

    def save_replay(self):
        return

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
