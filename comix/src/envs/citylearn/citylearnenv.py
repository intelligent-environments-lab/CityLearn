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
from gym.spaces import Box
from joblib import dump, load

from envs.multiagentenv import MultiAgentEnv

class CityLearnEnv(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
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

        self.n_actions = 2

        self.actions_spaces_low = np.zeros((self.n_agents, 2))
        self.actions_spaces_high = np.zeros((self.n_agents, 2)) + 1e-4
        self.actions_mask = np.zeros((self.n_agents, 2), dtype=bool)

        for i, act_sp in enumerate(self.actions_spaces):
            low = act_sp.low
            high = act_sp.high
            x = len(low) # TODO this should be map instead
            self.actions_spaces_low[i][:x] = low
            self.actions_spaces_high[i][:x] = high
            self.actions_mask[i][:x] = True

        self.action_space = tuple([Box(self.actions_spaces_low[a],
                                      self.actions_spaces_high[a])
                                  for a in range(self.n_agents)])

        # Provides information on Building type, Climate Zone, Annual DHW demand,
        # Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and
        # correllations among buildings
        self.building_info = self.env.get_building_information()
        self.episode_limit = 8759
        self.n_episode = 0

        self.costs = {}

        state = self.env.reset()
        self.state = self._normalize_state(state)

    def step(self, actions):
        """ Returns reward, terminated, info """
        original_actions = [actions[i][self.actions_mask[i]] \
                for i in range(self.n_agents)]
        state, reward, done, _ = self.env.step(original_actions)
        reward = sum(reward) / 100000
        self.state = self._normalize_state(state)
        self.t += 1
        info = {}
        if done:
            if self.t < self.episode_limit:
                info["episode_limit"] = False   # the next state will be masked out
            else:
                info["episode_limit"] = True    # the next state will not be masked out
                if "total" in self.costs:
                    info["cost"] = self.costs["total"][-1]
        return reward, done, info

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
        if self.n_episode > 0:
            cost = self.env.cost()
            for k, v in cost.items():
                if k not in self.costs:
                    self.costs[k] = [v]
                else:
                    self.costs[k].append(v)
        state = self.env.reset()
        self.state = self._normalize_state(state)
        self.n_episode += 1
        pass
        #return self.get_obs(), self.get_state()

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
