import importlib
import logging
import os
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union
import pandas as pd
from citylearn.building import Building
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import Agent
from citylearn.data import EnergySimulation, CarbonIntensity, Pricing, Weather
from citylearn.utilities import read_json

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True

class Simulator:
    def __init__(self, citylearn_env: CityLearnEnv, agents: List[Agent], episodes: int = None):
        r"""Initialize `Simulator`.

        Parameters
        ----------
        citylearn_env : CityLearnEnv
            Simulation environment.
        agents : List[Agent]
            Simulation agents for `citylearn_env.buildings` energy storage charging/discharging management.
        episodes : int
            Number of times to simulate until terminal state is reached.
        """

        self.citylearn_env = citylearn_env
        self.agents = agents
        self.episodes = episodes

    @property
    def citylearn_env(self) -> CityLearnEnv:
        """Simulation environment."""

        return self.__citylearn_env

    @property
    def agents(self) -> List[Agent]:
        """Simulation agents for `citylearn_env.buildings` energy storage charging/discharging management."""

        return self.__agents

    @property
    def episodes(self) -> int:
        """Number of times to simulate until terminal state is reached."""

        return self.__episodes

    @citylearn_env.setter
    def citylearn_env(self, citylearn_env: CityLearnEnv):
        self.__citylearn_env = citylearn_env

    @agents.setter
    def agents(self, agents: List[Agent]):
        if self.citylearn_env.central_agent:
            assert len(agents) == 1, 'Only 1 agent is expected when `citylearn_env.central_agent` = True.'
        else:
            assert len(agents) == len(self.citylearn_env.buildings), 'Length of `agents` and `citylearn_env.buildings` must be equal when using `citylearn_env.central_agent` = False.'

        self.__agents = agents

    @episodes.setter
    def episodes(self, episodes: int):
        episodes = 1 if episodes is None else int(episodes)
        assert episodes > 0, ':attr:`episodes` must be >= 0.'
        self.__episodes = episodes

    def simulate(self):
        """traditional simulation.
        
        Runs central or multi agent simulation.
        """

        for episode in range(self.episodes):
            observations_list = self.citylearn_env.reset()

            while not self.citylearn_env.done:
                logging.debug(f'Timestep: {self.citylearn_env.time_step}/{self.citylearn_env.time_steps - 1}, Episode: {episode}')
                actions_list = []

                # select actions
                for agent, observations in zip(self.agents, observations_list):
                    if agent.action_dimension > 0:
                        actions_list.append(agent.select_actions(observations))
                    else:
                        actions_list.append([]) 

                # apply actions to citylearn_env
                next_observations_list, reward_list, _, _ = self.citylearn_env.step(actions_list)

                # update
                for agent, observations, actions, reward, next_observations in zip(self.agents, observations_list, actions_list, reward_list, next_observations_list):
                    if agent.action_dimension > 0:
                        agent.add_to_buffer(observations, actions, reward, next_observations, done = self.citylearn_env.done)
                    else:
                        continue

                observations_list = [o for o in next_observations_list]

    @classmethod
    def load(cls, schema: Union[str, Path, Mapping[str, Any]]) -> Tuple[CityLearnEnv, List[Agent]]:
        """Return `CityLearnEnv` and `Controller` objects as defined by the `schema`.

        Parameters
        ----------
        schema: Union[str, Path, Mapping[str, Any]]
            Filepath to JSON representation or `dict` object of CityLearn schema.
        
        Returns
        -------
        citylearn_env: CityLearnEnv
            Simulation environment.
        agents: List[Agent]
            Simulation agents for `citylearn_env.buildings` energy storage charging/discharging management.
        """

        if not isinstance(schema, dict):
            schema = read_json(schema)
            schema['root_directory'] = os.path.split(schema) if schema['root_directory'] is None else schema['root_directory']
        else:
            schema['root_directory'] = '' if schema['root_directory'] is None else schema['root_directory']

        central_agent = schema['central_agent']
        observations = {s: v for s, v in schema['observations'].items() if v['active']}
        actions = {a: v for a, v in schema['actions'].items() if v['active']}
        shared_observations = [k for k, v in observations.items() if v['shared_in_central_agent']]
        simulation_start_timestep = schema['simulation_start_timestep']
        simulation_end_timestep = schema['simulation_end_timestep']
        timesteps = simulation_end_timestep - simulation_start_timestep
        buildings = ()
        
        for building_name, building_schema in schema['buildings'].items():
            if building_schema['include']:
                # data
                energy_simulation = pd.read_csv(os.path.join(schema['root_directory'],building_schema['energy_simulation'])).iloc[simulation_start_timestep:simulation_end_timestep + 1].copy()
                energy_simulation = EnergySimulation(*energy_simulation.values.T)
                weather = pd.read_csv(os.path.join(schema['root_directory'],building_schema['weather'])).iloc[simulation_start_timestep:simulation_end_timestep + 1].copy()
                weather = Weather(*weather.values.T)

                if building_schema.get('carbon_intensity', None) is not None:
                    carbon_intensity = pd.read_csv(os.path.join(schema['root_directory'],building_schema['carbon_intensity'])).iloc[simulation_start_timestep:simulation_end_timestep + 1].copy()
                    carbon_intensity = carbon_intensity['kg_CO2/kWh'].tolist()
                    carbon_intensity = CarbonIntensity(carbon_intensity)
                else:
                    carbon_intensity = None

                if building_schema.get('pricing', None) is not None:
                    pricing = pd.read_csv(os.path.join(schema['root_directory'],building_schema['pricing'])).iloc[simulation_start_timestep:simulation_end_timestep + 1].copy()
                    pricing = Pricing(*pricing.values.T)
                else:
                    pricing = None
                    
                # observation and action metadata
                inactive_observations = [] if building_schema.get('inactive_observations', None) is None else building_schema['inactive_observations']
                inactive_actions = [] if building_schema.get('inactive_actions', None) is None else building_schema['inactive_actions']
                observation_metadata = {s: False if s in inactive_observations else True for s in observations}
                action_metadata = {a: False if a in inactive_actions else True for a in actions}

                # construct building
                building = Building(energy_simulation, weather, observation_metadata, action_metadata, carbon_intensity=carbon_intensity, pricing=pricing, name=building_name)

                # update devices
                device_metadata = {
                    'dhw_storage': {'autosizer': building.autosize_dhw_storage},  
                    'cooling_storage': {'autosizer': building.autosize_cooling_storage}, 
                    'heating_storage': {'autosizer': building.autosize_heating_storage}, 
                    'electrical_storage': {'autosizer': building.autosize_electrical_storage}, 
                    'cooling_device': {'autosizer': building.autosize_cooling_device}, 
                    'heating_device': {'autosizer': building.autosize_heating_device}, 
                    'dhw_device': {'autosizer': building.autosize_dhw_device}, 
                    'pv': {'autosizer': building.autosize_pv}
                }

                for name in device_metadata:
                    if building_schema.get(name, None) is None:
                        device = None
                    else:
                        device_type = building_schema[name]['type']
                        device_module = '.'.join(device_type.split('.')[0:-1])
                        device_name = device_type.split('.')[-1]
                        constructor = getattr(importlib.import_module(device_module),device_name)
                        attributes = building_schema[name].get('attributes',{})
                        device = constructor(**attributes)
                        autosize = False if building_schema[name].get('autosize', None) is None else building_schema[name]['autosize']
                        building.__setattr__(name, device)

                        if autosize:
                            autosizer = device_metadata[name]['autosizer']
                            autosize_kwargs = {} if building_schema[name].get('autosize_attributes', None) is None else building_schema[name]['autosize_attributes']
                            autosizer(**autosize_kwargs)
                        else:
                            pass
                
                building.observation_space = building.estimate_observation_space()
                building.action_space = building.estimate_action_space()
                buildings += (building,)
                
            else:
                continue

        reward_function_type = schema['reward_function']
        reward_function_module = '.'.join(reward_function_type.split('.')[0:-1])
        reward_function_name = reward_function_type.split('.')[-1]
        reward_function_constructor = getattr(importlib.import_module(reward_function_module), reward_function_name)
        agent_count = 1 if central_agent else len(buildings)
        reward_function = reward_function_constructor(agent_count=agent_count)
        citylearn_env = CityLearnEnv(list(buildings), timesteps, reward_function, central_agent = central_agent, shared_observations = shared_observations)
        agent_type = schema['agent']['type']
        agent_module = '.'.join(agent_type.split('.')[0:-1])
        agent_name = agent_type.split('.')[-1]
        agent_constructor = getattr(importlib.import_module(agent_module), agent_name)
        agent_attributes = schema['agent'].get('attributes', {})
        agent_attributes = [{
            'building_ids':[b.uid for b in buildings],
            'action_space':citylearn_env.action_space[i],
            'observation_space':citylearn_env.observation_space[i],
            'encoders':citylearn_env.observation_encoders[i],
            **agent_attributes
        }  for i in range(agent_count)]
        agents = [agent_constructor(**agent_attribute) for agent_attribute in agent_attributes]
        episodes = schema['episodes']

        return citylearn_env, agents, episodes