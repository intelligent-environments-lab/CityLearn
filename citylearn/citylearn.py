import importlib
import logging
import os
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union
from gym import Env, spaces
import numpy as np
import pandas as pd
from citylearn.base import Environment
from citylearn.building import Building
from citylearn.cost_function import CostFunction
from citylearn.data import DataSet, EnergySimulation, CarbonIntensity, Pricing, Weather
from citylearn.utilities import read_json

LOGGER = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True

class CityLearnEnv(Environment, Env):
    def __init__(self, 
        schema: Union[str, Path, Mapping[str, Any]], root_directory: Union[str, Path] = None, buildings: List[Building] = None, simulation_start_time_step: int = None, simulation_end_time_step: int = None, 
        reward_function: 'citylearn.reward_function.RewardFunction' = None, central_agent: bool = None, shared_observations: List[str] = None, **kwargs
    ):
        r"""Initialize `CityLearnEnv`.

        Parameters
        ----------
        schema: Union[str, Path, Mapping[str, Any]]
            Name of CityLearn data set, filepath to JSON representation or :code:`dict` object of a CityLearn schema.
            Call :py:meth:`citylearn.data.DataSet.get_names` for list of available CityLearn data sets.
        root_directory: Union[str, Path]
            Absolute path to directory that contains the data files including the schema. If provided, will override :code:`root_directory` definition in schema.
        buildings: List[Building], optional
            Buildings in CityLearn environment. If provided, will override :code:`buildings` definition in schema.
        simulation_start_time_step: int, optional
            Time step to start reading from data files. If provided, will override :code:`simulation_start_time_step` definition in schema.
        end_time_step: int, optional
            Time step to end reading from data files. If provided, will override :code:`simulation_end_time_step` definition in schema.
        reward_function: citylearn.reward_function.RewardFunction, optional
            Reward function class instance. If provided, will override :code:`reward_function` definition in schema.
        central_agent: bool, optional
            Expect 1 central agent to control all buildings. If provided, will override :code:`central` definition in schema.
        shared_observations: List[str], optional
            Names of common observations across all buildings i.e. observations that have the same value irrespective of the building.
            If provided, will override :code:`observations:<observation>:shared_in_central_agent` definitions in schema.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super classes.
        """

        self.schema = schema
        self.__rewards = None
        self.root_directory, self.buildings, self.simulation_start_time_step, self.simulation_end_time_step, self.seconds_per_time_step,\
            self.reward_function, self.central_agent, self.shared_observations = self.__load(
                root_directory=root_directory,
                buildings=buildings,
                simulation_start_time_step=simulation_start_time_step,
                simulation_end_time_step=simulation_end_time_step,
                reward_function=reward_function,
                central_agent=central_agent,
                shared_observations=shared_observations,
            )
        super().__init__(**kwargs)

    @property
    def schema(self) -> Union[str, Path, Mapping[str, Any]]:
        """Filepath to JSON representation or `dict` object of CityLearn schema."""

        return self.__schema

    @property
    def root_directory(self) -> Union[str, Path]:
        """Absolute path to directory that contains the data files including the schema."""

        return self.__root_directory

    @property
    def buildings(self) -> List[Building]:
        """Buildings in CityLearn environment."""

        return self.__buildings

    @property
    def simulation_start_time_step(self) -> int:
        """Time step to start reading from data files."""

        return self.__simulation_start_time_step

    @property
    def simulation_end_time_step(self) -> int:
        """Time step to end reading from data files."""

        return self.__simulation_end_time_step

    @property
    def time_steps(self) -> int:
        """Number of simulation time steps."""

        return (self.simulation_end_time_step - self.simulation_start_time_step) + 1

    @property
    def reward_function(self) -> 'citylearn.reward_function.RewardFunction':
        """Reward function class instance."""

        return self.__reward_function

    @property
    def rewards(self) -> List[List[float]]:
        """Reward time series"""

        return self.__rewards

    @property
    def central_agent(self) -> bool:
        """Expect 1 central agent to control all buildings."""

        return self.__central_agent

    @property
    def shared_observations(self) -> List[str]:
        """Names of common observations across all buildings i.e. observations that have the same value irrespective of the building."""

        return self.__shared_observations

    @property
    def done(self) -> bool:
        """Check if simulation has reached completion."""

        return self.time_step == self.time_steps - 1

    @property
    def observation_space(self) -> List[spaces.Box]:
        """Controller(s) observation spaces.

        Returns
        -------
        observation_space : List[spaces.Box]
            List of agent(s) observation spaces.
        
        Notes
        -----
        If `central_agent` is True, a list of 1 `spaces.Box` object is returned that contains all buildings' limits with the limits in the same order as `buildings`. 
        The `shared_observations` limits are only included in the first building's limits. If `central_agent` is False, a list of `space.Box` objects as
        many as `buildings` is returned in the same order as `buildings`.
        """

        if self.central_agent:
            low_limit = [
                v for i, b in enumerate(self.buildings) for v, s in zip(b.observation_space.low, b.active_observations) 
                if i == 0 or s not in self.shared_observations
            ]
            high_limit = [
                v for i, b in enumerate(self.buildings) for v, s in zip(b.observation_space.high, b.active_observations) 
                if i == 0 or s not in self.shared_observations
            ]
            observation_space = [spaces.Box(low=np.array(low_limit), high=np.array(high_limit), dtype=np.float32)]
        else:
            observation_space = [b.observation_space for b in self.buildings]
        
        return observation_space

    @property
    def action_space(self) -> List[spaces.Box]:
        """Controller(s) action spaces.

        Returns
        -------
        action_space : List[spaces.Box]
            List of agent(s) action spaces.
        
        Notes
        -----
        If `central_agent` is True, a list of 1 `spaces.Box` object is returned that contains all buildings' limits with the limits in the same order as `buildings`. 
        If `central_agent` is False, a list of `space.Box` objects as many as `buildings` is returned in the same order as `buildings`.
        """

        if self.central_agent:
            low_limit = [v for b in self.buildings for v in b.action_space.low]
            high_limit = [v for b in self.buildings for v in b.action_space.high]
            action_space = [spaces.Box(low=np.array(low_limit), high=np.array(high_limit), dtype=np.float32)]
        else:
            action_space = [b.action_space for b in self.buildings]
        
        return action_space

    @property
    def observations(self) -> List[List[float]]:
        """Observations at current time step.
        
        Notes
        -----
        If `central_agent` is True, a list of 1 sublist containing all building observation values is returned in the same order as `buildings`. 
        The `shared_observations` values are only included in the first building's observation values. If `central_agent` is False, a list of sublists 
        is returned where each sublist is a list of 1 building's observation values and the sublist in the same order as `buildings`.
        """

        return [[
            v for i, b in enumerate(self.buildings) for k, v in b.observations().items() if i == 0 or k not in self.shared_observations
        ]] if self.central_agent else [list(b.observations().values()) for b in self.buildings]


    @property
    def observation_names(self) -> List[List[str]]:
        """Names of returned observations.

        Notes
        -----
        If `central_agent` is True, a list of 1 sublist containing all building observation names is returned in the same order as `buildings`. 
        The `shared_observations` names are only included in the first building's observation names. If `central_agent` is False, a list of sublists 
        is returned where each sublist is a list of 1 building's observation names and the sublist in the same order as `buildings`.
        """

        return [[
            k for i, b in enumerate(self.buildings) for k, v in b.observations().items() if i == 0 or k not in self.shared_observations
        ]] if self.central_agent else [list(b.observations().keys()) for b in self.buildings]

    @property
    def net_electricity_consumption_without_storage_and_pv_emission(self) -> np.ndarray:
        """Summed `Building.net_electricity_consumption_without_storage_and_pv_emission` time series, in [kg_co2]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_and_pv_emission for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def net_electricity_consumption_without_storage_and_pv_cost(self) -> np.ndarray:
        """Summed `Building.net_electricity_consumption_without_storage_and_pv_cost` time series, in [$]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_and_pv_cost for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def net_electricity_consumption_without_storage_and_pv(self) -> np.ndarray:
        """Summed `Building.net_electricity_consumption_without_storage_and_pv` time series, in [kWh]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_and_pv for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def net_electricity_consumption_without_storage_emission(self) -> np.ndarray:
        """Summed `Building.net_electricity_consumption_without_storage_emission` time series, in [kg_co2]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_emission for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_without_storage_cost(self) -> np.ndarray:
        """Summed `Building.net_electricity_consumption_without_storage_cost` time series, in [$]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_cost for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def net_electricity_consumption_without_storage(self) -> np.ndarray:
        """Summed `Building.net_electricity_consumption_without_storage` time series, in [kWh]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def net_electricity_consumption_emission(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_emission` time series, in [kg_co2]."""

        return self.__net_electricity_consumption_emission

    @property
    def net_electricity_consumption_cost(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_cost` time series, in [$]."""

        return self.__net_electricity_consumption_cost

    @property
    def net_electricity_consumption(self) -> List[float]:
        """Summed `Building.net_electricity_consumption` time series, in [kWh]."""

        return self.__net_electricity_consumption

    @property
    def cooling_electricity_consumption(self) -> np.ndarray:
        """Summed `Building.cooling_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.cooling_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def heating_electricity_consumption(self) -> np.ndarray:
        """Summed `Building.heating_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.heating_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def dhw_electricity_consumption(self) -> np.ndarray:
        """Summed `Building.dhw_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.dhw_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def cooling_storage_electricity_consumption(self) -> np.ndarray:
        """Summed `Building.cooling_storage_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.cooling_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def heating_storage_electricity_consumption(self) -> np.ndarray:
        """Summed `Building.heating_storage_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.heating_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def dhw_storage_electricity_consumption(self) -> np.ndarray:
        """Summed `Building.dhw_storage_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.dhw_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def electrical_storage_electricity_consumption(self) -> np.ndarray:
        """Summed `Building.electrical_storage_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.electrical_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_cooling_device_to_cooling_storage(self) -> np.ndarray:
        """Summed `Building.energy_from_cooling_device_to_cooling_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_cooling_device_to_cooling_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_heating_device_to_heating_storage(self) -> np.ndarray:
        """Summed `Building.energy_from_heating_device_to_heating_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_heating_device_to_heating_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_dhw_device_to_dhw_storage(self) -> np.ndarray:
        """Summed `Building.energy_from_dhw_device_to_dhw_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_dhw_device_to_dhw_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_to_electrical_storage(self) -> np.ndarray:
        """Summed `Building.energy_to_electrical_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_to_electrical_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_cooling_device(self) -> np.ndarray:
        """Summed `Building.energy_from_cooling_device` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_cooling_device for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_heating_device(self) -> np.ndarray:
        """Summed `Building.energy_from_heating_device` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_heating_device for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_dhw_device(self) -> np.ndarray:
        """Summed `Building.energy_from_dhw_device` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_dhw_device for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_cooling_storage(self) -> np.ndarray:
        """Summed `Building.energy_from_cooling_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_cooling_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_heating_storage(self) -> np.ndarray:
        """Summed `Building.energy_from_heating_storage` time series, in [kWh]."""
        
        return pd.DataFrame([b.energy_from_heating_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_dhw_storage(self) -> np.ndarray:
        """Summed `Building.energy_from_dhw_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_dhw_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def energy_from_electrical_storage(self) -> np.ndarray:
        """Summed `Building.energy_from_electrical_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_electrical_storage for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def cooling_demand(self) -> np.ndarray:
        """Summed `Building.cooling_demand`, in [kWh]."""

        return pd.DataFrame([b.cooling_demand for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def heating_demand(self) -> np.ndarray:
        """Summed `Building.heating_demand`, in [kWh]."""

        return pd.DataFrame([b.heating_demand for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def dhw_demand(self) -> np.ndarray:
        """Summed `Building.dhw_demand`, in [kWh]."""

        return pd.DataFrame([b.dhw_demand for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def non_shiftable_load_demand(self) -> np.ndarray:
        """Summed `Building.non_shiftable_load_demand`, in [kWh]."""

        return pd.DataFrame([b.non_shiftable_load_demand for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @property
    def solar_generation(self) -> np.ndarray:
        """Summed `Building.solar_generation, in [kWh]`."""

        return pd.DataFrame([b.solar_generation for b in self.buildings]).sum(axis = 0, min_count = 1).to_numpy()

    @schema.setter
    def schema(self, schema: Union[str, Path, Mapping[str, Any]]):
        self.__schema = schema

    @root_directory.setter
    def root_directory(self, root_directory: Union[str, Path]):
        self.__root_directory = root_directory

    @buildings.setter
    def buildings(self, buildings: List[Building]):
        self.__buildings = buildings

    @simulation_start_time_step.setter
    def simulation_start_time_step(self, simulation_start_time_step: int):
        assert simulation_start_time_step >= 0, 'simulation_start_time_step must be >= 0'
        self.__simulation_start_time_step = simulation_start_time_step

    @simulation_end_time_step.setter
    def simulation_end_time_step(self, simulation_end_time_step: int):
        assert simulation_end_time_step >= 0, 'simulation_end_time_step must be >= 0'
        self.__simulation_end_time_step = simulation_end_time_step

    @reward_function.setter
    def reward_function(self, reward_function: 'citylearn.reward_function.RewardFunction'):
        self.__reward_function = reward_function

    @central_agent.setter
    def central_agent(self, central_agent: bool):
        self.__central_agent = central_agent

    @shared_observations.setter
    def shared_observations(self, shared_observations: List[str]):
        self.__shared_observations = self.get_default_shared_observations() if shared_observations is None else shared_observations

    @staticmethod
    def get_default_shared_observations() -> List[str]:
        """Names of default common observations across all buildings i.e. observations that have the same value irrespective of the building.
        
        Notes
        -----
        May be used to assigned :attr:`shared_observations` value during `CityLearnEnv` object initialization.
        """

        return [
            'month', 'day_type', 'hour', 'daylight_savings_status',
            'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',
            'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h',
            'outdoor_relative_humidity', 'outdoor_relative_humidity_predicted_6h',
            'outdoor_relative_humidity_predicted_12h', 'outdoor_relative_humidity_predicted_24h',
            'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h',
            'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h',
            'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h',
            'direct_solar_irradiance_predicted_12h', 'direct_solar_irradiance_predicted_24h',
            'carbon_intensity',
        ]


    def step(self, actions: List[List[float]]) -> Tuple[List[List[float]], List[float], bool, dict]:
        """Apply actions to `buildings` and advance to next time step.
        
        Parameters
        ----------
        actions: List[List[float]]
            Fractions of `buildings` storage devices' capacities to charge/discharge by. 
            If `central_agent` is True, `actions` parameter should be a list of 1 list containing all buildings' actions and follows
            the ordering of buildings in `buildings`. If `central_agent` is False, `actions` parameter should be a list of sublists
            where each sublists contains the actions for each building in `buildings`  and follows the ordering of buildings in `buildings`.

        Returns
        -------
        observations: List[List[float]]
            :attr:`observations` current value.
        reward: List[float] 
            :meth:`get_reward` current value.
        done: bool 
            A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
            A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
            a certain timelimit was exceeded, or the physics simulation has entered an invalid observation.
        info: dict
            A dictionary that may contain additional information regarding the reason for a ``done`` signal.
            `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
            Override :meth"`get_info` to get custom key-value pairs in `info`.
        """

        actions = self.__parse_actions(actions)

        for building, building_actions in zip(self.buildings, actions):
            building.apply_actions(**building_actions)

        self.next_time_step()
        reward = self.reward_function.calculate()
        self.__rewards.append(reward)
        return self.observations, reward, self.done, self.get_info()

    def get_info(self) -> Mapping[Any, Any]:
        return {}

    def __parse_actions(self, actions: List[List[float]]) -> List[Mapping[str, float]]:
        """Return mapping of action name to action value for each building."""

        actions = list(actions)
        building_actions = []

        if self.central_agent:
            actions = actions[0]
            
            for building in self.buildings:
                size = building.action_space.shape[0]
                building_actions.append(actions[0:size])
                actions = actions[size:]

        else:
            building_actions = [list(a) for a in actions]

        active_actions = [[k for k, v in b.action_metadata.items() if v] for b in self.buildings]
        actions = [{k:a for k, a in zip(active_actions[i],building_actions[i])} for i in range(len(active_actions))]
        actions = [{f'{k}_action':actions[i].get(k, 0.0) for k in b.action_metadata} for i, b in enumerate(self.buildings)]
        return actions
    
    def get_building_information(self) -> Tuple[Mapping[str, Any]]:
        """Get buildings PV capacity, end-use annual demands, and correlations with other buildings end-use annual demands.

        Returns
        -------
        building_information: List[Mapping[str, Any]]
            Building information summary.
        """

        np.seterr(divide='ignore', invalid='ignore')
        building_info = ()
        n_years = max(1, (self.time_steps*self.seconds_per_time_step)/(8760*3600))

        for building in self.buildings:
            building_dict = {}
            building_dict['solar_power'] = round(building.pv.nominal_power, 3)
            building_dict['annual_dhw_demand'] = round(sum(building.energy_simulation.dhw_demand)/n_years, 3)
            building_dict['annual_cooling_demand'] = round(sum(building.energy_simulation.cooling_demand)/n_years, 3)
            building_dict['annual_heating_demand'] = round(sum(building.energy_simulation.heating_demand)/n_years, 3)
            building_dict['annual_nonshiftable_electrical_demand'] = round(sum(building.energy_simulation.non_shiftable_load)/n_years, 3)
            building_dict['dhw_storage_capacity'] = building.dhw_storage.capacity
            building_dict['cooling_storage_capacity'] = building.cooling_storage.capacity
            building_dict['heating_storage_capacity'] = building.heating_storage.capacity
            building_dict['electrical_storage_capacity'] = building.electrical_storage.capacity
            building_dict['correlations_dhw'] = ()
            building_dict['correlations_cooling_demand'] = ()
            building_dict['correlations_heating_demand'] = ()
            building_dict['correlations_non_shiftable_load'] = ()
            
            for corr_building in self.buildings:
                building_dict['correlations_dhw'] += (round((np.corrcoef(
                    np.array(building.energy_simulation.dhw_demand), np.array(corr_building.energy_simulation.dhw_demand)
                ))[0][1], 3),)
                building_dict['correlations_cooling_demand'] += (round((np.corrcoef(
                    np.array(building.energy_simulation.cooling_demand), np.array(corr_building.energy_simulation.cooling_demand)
                ))[0][1], 3),)
                building_dict['correlations_heating_demand'] += (round((np.corrcoef(
                    np.array(building.energy_simulation.heating_demand), np.array(corr_building.energy_simulation.heating_demand)
                ))[0][1], 3),)
                building_dict['correlations_non_shiftable_load'] += (round((np.corrcoef(
                    np.array(building.energy_simulation.non_shiftable_load), np.array(corr_building.energy_simulation.non_shiftable_load)
                ))[0][1], 3),)

            building_info += (building_dict ,)
        
        return building_info
    
    def evaluate(self) -> pd.DataFrame:
        r"""Evaluate cost functions at current time step.

        Calculates and returns building-level and district-level cost functions normalized w.r.t. the no control scenario.
        
        Returns
        -------
        cost_functions: pd.DataFrame
            Cost function summary.

        Notes
        -----
        The equation for the returned cost function values is :math:`\frac{C_{\textrm{control}}}{C_{\textrm{no control}}}` 
        where :math:`C_{\textrm{control}}` is the value when the agent(s) control the environment and :math:`C_{\textrm{no control}}`
        is the value when none of the flexible distributed energy resources in the environment are actively in use and controlled.
        """

        building_level = []

        for b in self.buildings:
            building_level += [{
                'name': b.name,
                'cost_function': 'electricity_consumption',
                'value': CostFunction.electricity_consumption(b.net_electricity_consumption)[-1]/\
                    CostFunction.electricity_consumption(b.net_electricity_consumption_without_storage)[-1],
                }, {
                'name': b.name,
                'cost_function': 'zero_net_energy',
                'value': CostFunction.zero_net_energy(b.net_electricity_consumption)[-1]/\
                    CostFunction.zero_net_energy(b.net_electricity_consumption_without_storage)[-1],
                }, {
                'name': b.name,
                'cost_function': 'carbon_emissions',
                'value': CostFunction.carbon_emissions(b.net_electricity_consumption_emission)[-1]/\
                    CostFunction.carbon_emissions(b.net_electricity_consumption_without_storage_emission)[-1]\
                        if sum(b.carbon_intensity.carbon_intensity) != 0 else None,
                }, {
                'name': b.name,
                'cost_function': 'cost',
                'value': CostFunction.cost(b.net_electricity_consumption_cost)[-1]/\
                    CostFunction.cost(b.net_electricity_consumption_without_storage_cost)[-1]\
                        if sum(b.pricing.electricity_pricing) != 0 else None,
                }]

        building_level = pd.DataFrame(building_level)
        building_level['level'] = 'building'

        ## district level
        district_level = pd.DataFrame([{
            'cost_function': 'ramping',
            'value': CostFunction.ramping(self.net_electricity_consumption)[-1]/\
                CostFunction.ramping(self.net_electricity_consumption_without_storage)[-1],
            }, {
            'cost_function': '1 - load_factor',
            'value': CostFunction.load_factor(self.net_electricity_consumption)[-1]/\
                CostFunction.load_factor(self.net_electricity_consumption_without_storage)[-1],
            }, {
            'cost_function': 'average_daily_peak',
            'value': CostFunction.average_daily_peak(self.net_electricity_consumption)[-1]/\
                CostFunction.average_daily_peak(self.net_electricity_consumption_without_storage)[-1],
            }, {
            'cost_function': 'peak_demand',
            'value': CostFunction.peak_demand(self.net_electricity_consumption)[-1]/\
                CostFunction.peak_demand(self.net_electricity_consumption_without_storage)[-1],
            }])

        district_level = pd.concat([district_level, building_level], ignore_index=True, sort=False)
        district_level = district_level.groupby(['cost_function'])[['value']].mean().reset_index()
        district_level['name'] = 'District'
        district_level['level'] = 'district'
        cost_functions = pd.concat([district_level, building_level], ignore_index=True, sort=False)

        return cost_functions

    def next_time_step(self):
        r"""Advance all buildings to next `time_step`."""

        for building in self.buildings:
            building.next_time_step()
        
        super().next_time_step()
        self.update_variables()

    def reset(self) -> List[List[float]]:
        r"""Reset `CityLearnEnv` to initial state.
        
        Returns
        -------
        observations: List[List[float]]
            :attr:`observations`. 
        """

        # object reset
        super().reset()

        for building in self.buildings:
            building.reset()

        # variable reset
        self.__rewards = [[]]
        self.__net_electricity_consumption = []
        self.__net_electricity_consumption_cost = []
        self.__net_electricity_consumption_emission = []
        self.update_variables()

        return self.observations

    def update_variables(self):
        # net electricity consumption
        self.__net_electricity_consumption.append(sum([b.net_electricity_consumption[self.time_step] for b in self.buildings]))

        # net electriciy consumption cost
        self.__net_electricity_consumption_cost.append(sum([b.net_electricity_consumption_cost[self.time_step] for b in self.buildings]))

        # net electriciy consumption emission
        self.__net_electricity_consumption_emission.append(sum([b.net_electricity_consumption_emission[self.time_step] for b in self.buildings]))

    def load_agent(self) -> 'citylearn.agents.base.Agent':
        """Return :class:`Agent` or sub class object as defined by the `schema`.

        Parameters
        ----------
        **kwargs : dict
            Parameters to override schema definitions. See :py:class:`citylearn.citylearn.CityLearnEnv` initialization parameters for valid kwargs.
        
        Returns
        -------
        agents: Agent
            Simulation agent(s) for `citylearn_env.buildings` energy storage charging/discharging management.
        """

        agent_type = self.schema['agent']['type']
        agent_module = '.'.join(agent_type.split('.')[0:-1])
        agent_name = agent_type.split('.')[-1]
        agent_constructor = getattr(importlib.import_module(agent_module), agent_name)
        agent_attributes = self.schema['agent'].get('attributes', {})
        agent_attributes = {'env': self, **agent_attributes}
        agent = agent_constructor(**agent_attributes)
        return agent

    def __load(self, **kwargs) -> Tuple[List[Building], int, float, 'citylearn.reward_function.RewardFunction', bool, List[str]]:
        """Return `CityLearnEnv` and `Controller` objects as defined by the `schema`.
        
        Returns
        -------
        buildings : List[Building]
            Buildings in CityLearn environment.
        time_steps : int
            Number of simulation time steps.
        seconds_per_time_step: float
            Number of seconds in 1 `time_step` and must be set to >= 1.
        reward_function : citylearn.reward_function.RewardFunction
            Reward function class instance.
        central_agent : bool, optional
            Expect 1 central agent to control all building storage device.
        shared_observations : List[str], optional
            Names of common observations across all buildings i.e. observations that have the same value irrespective of the building.
        """
        
        if isinstance(self.schema, (str, Path)) and os.path.isfile(self.schema):
            schema_filepath = Path(self.schema) if isinstance(self.schema, str) else self.schema
            self.schema = read_json(self.schema)
            self.schema['root_directory'] = os.path.split(schema_filepath.absolute())[0] if self.schema['root_directory'] is None\
                else self.schema['root_directory']
        elif isinstance(self.schema, str) and self.schema in DataSet.get_names():
            self.schema = DataSet.get_schema(self.schema)
            self.schema['root_directory'] = '' if self.schema['root_directory'] is None else self.schema['root_directory']
        elif isinstance(self.schema, dict):
            self.schema['root_directory'] = '' if self.schema['root_directory'] is None else self.schema['root_directory']
        else:
            raise UnknownSchemaError()

        root_directory = kwargs['root_directory'] if kwargs.get('root_directory') is not None else self.schema['root_directory']
        central_agent =  kwargs['central_agent'] if kwargs.get('central_agent') is not None else self.schema['central_agent']
        observations = {s: v for s, v in self.schema['observations'].items() if v['active']}
        actions = {a: v for a, v in self.schema['actions'].items() if v['active']}
        shared_observations =  kwargs['shared_observations'] if kwargs.get('shared_observations') is not None else\
            [k for k, v in observations.items() if v['shared_in_central_agent']]
        simulation_start_time_step = kwargs['simulation_start_time_step'] if kwargs.get('simulation_start_time_step') is not None else\
            self.schema['simulation_start_time_step']
        simulation_end_time_step = kwargs['simulation_end_time_step'] if kwargs.get('simulation_end_time_step') is not None else\
            self.schema['simulation_end_time_step']
        seconds_per_time_step = self.schema['seconds_per_time_step']

        if kwargs.get('buildings') is not None and len(kwargs['buildings']) > 0:
            buildings = kwargs['buildings']
        
        else:
            buildings = ()
            
            for building_name, building_schema in self.schema['buildings'].items():
                if building_schema['include']:
                    # data
                    energy_simulation = pd.read_csv(os.path.join(root_directory,building_schema['energy_simulation'])).iloc[simulation_start_time_step:simulation_end_time_step + 1].copy()
                    energy_simulation = EnergySimulation(*energy_simulation.values.T)
                    weather = pd.read_csv(os.path.join(root_directory,building_schema['weather'])).iloc[simulation_start_time_step:simulation_end_time_step + 1].copy()
                    weather = Weather(*weather.values.T)

                    if building_schema.get('carbon_intensity', None) is not None:
                        carbon_intensity = pd.read_csv(os.path.join(root_directory,building_schema['carbon_intensity'])).iloc[simulation_start_time_step:simulation_end_time_step + 1].copy()
                        carbon_intensity = carbon_intensity['kg_CO2/kWh'].tolist()
                        carbon_intensity = CarbonIntensity(carbon_intensity)
                    else:
                        carbon_intensity = None

                    if building_schema.get('pricing', None) is not None:
                        pricing = pd.read_csv(os.path.join(root_directory,building_schema['pricing'])).iloc[simulation_start_time_step:simulation_end_time_step + 1].copy()
                        pricing = Pricing(*pricing.values.T)
                    else:
                        pricing = None
                        
                    # observation and action metadata
                    inactive_observations = [] if building_schema.get('inactive_observations', None) is None else building_schema['inactive_observations']
                    inactive_actions = [] if building_schema.get('inactive_actions', None) is None else building_schema['inactive_actions']
                    observation_metadata = {s: False if s in inactive_observations else True for s in observations}
                    action_metadata = {a: False if a in inactive_actions else True for a in actions}

                    # construct building
                    building_type = 'citylearn.citylearn.Building' if building_schema.get('type', None) is None else building_schema['type']
                    building_type_module = '.'.join(building_type.split('.')[0:-1])
                    building_type_name = building_type.split('.')[-1]
                    building_constructor = getattr(importlib.import_module(building_type_module),building_type_name)

                    building: Building = building_constructor(
                        energy_simulation=energy_simulation, 
                        weather=weather, 
                        observation_metadata=observation_metadata, 
                        action_metadata=action_metadata, 
                        carbon_intensity=carbon_intensity, 
                        pricing=pricing,
                        name=building_name, 
                        seconds_per_time_step=seconds_per_time_step,
                    )

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
                            attributes['seconds_per_time_step'] = seconds_per_time_step
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
        
        buildings = list(buildings)

        if kwargs.get('reward_function') is not None:
            reward_function = kwargs['reward_function']
        else:
            reward_function_type = self.schema['reward_function']['type']
            reward_function_attributes = self.schema['reward_function'].get('attributes',None)
            reward_function_attributes = {} if reward_function_attributes is None else reward_function_attributes
            reward_function_module = '.'.join(reward_function_type.split('.')[0:-1])
            reward_function_name = reward_function_type.split('.')[-1]
            reward_function_constructor = getattr(importlib.import_module(reward_function_module), reward_function_name)
            reward_function = reward_function_constructor(self,**reward_function_attributes)

        return root_directory, buildings, simulation_start_time_step, simulation_end_time_step, seconds_per_time_step, reward_function, central_agent, shared_observations
        
class Error(Exception):
    """Base class for other exceptions."""

class UnknownSchemaError(Error):
    """Raised when a schema is not a data set name, dict nor filepath."""
    __MESSAGE = 'Unknown schema parsed into constructor. Schema must be name of CityLearn data set,'\
        ' a filepath to JSON representation or `dict` object of a CityLearn schema.'\
        ' Call citylearn.data.DataSet.get_names() for list of available CityLearn data sets.'
  
    def __init__(self,message=None):
        super().__init__(self.__MESSAGE if message is None else message)