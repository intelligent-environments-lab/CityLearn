import importlib
import logging
import os
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union
from gym import Env, spaces
import numpy as np
import pandas as pd
from citylearn.base import Environment
from citylearn.agents.base import Agent
from citylearn.data import EnergySimulation, CarbonIntensity, Pricing, Weather
from citylearn.energy_model import Battery, ElectricHeater, HeatPump, PV, StorageTank
from citylearn.preprocessing import Encoder, PeriodicNormalization, OnehotEncoding, RemoveFeature, Normalize
from citylearn.reward_function import RewardFunction
from citylearn.utilities import read_json

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True
    
class Building(Environment):
    def __init__(
        self, energy_simulation: EnergySimulation, weather: Weather, observation_metadata: Mapping[str, bool], action_metadata: Mapping[str, bool], carbon_intensity: CarbonIntensity = None, 
        pricing: Pricing = None, dhw_storage: StorageTank = None, cooling_storage: StorageTank = None, heating_storage: StorageTank = None, electrical_storage: Battery = None, 
        dhw_device: Union[HeatPump, ElectricHeater] = None, cooling_device: HeatPump = None, heating_device: Union[HeatPump, ElectricHeater] = None, pv: PV = None, name: str = None, **kwargs
    ):
        r"""Initialize `Building`.

        Parameters
        ----------
        energy_simulation : EnergySimulation
            Temporal features, cooling, heating, dhw and plug loads, solar generation and indoor environment time series.
        weather : Weather
            Outdoor weather conditions and forecasts time sereis.
        observation_metadata : dict
            Mapping of active and inactive observations.
        action_metadata : dict
            Mapping od active and inactive actions.
        carbon_intensity : CarbonIntensity, optional
            Emission rate time series.
        pricing : Pricing, optional
            Energy pricing and forecasts time series.
        dhw_storage : StorageTank, optional
            Hot water storage object for domestic hot water.
        cooling_storage : StorageTank, optional
            Cold water storage object for space cooling.
        heating_storage : StorageTank, optional
            Hot water storage object for space heating.
        electrical_storage : Battery, optional
            Electric storage object for meeting electric loads.
        dhw_device : Union[HeatPump, ElectricHeater], optional
            Electric device for meeting hot domestic hot water demand and charging `dhw_storage`.
        cooling_device : HeatPump, optional
            Electric device for meeting space cooling demand and charging `cooling_storage`.
        heating_device : Union[HeatPump, ElectricHeater], optional
            Electric device for meeting space heating demand and charging `heating_storage`.
        pv : PV, optional
            PV object for offsetting electricity demand from grid.
        name : str, optional
            Unique building name.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize `Environment` super class.
        """

        self.name = name
        self.energy_simulation = energy_simulation
        self.weather = weather
        self.carbon_intensity = carbon_intensity
        self.pricing = pricing
        self.dhw_storage = dhw_storage
        self.cooling_storage = cooling_storage
        self.heating_storage = heating_storage
        self.electrical_storage = electrical_storage
        self.dhw_device = dhw_device
        self.cooling_device = cooling_device
        self.heating_device = heating_device
        self.pv = pv
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.observation_space = self.estimate_observation_space()
        self.action_space = self.estimate_action_space()
        super().__init__(**kwargs)

    @property
    def energy_simulation(self) -> EnergySimulation:
        """Temporal features, cooling, heating, dhw and plug loads, solar generation and indoor environment time series."""

        return self.__energy_simulation

    @property
    def weather(self) -> Weather:
        """Outdoor weather conditions and forecasts time series."""

        return self.__weather

    @property
    def observation_metadata(self) -> Mapping[str, bool]:
        """Mapping of active and inactive observations."""

        return self.__observation_metadata

    @property
    def action_metadata(self) -> Mapping[str, bool]:
        """Mapping od active and inactive actions."""

        return self.__action_metadata

    @property
    def carbon_intensity(self) -> CarbonIntensity:
        """Emission rate time series."""

        return self.__carbon_intensity

    @property
    def pricing(self) -> Pricing:
        """Energy pricing and forecasts time series."""

        return self.__pricing

    @property
    def dhw_storage(self) -> StorageTank:
        """Hot water storage object for domestic hot water."""

        return self.__dhw_storage

    @property
    def cooling_storage(self) -> StorageTank:
        """Cold water storage object for space cooling."""

        return self.__cooling_storage

    @property
    def heating_storage(self) -> StorageTank:
        """Hot water storage object for space heating."""

        return self.__heating_storage

    @property
    def electrical_storage(self) -> Battery:
        """Electric storage object for meeting electric loads."""

        return self.__electrical_storage

    @property
    def dhw_device(self) -> Union[HeatPump, ElectricHeater]:
        """Electric device for meeting hot domestic hot water demand and charging `dhw_storage`."""

        return self.__dhw_device

    @property
    def cooling_device(self) -> HeatPump:
        """Electric device for meeting space cooling demand and charging `cooling_storage`."""

        return self.__cooling_device

    @property
    def heating_device(self) -> Union[HeatPump, ElectricHeater]:
        """Electric device for meeting space heating demand and charging `heating_storage`."""

        return self.__heating_device

    @property
    def pv(self) -> PV:
        """PV object for offsetting electricity demand from grid."""

        return self.__pv

    @property
    def name(self) -> str:
        """Unique building name."""

        return self.__name

    @property
    def observation_space(self) -> spaces.Box:
        """Controller observation space."""

        return self.__observation_space

    @property
    def action_space(self) -> spaces.Box:
        """Controller action spaces."""

        return self.__action_space

    @property
    def observation_encoders(self) -> List[Encoder]:
        r"""Get observation value transformers/encoders for use in agent algorithm.

        The encoder classes are defined in the `preprocessing.py` module and include `PeriodicNormalization` for cyclic observations,
        `OnehotEncoding` for categorical obeservations, `RemoveFeature` for non-applicable observations given available storage systems and devices
        and `Normalize` for observations with known mnimum and maximum boundaries.
        
        Returns
        -------
        encoders : List[Encoder]
            Encoder classes for observations ordered with respect to `active_observations`.
        """

        remove_features = ['net_electricity_consumption']
        remove_features += [
            'solar_generation', 'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h',
            'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h',
            'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h',
            'direct_solar_irradiance_predicted_12h', 'direct_solar_irradiance_predicted_24h',
        ] if self.pv.capacity == 0 else []
        demand_observations = {
            'dhw_storage_soc': np.nansum(self.energy_simulation.dhw_demand),
            'cooling_storage_soc': np.nansum(self.energy_simulation.cooling_demand),
            'heating_storage_soc': np.nansum(self.energy_simulation.heating_demand),
            'electrical_storage_soc': np.nansum(np.nansum([
                list(self.energy_simulation.dhw_demand),
                list(self.energy_simulation.cooling_demand),
                list(self.energy_simulation.heating_demand),
                list(self.energy_simulation.non_shiftable_load)
            ], axis = 0)),
            'non_shiftable_load': np.nansum(self.energy_simulation.non_shiftable_load),
        }
        remove_features += [k for k, v in demand_observations.items() if v == 0]
        remove_features = [f for f in remove_features if f in self.active_observations]
        encoders = []

        for i, observation in enumerate(self.active_observations):
            if observation in ['month', 'hour']:
                encoders.append(PeriodicNormalization(self.observation_space.high[i]))
            
            elif observation == 'day_type':
                encoders.append(OnehotEncoding([0, 1, 2, 3, 4, 5, 6, 7, 8]))
            
            elif observation == "daylight_savings_status":
                encoders.append(OnehotEncoding([0, 1, 2]))
            
            elif observation in remove_features:
                encoders.append(RemoveFeature())
            
            else:
                encoders.append(Normalize(self.observation_space.low[i], self.observation_space.high[i]))

        return encoders

    @property
    def observations(self) -> Mapping[str, float]:
        """Observations at current time step."""

        observations = {}
        data = {
            **{k: v[self.time_step] for k, v in vars(self.energy_simulation).items()},
            **{k: v[self.time_step] for k, v in vars(self.weather).items()},
            **{k: v[self.time_step] for k, v in vars(self.pricing).items()},
            'solar_generation':self.pv.get_generation(self.energy_simulation.solar_generation[self.time_step]),
            **{
                'cooling_storage_soc':self.cooling_storage.soc[-1]/self.cooling_storage.capacity if self.time_step > 0 else self.cooling_storage.initial_soc/self.cooling_storage.capacity,
                'heating_storage_soc':self.heating_storage.soc[-1]/self.heating_storage.capacity if self.time_step > 0 else self.heating_storage.initial_soc/self.heating_storage.capacity,
                'dhw_storage_soc':self.dhw_storage.soc[-1]/self.dhw_storage.capacity if self.time_step > 0 else self.dhw_storage.initial_soc/self.dhw_storage.capacity,
                'electrical_storage_soc':self.electrical_storage.soc[-1]/self.electrical_storage.capacity if self.time_step > 0 else self.electrical_storage.initial_soc/self.electrical_storage.capacity,
            },
            'net_electricity_consumption': self.net_electricity_consumption[-1] if self.time_step > 0 else 0.0,
            **{k: v[self.time_step] for k, v in vars(self.carbon_intensity).items()},
        }
        observations = {k: data[k] for k in self.active_observations if k in data.keys()}
        unknown_observations = list(set([k for k in self.active_observations]).difference(observations.keys()))
        assert len(unknown_observations) == 0, f'Unkown observations: {unknown_observations}'
        return observations

    @property
    def active_observations(self) -> List[str]:
        """Observations in `observation_metadata` with True value i.e. obeservable."""

        return [k for k, v in self.observation_metadata.items() if v]

    @property
    def active_actions(self) -> List[str]:
        """Actions in `action_metadata` with True value i.e. indicates which storage systems are to be controlled during simulation."""

        return [k for k, v in self.action_metadata.items() if v]

    @property
    def net_electricity_consumption_without_storage_and_pv_emission(self) -> List[float]:
        """Carbon emmissions from `net_electricity_consumption_without_storage_and_pv` time series, in [kg_co2]."""

        return (
            self.carbon_intensity.carbon_intensity[0:self.time_step]*self.net_electricity_consumption_without_storage_and_pv
        ).clip(min=0).tolist()

    @property
    def net_electricity_consumption_without_storage_and_pv_price(self) -> List[float]:
        """net_electricity_consumption_without_storage_and_pv` cost time series, in [$]."""

        return (np.array(self.pricing.electricity_pricing[0:self.time_step])*self.net_electricity_consumption_without_storage_and_pv).tolist()

    @property
    def net_electricity_consumption_without_storage_and_pv(self) -> List[float]:
        """Net electricity consumption in the absence of flexibility provided by `cooling_storage` and self generation time series, in [kWh]. 
        
        Notes
        -----
        net_electricity_consumption_without_storage_and_pv = `net_electricity_consumption_without_storage` - `solar_generation`
        """

        return (np.array(self.net_electricity_consumption_without_storage) - self.solar_generation).tolist()

    @property
    def net_electricity_consumption_without_storage_emission(self) -> List[float]:
        """Carbon emmissions from `net_electricity_consumption_without_storage` time series, in [kg_co2]."""

        return (self.carbon_intensity.carbon_intensity[0:self.time_step]*self.net_electricity_consumption_without_storage).clip(min=0).tolist()

    @property
    def net_electricity_consumption_without_storage_price(self) -> List[float]:
        """`net_electricity_consumption_without_storage` cost time series, in [$]."""

        return (np.array(self.pricing.electricity_pricing[0:self.time_step])*self.net_electricity_consumption_without_storage).tolist()

    @property
    def net_electricity_consumption_without_storage(self) -> List[float]:
        """net electricity consumption in the absence of flexibility provided by storage devices time series, in [kWh]. 
        
        Notes
        -----
        net_electricity_consumption_without_storage = `net_electricity_consumption` - (`cooling_storage_electricity_consumption` + `heating_storage_electricity_consumption` + `dhw_storage_electricity_consumption` + `electrical_storage_electricity_consumption`)
        """

        return (self.net_electricity_consumption - np.sum([
            self.cooling_storage_electricity_consumption,
            self.heating_storage_electricity_consumption,
            self.dhw_storage_electricity_consumption,
            self.electrical_storage_electricity_consumption
        ], axis = 0)).tolist()

    @property
    def net_electricity_consumption_emission(self) -> List[float]:
        """carbon emmissions from `net_electricity_consumption` time series, in [kg_co2]."""

        return (self.carbon_intensity.carbon_intensity[0:self.time_step]*self.net_electricity_consumption).clip(min=0).tolist()

    @property
    def net_electricity_consumption_price(self) -> List[float]:
        """`net_electricity_consumption` cost time series, in [$]."""

        return (np.array(self.pricing.electricity_pricing[0:self.time_step])*self.net_electricity_consumption).tolist()

    @property
    def net_electricity_consumption(self) -> List[float]:
        """net electricity consumption time series, in [kWh]. 
        
        Notes
        -----
        net_electricity_consumption = `cooling_electricity_consumption` + `heating_electricity_consumption` + `dhw_electricity_consumption` + `electrical_storage_electricity_consumption` + `non_shiftable_load_demand` + `solar_generation`
        """

        return np.sum([
            self.cooling_electricity_consumption,
            self.heating_electricity_consumption,
            self.dhw_electricity_consumption,
            self.electrical_storage_electricity_consumption,
            self.non_shiftable_load_demand,
            self.solar_generation,
        ], axis = 0).tolist()

    @property
    def cooling_electricity_consumption(self) -> List[float]:
        """`cooling_device` net electricity consumption in meeting domestic hot water and `cooling_stoage` energy demand time series, in [kWh]. 
        
        Positive values indicate `cooling_device` electricity consumption to charge `cooling_storage` and/or meet `cooling_demand` while negative values indicate avoided `cooling_device` 
        electricity consumption by discharging `cooling_storage` to meet `cooling_demand`.
        """

        demand = np.sum([self.cooling_demand, self.cooling_storage.energy_balance], axis = 0)
        consumption = self.cooling_device.get_input_power(demand, self.weather.outdoor_dry_bulb_temperature[:self.time_step], False)
        return list(consumption)

    @property
    def heating_electricity_consumption(self) -> List[float]:
        """`heating_device` net electricity consumption in meeting domestic hot water and `heating_stoage` energy demand time series, in [kWh]. 
        
        Positive values indicate `heating_device` electricity consumption to charge `heating_storage` and/or meet `heating_demand` while negative values indicate avoided `heating_device` 
        electricity consumption by discharging `heating_storage` to meet `heating_demand`.
        """

        demand = np.sum([self.heating_demand, self.heating_storage.energy_balance], axis = 0)

        if isinstance(self.heating_device, HeatPump):
            consumption = self.heating_device.get_input_power(demand, self.weather.outdoor_dry_bulb_temperature[:self.time_step], True)
        else:
            consumption = self.dhw_device.get_input_power(demand)

        return list(consumption)

    @property
    def dhw_electricity_consumption(self) -> List[float]:
        """`dhw_device` net electricity consumption in meeting domestic hot water and `dhw_stoage` energy demand time series, in [kWh]. 
        
        Positive values indicate `dhw_device` electricity consumption to charge `dhw_storage` and/or meet `dhw_demand` while negative values indicate avoided `dhw_device` 
        electricity consumption by discharging `dhw_storage` to meet `dhw_demand`.
        """

        demand = np.sum([self.dhw_demand, self.dhw_storage.energy_balance], axis = 0)

        if isinstance(self.dhw_device, HeatPump):
            consumption = self.dhw_device.get_input_power(demand, self.weather.outdoor_dry_bulb_temperature[:self.time_step], True)
        else:
            consumption = self.dhw_device.get_input_power(demand)

        return list(consumption)

    @property
    def cooling_storage_electricity_consumption(self) -> List[float]:
        """`cooling_storage` net electricity consumption time series, in [kWh]. 
        
        Positive values indicate `cooling_device` electricity consumption to charge `cooling_storage` while negative values indicate avoided `cooling_device` 
        electricity consumption by discharging `cooling_storage` to meet `cooling_demand`.
        """

        consumption = self.cooling_device.get_input_power(self.cooling_storage.energy_balance, self.weather.outdoor_dry_bulb_temperature[:self.time_step], False)
        return list(consumption)

    @property
    def heating_storage_electricity_consumption(self) -> List[float]:
        """`heating_storage` net electricity consumption time series, in [kWh]. 
        
        Positive values indicate `heating_device` electricity consumption to charge `heating_storage` while negative values indicate avoided `heating_device` 
        electricity consumption by discharging `heating_storage` to meet `heating_demand`.
        """

        if isinstance(self.heating_device, HeatPump):
            consumption = self.heating_device.get_input_power(self.heating_storage.energy_balance, self.weather.outdoor_dry_bulb_temperature[:self.time_step], True)
        else:
            consumption = self.heating_device.get_input_power(self.heating_storage.energy_balance)

        return list(consumption)

    @property
    def dhw_storage_electricity_consumption(self) -> List[float]:
        """`dhw_storage` net electricity consumption time series, in [kWh]. 
        
        Positive values indicate `dhw_device` electricity consumption to charge `dhw_storage` while negative values indicate avoided `dhw_device` 
        electricity consumption by discharging `dhw_storage` to meet `dhw_demand`.
        """

        if isinstance(self.dhw_device, HeatPump):
            consumption = self.dhw_device.get_input_power(self.dhw_storage.energy_balance, self.weather.outdoor_dry_bulb_temperature[:self.time_step], True)
        else:
            consumption = self.dhw_device.get_input_power(self.dhw_storage.energy_balance)

        return list(consumption)

    @property
    def electrical_storage_electricity_consumption(self) -> List[float]:
        """Energy supply from grid and/or `PV` to `electrical_storage` time series, in [kWh]."""

        return self.electrical_storage.electricity_consumption

    @property
    def energy_from_cooling_device_to_cooling_storage(self) -> List[float]:
        """Energy supply from `cooling_device` to `cooling_storage` time series, in [kWh]."""

        return np.array(self.cooling_storage.energy_balance).clip(min=0).tolist()

    @property
    def energy_from_heating_device_to_heating_storage(self) -> List[float]:
        """Energy supply from `heating_device` to `heating_storage` time series, in [kWh]."""

        return np.array(self.heating_storage.energy_balance).clip(min=0).tolist()

    @property
    def energy_from_dhw_device_to_dhw_storage(self) -> List[float]:
        """Energy supply from `dhw_device` to `dhw_storage` time series, in [kWh]."""

        return np.array(self.dhw_storage.energy_balance).clip(min=0).tolist()

    @property
    def energy_to_electrical_storage(self) -> List[float]:
        """Energy supply from `electrical_device` to building time series, in [kWh]."""

        return np.array(self.electrical_storage.energy_balance).clip(min=0).tolist()

    @property
    def energy_from_cooling_device(self) -> List[float]:
        """Energy supply from `cooling_device` to building time series, in [kWh]."""

        return (np.array(self.cooling_demand) - self.energy_from_cooling_storage).tolist()

    @property
    def energy_from_heating_device(self) -> List[float]:
        """Energy supply from `heating_device` to building time series, in [kWh]."""

        return (np.array(self.heating_demand) - self.energy_from_heating_storage).tolist()

    @property
    def energy_from_dhw_device(self) -> List[float]:
        """Energy supply from `dhw_device` to building time series, in [kWh]."""

        return (np.array(self.dhw_demand) - self.energy_from_dhw_storage).tolist()

    @property
    def energy_from_cooling_storage(self) -> List[float]:
        """Energy supply from `cooling_storage` to building time series, in [kWh]."""

        return (np.array(self.cooling_storage.energy_balance).clip(max = 0)*-1).tolist()

    @property
    def energy_from_heating_storage(self) -> List[float]:
        """Energy supply from `heating_storage` to building time series, in [kWh]."""

        return (np.array(self.heating_storage.energy_balance).clip(max = 0)*-1).tolist()

    @property
    def energy_from_dhw_storage(self) -> List[float]:
        """Energy supply from `dhw_storage` to building time series, in [kWh]."""

        return (np.array(self.dhw_storage.energy_balance).clip(max = 0)*-1).tolist()

    @property
    def energy_from_electrical_storage(self) -> List[float]:
        """Energy supply from `electrical_storage` to building time series, in [kWh]."""

        return (np.array(self.electrical_storage.energy_balance).clip(max = 0)*-1).tolist()

    @property
    def cooling_demand(self) -> List[float]:
        """Space cooling demand to be met by `cooling_device` and/or `cooling_storage` time series, in [kWh]."""

        return self.energy_simulation.cooling_demand.tolist()[0:self.time_step]

    @property
    def heating_demand(self) -> List[float]:
        """Space heating demand to be met by `heating_device` and/or `heating_storage` time series, in [kWh]."""

        return self.energy_simulation.heating_demand.tolist()[0:self.time_step]

    @property
    def dhw_demand(self) -> List[float]:
        """Domestic hot water demand to be met by `dhw_device` and/or `dhw_storage` time series, in [kWh]."""

        return self.energy_simulation.dhw_demand.tolist()[0:self.time_step]

    @property
    def non_shiftable_load_demand(self) -> List[float]:
        """Electricity load that must be met by the grid, or `PV` and/or `electrical_storage` if available time series, in [kWh]."""

        return self.energy_simulation.non_shiftable_load.tolist()[0:self.time_step]

    @property
    def solar_generation(self) -> List[float]:
        """`PV` solar generation (negative value) time series, in [kWh]."""

        return (np.array(self.pv.get_generation(self.energy_simulation.solar_generation[0:self.time_step]))*-1).tolist()

    @energy_simulation.setter
    def energy_simulation(self, energy_simulation: EnergySimulation):
        self.__energy_simulation = energy_simulation

    @weather.setter
    def weather(self, weather: Weather):
        self.__weather = weather

    @observation_metadata.setter
    def observation_metadata(self, observation_metadata: Mapping[str, bool]):
        self.__observation_metadata = observation_metadata

    @action_metadata.setter
    def action_metadata(self, action_metadata: Mapping[str, bool]):
        self.__action_metadata = action_metadata

    @carbon_intensity.setter
    def carbon_intensity(self, carbon_intensity: CarbonIntensity):
        if carbon_intensity is None:
            self.__carbon_intensity = CarbonIntensity(np.zeros(len(self.energy_simulation.hour), dtype = float))
        else:
            self.__carbon_intensity = carbon_intensity

    @pricing.setter
    def pricing(self, pricing: Pricing):
        if pricing is None:
            self.__pricing = Pricing(
                np.zeros(len(self.energy_simulation.hour), dtype = float),
                np.zeros(len(self.energy_simulation.hour), dtype = float),
                np.zeros(len(self.energy_simulation.hour), dtype = float)
            )
        else:
            self.__pricing = pricing

    @dhw_storage.setter
    def dhw_storage(self, dhw_storage: StorageTank):
        self.__dhw_storage = StorageTank(0.0) if dhw_storage is None else dhw_storage

    @cooling_storage.setter
    def cooling_storage(self, cooling_storage: StorageTank):
        self.__cooling_storage = StorageTank(0.0) if cooling_storage is None else cooling_storage

    @heating_storage.setter
    def heating_storage(self, heating_storage: StorageTank):
        self.__heating_storage = StorageTank(0.0) if heating_storage is None else heating_storage

    @electrical_storage.setter
    def electrical_storage(self, electrical_storage: Battery):
        self.__electrical_storage = Battery(0.0, 0.0) if electrical_storage is None else electrical_storage

    @dhw_device.setter
    def dhw_device(self, dhw_device: Union[HeatPump, ElectricHeater]):
        self.__dhw_device = ElectricHeater(0.0) if dhw_device is None else dhw_device

    @cooling_device.setter
    def cooling_device(self, cooling_device: HeatPump):
        self.__cooling_device = HeatPump(0.0) if cooling_device is None else cooling_device

    @heating_device.setter
    def heating_device(self, heating_device: Union[HeatPump, ElectricHeater]):
        self.__heating_device = HeatPump(0.0) if heating_device is None else heating_device

    @pv.setter
    def pv(self, pv: PV):
        self.__pv = PV(0.0) if pv is None else pv

    @observation_space.setter
    def observation_space(self, observation_space: spaces.Box):
        self.__observation_space = observation_space

    @action_space.setter
    def action_space(self, action_space: spaces.Box):
        self.__action_space = action_space

    @name.setter
    def name(self, name: str):
        self.__name = self.uid if name is None else name

    def apply_actions(self, cooling_storage_action: float = 0, heating_storage_action: float = 0, dhw_storage_action: float = 0, electrical_storage_action: float = 0):
        r"""Charge/discharge storage devices.

        Parameters
        ----------
        cooling_storage_action : float, default: 0
            Fraction of `cooling_storage` `capacity` to charge/discharge by.
        heating_storage_action : float, default: 0
            Fraction of `heating_storage` `capacity` to charge/discharge by.
        dhw_storage_action : float, default: 0
            Fraction of `dhw_storage` `capacity` to charge/discharge by.
        electrical_storage_action : float, default: 0
            Fraction of `electrical_storage` `capacity` to charge/discharge by.
        """

        self.update_cooling(cooling_storage_action)
        self.update_heating(heating_storage_action)
        self.update_dhw(dhw_storage_action)
        self.update_electrical_storage(electrical_storage_action)

    def update_cooling(self, action: float = 0):
        r"""Charge/discharge `cooling_storage`.

        Parameters
        ----------
        action : float, default: 0
            Fraction of `cooling_storage` `capacity` to charge/discharge by.
        """

        energy = action*self.cooling_storage.capacity
        space_demand = self.energy_simulation.cooling_demand[self.time_step]
        space_demand = 0 if space_demand is None else space_demand # case where space demand is unknown
        max_output = self.cooling_device.get_max_output_power(self.weather.outdoor_dry_bulb_temperature[self.time_step], False)
        energy = max(-space_demand, min(max_output - space_demand, energy))
        self.cooling_storage.charge(energy)
        input_power = self.cooling_device.get_input_power(space_demand + energy, self.weather.outdoor_dry_bulb_temperature[self.time_step], False)
        self.cooling_device.update_electricity_consumption(input_power)

    def update_heating(self, action: float = 0):
        r"""Charge/discharge `heating_storage`.

        Parameters
        ----------
        action : float, default: 0
            Fraction of `heating_storage` `capacity` to charge/discharge by.
        """

        energy = action*self.heating_storage.capacity
        space_demand = self.energy_simulation.heating_demand[self.time_step]
        space_demand = 0 if space_demand is None else space_demand # case where space demand is unknown
        max_output = self.heating_device.get_max_output_power(self.weather.outdoor_dry_bulb_temperature[self.time_step], False)\
            if isinstance(self.heating_device, HeatPump) else self.heating_device.get_max_output_power()
        energy = max(-space_demand, min(max_output - space_demand, energy))
        self.heating_storage.charge(energy)
        demand = space_demand + energy
        input_power = self.heating_device.get_input_power(demand, self.weather.outdoor_dry_bulb_temperature[self.time_step], False)\
            if isinstance(self.heating_device, HeatPump) else self.heating_device.get_input_power(demand)
        self.heating_device.update_electricity_consumption(input_power) 

    def update_dhw(self, action: float = 0):
        r"""Charge/discharge `dhw_storage`.

        Parameters
        ----------
        action : float, default: 0
            Fraction of `dhw_storage` `capacity` to charge/discharge by.
        """

        energy = action*self.dhw_storage.capacity
        space_demand = self.energy_simulation.dhw_demand[self.time_step]
        space_demand = 0 if space_demand is None else space_demand # case where space demand is unknown
        max_output = self.dhw_device.get_max_output_power(self.weather.outdoor_dry_bulb_temperature[self.time_step], False)\
            if isinstance(self.dhw_device, HeatPump) else self.dhw_device.get_max_output_power()
        energy = max(-space_demand, min(max_output - space_demand, energy))
        self.dhw_storage.charge(energy)
        demand = space_demand + energy
        input_power = self.dhw_device.get_input_power(demand, self.weather.outdoor_dry_bulb_temperature[self.time_step], False)\
            if isinstance(self.dhw_device, HeatPump) else self.dhw_device.get_input_power(demand)
        self.dhw_device.update_electricity_consumption(input_power) 

    def update_electrical_storage(self, action: float = 0):
        r"""Charge/discharge `electrical_storage`.

        Parameters
        ----------
        action : float, default: 0
            Fraction of `electrical_storage` `capacity` to charge/discharge by.
        """

        energy = action*self.electrical_storage.capacity
        self.electrical_storage.charge(energy)

    def estimate_observation_space(self) -> spaces.Box:
        r"""Get estimate of observation spaces.

        Find minimum and maximum possible values of all the observations, which can then be used by the RL agent to scale the observations and train any function approximators more effectively.

        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.

        Notes
        -----
        Lower and upper bounds of net electricity consumption are rough estimates and may not be completely accurate hence,
        scaling this observation-variable using these bounds may result in normalized values above 1 or below 0.
        """

        low_limit, high_limit = [], []
        data = {
            'solar_generation':np.array(self.pv.get_generation(self.energy_simulation.solar_generation)),
            **vars(self.energy_simulation),
            **vars(self.weather),
            **vars(self.carbon_intensity),
            **vars(self.pricing),
        }

        for key in self.active_observations:
            if key == 'net_electricity_consumption':
                low_limit.append(0.0)
                net_electric_consumption = self.energy_simulation.non_shiftable_load\
                    + (self.energy_simulation.dhw_demand/0.8)\
                        + self.energy_simulation.cooling_demand\
                            + self.energy_simulation.heating_demand\
                                + (self.dhw_storage.capacity/0.8)\
                                    + (self.cooling_storage.capacity/2.0)\
                                        + (self.heating_storage.capacity/2.0)\
                                            - data['solar_generation']
                high_limit.append(max(net_electric_consumption))

            elif key in ['cooling_storage_soc', 'heating_storage_soc', 'dhw_storage_soc', 'electrical_storage_soc']:
                low_limit.append(0.0)
                high_limit.append(1.0)

            else:
                low_limit.append(min(data[key]))
                high_limit.append(max(data[key]))

        return spaces.Box(low=np.array(low_limit), high=np.array(high_limit), dtype=np.float32)
    
    def estimate_action_space(self) -> spaces.Box:
        r"""Get estimate of action spaces.

        Find minimum and maximum possible values of all the actions, which can then be used by the RL agent to scale the selected actions.

        Returns
        -------
        action_space : spaces.Box
            Action low and high limits.

        Notes
        -----
        The lower and upper bounds for the `cooling_storage`, `heating_storage` and `dhw_storage` actions are set to (+/-) 1/maximum_demand for each respective end use, 
        as the energy storage device can't provide the building with more energy than it will ever need for a given time step. . 
        For example, if `cooling_storage` capacity is 20 kWh and the maximum `cooling_demand` is 5 kWh, its actions will be bounded between -5/20 and 5/20.
        These boundaries should speed up the learning process of the agents and make them more stable compared to setting them to -1 and 1. 
        """
        
        low_limit, high_limit = [], []
 
        for key in self.active_actions:
            if key == 'electrical_storage':
                low_limit.append(-1.0)
                high_limit.append(1.0)
            
            else:
                capacity = vars(self)[f'_{self.__class__.__name__}__{key}'].capacity
                maximum_demand = vars(self)[f'_{self.__class__.__name__}__energy_simulation_{key.split("_")[-1]}_demand'].max()

                try:
                    low_limit.append(max([-1.0/(maximum_demand/capacity), -1.0]))
                    high_limit.append(min([1.0/(maximum_demand/capacity), 1.0]))
                except ZeroDivisionError:
                    low_limit.append(-1.0)
                    high_limit.append(1.0)
                    
        return spaces.Box(low=np.array(low_limit), high=np.array(high_limit), dtype=np.float32)

    def autosize_cooling_device(self, **kwargs):
        """Autosize `cooling_device` `nominal_power` to minimum power needed to always meet `cooling_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `cooling_device` `autosize` function.
        """

        self.cooling_device.autosize(self.weather.outdoor_dry_bulb_temperature, cooling_demand = self.energy_simulation.cooling_demand, **kwargs)

    def autosize_heating_device(self, **kwargs):
        """Autosize `heating_device` `nominal_power` to minimum power needed to always meet `heating_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `heating_device` `autosize` function.
        """

        self.heating_device.autosize(self.weather.outdoor_dry_bulb_temperature, heating_demand = self.energy_simulation.heating_demand, **kwargs)\
            if isinstance(self.heating_device, HeatPump) else self.heating_device.autosize(self.energy_simulation.heating_demand, **kwargs)

    def autosize_dhw_device(self, **kwargs):
        """Autosize `dhw_device` `nominal_power` to minimum power needed to always meet `dhw_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `dhw_device` `autosize` function.
        """

        self.dhw_device.autosize(self.weather.outdoor_dry_bulb_temperature, heating_demand = self.energy_simulation.dhw_demand, **kwargs)\
            if isinstance(self.dhw_device, HeatPump) else self.dhw_device.autosize(self.energy_simulation.dhw_demand, **kwargs)

    def autosize_cooling_storage(self, **kwargs):
        """Autosize `cooling_storage` `capacity` to minimum capacity needed to always meet `cooling_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `cooling_storage` `autosize` function.
        """

        self.cooling_storage.autosize(self.energy_simulation.cooling_demand, **kwargs)

    def autosize_heating_storage(self, **kwargs):
        """Autosize `heating_storage` `capacity` to minimum capacity needed to always meet `heating_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `heating_storage` `autosize` function.
        """

        self.heating_storage.autosize(self.energy_simulation.heating_demand, **kwargs)

    def autosize_dhw_storage(self, **kwargs):
        """Autosize `dhw_storage` `capacity` to minimum capacity needed to always meet `dhw_demand`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `dhw_storage` `autosize` function.
        """

        self.dhw_storage.autosize(self.energy_simulation.dhw_demand, **kwargs)

    def autosize_electrical_storage(self, **kwargs):
        """Autosize `electrical_storage` `capacity` to minimum capacity needed to store maximum `solar_generation`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `electrical_storage` `autosize` function.
        """

        self.electrical_storage.autosize(self.pv.get_generation(self.energy_simulation.solar_generation), **kwargs)

    def autosize_pv(self, **kwargs):
        """Autosize `PV` `capacity` to minimum capacity needed to store maximum `solar_generation`.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments parsed to `electrical_storage` `autosize` function.
        """

        self.pv.autosize(self.pv.get_generation(self.energy_simulation.solar_generation), **kwargs)

    def next_time_step(self):
        r"""Advance all energy storage and electric devices and, PV to next `time_step`."""

        self.cooling_device.next_time_step()
        self.heating_device.next_time_step()
        self.dhw_device.next_time_step()
        self.cooling_storage.next_time_step()
        self.heating_storage.next_time_step()
        self.dhw_storage.next_time_step()
        self.electrical_storage.next_time_step()
        self.pv.next_time_step()
        super().next_time_step()

    def reset(self):
        r"""Reset `Building` to initial state."""

        super().reset()
        self.cooling_storage.reset()
        self.heating_storage.reset()
        self.dhw_storage.reset()
        self.electrical_storage.reset()
        self.cooling_device.reset()
        self.heating_device.reset()
        self.dhw_device.reset()
        self.pv.reset()

class District(Environment, Env):
    def __init__(self, buildings: List[Building], time_steps: int, reward_function: RewardFunction, central_agent: bool = False, shared_observations: List[str] = None, **kwargs):
        r"""Initialize `District`.

        Parameters
        ----------
        buildings : List[Building]
            Buildings in district.
        time_steps : int
            Number of simulation time steps.
        reward_function : RewardFunction
            Reward function class instance.
        central_agent : bool, optional
            Expect 1 central agent to control all building storage device.
        shared_observations : List[str], optional
            Names of common observations across all buildings i.e. observations that have the same value irrespective of the building.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize `Environment` and `gym.Env` super classes.
        """

        self.buildings = buildings
        self.time_steps = time_steps
        self.reward_function = reward_function
        self.central_agent = central_agent
        self.shared_observations = shared_observations
        super().__init__()

    @property
    def buildings(self) -> List[Building]:
        """Buildings in district."""

        return self.__buildings

    @property
    def time_steps(self) -> int:
        """Number of simulation time steps."""

        return self.__time_steps

    @property
    def reward_function(self) -> RewardFunction:
        """Reward function class instance"""

        return self.__reward_function

    @property
    def central_agent(self) -> bool:
        """Expect 1 central agent to control all building storage device."""

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
    def observation_encoders(self) -> Union[List[Encoder], List[List[Encoder]]]:
        r"""Get observation value transformers/encoders for use in buildings' agent(s) algorithm.

        See `Building.observation_encoders` documentation for more information.
        
        Returns
        -------
        encoders : Union[List[Encoder], List[List[Encoder]]
            Encoder classes for observations ordered with respect to `active_observations` in each building.

        Notes
        -----
        If `central_agent` is True, 1 list containing all buildings' encoders is returned in the same order as `buildings`. 
        The `shared_observations` encoders are only included in the first building's listing.
        If `central_agent` is False, a list of sublists is returned where each sublist is a list of 1 building's encoders 
        and the sublist in the same order as `buildings`.
        """

        return [
            k for i, b in enumerate(self.buildings) for k, s in zip(b.observation_encoders, b.active_observations) 
            if i == 0 or s not in self.shared_observations
        ] if self.central_agent else [b.observation_encoders for b in self.buildings]

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
            v for i, b in enumerate(self.buildings) for k, v in b.observations.items() if i == 0 or k not in self.shared_observations
        ]] if self.central_agent else [list(b.observations.values()) for b in self.buildings]

    @property
    def net_electricity_consumption_without_storage_and_pv_emission(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_without_storage_and_pv_emission` time series, in [kg_co2]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_and_pv_emission for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_without_storage_and_pv_price(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_without_storage_and_pv_price` time series, in [$]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_and_pv_price for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_without_storage_and_pv(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_without_storage_and_pv` time series, in [kWh]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_and_pv for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_without_storage_emission(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_without_storage_emission` time series, in [kg_co2]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_emission for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_without_storage_price(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_without_storage_price` time series, in [$]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage_price for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_without_storage(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_without_storage` time series, in [kWh]."""

        return pd.DataFrame([b.net_electricity_consumption_without_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_emission(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_emission` time series, in [kg_co2]."""

        return pd.DataFrame([b.net_electricity_consumption_emission for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_price(self) -> List[float]:
        """Summed `Building.net_electricity_consumption_price` time series, in [$]."""

        return pd.DataFrame([b.net_electricity_consumption_price for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption(self) -> List[float]:
        """Summed `Building.net_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.net_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def cooling_electricity_consumption(self) -> List[float]:
        """Summed `Building.cooling_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.cooling_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def heating_electricity_consumption(self) -> List[float]:
        """Summed `Building.heating_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.heating_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def dhw_electricity_consumption(self) -> List[float]:
        """Summed `Building.dhw_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.dhw_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def cooling_storage_electricity_consumption(self) -> List[float]:
        """Summed `Building.cooling_storage_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.cooling_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def heating_storage_electricity_consumption(self) -> List[float]:
        """Summed `Building.heating_storage_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.heating_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def dhw_storage_electricity_consumption(self) -> List[float]:
        """Summed `Building.dhw_storage_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.dhw_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def electrical_storage_electricity_consumption(self) -> List[float]:
        """Summed `Building.electrical_storage_electricity_consumption` time series, in [kWh]."""

        return pd.DataFrame([b.electrical_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_cooling_device_to_cooling_storage(self) -> List[float]:
        """Summed `Building.energy_from_cooling_device_to_cooling_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_cooling_device_to_cooling_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_heating_device_to_heating_storage(self) -> List[float]:
        """Summed `Building.energy_from_heating_device_to_heating_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_heating_device_to_heating_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_dhw_device_to_dhw_storage(self) -> List[float]:
        """Summed `Building.energy_from_dhw_device_to_dhw_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_dhw_device_to_dhw_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_to_electrical_storage(self) -> List[float]:
        """Summed `Building.energy_to_electrical_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_to_electrical_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_cooling_device(self) -> List[float]:
        """Summed `Building.energy_from_cooling_device` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_cooling_device for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_heating_device(self) -> List[float]:
        """Summed `Building.energy_from_heating_device` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_heating_device for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_dhw_device(self) -> List[float]:
        """Summed `Building.energy_from_dhw_device` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_dhw_device for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_cooling_storage(self) -> List[float]:
        """Summed `Building.energy_from_cooling_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_cooling_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_heating_storage(self) -> List[float]:
        """Summed `Building.energy_from_heating_storage` time series, in [kWh]."""
        
        return pd.DataFrame([b.energy_from_heating_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_dhw_storage(self) -> List[float]:
        """Summed `Building.energy_from_dhw_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_dhw_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_electrical_storage(self) -> List[float]:
        """Summed `Building.energy_from_electrical_storage` time series, in [kWh]."""

        return pd.DataFrame([b.energy_from_electrical_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def cooling_demand(self) -> List[float]:
        """Summed `Building.cooling_demand`, in [kWh]."""

        return pd.DataFrame([b.cooling_demand for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def heating_demand(self) -> List[float]:
        """Summed `Building.heating_demand`, in [kWh]."""

        return pd.DataFrame([b.heating_demand for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def dhw_demand(self) -> List[float]:
        """Summed `Building.dhw_demand`, in [kWh]."""

        return pd.DataFrame([b.dhw_demand for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def non_shiftable_load_demand(self) -> List[float]:
        """Summed `Building.non_shiftable_load_demand`, in [kWh]."""

        return pd.DataFrame([b.non_shiftable_load_demand for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def solar_generation(self) -> List[float]:
        """Summed `Building.solar_generation, in [kWh]`."""

        return pd.DataFrame([b.solar_generation for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @buildings.setter
    def buildings(self, buildings: List[Building]):
        self.__buildings = buildings

    @time_steps.setter
    def time_steps(self, time_steps: int):
        assert time_steps >= 1, 'time_steps must be >= 1'
        self.__time_steps = time_steps

    @reward_function.setter
    def reward_function(self, reward_function: RewardFunction):
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
        May be used to assigned :attr:`shared_observations` value during `District` object initialization.
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


    def step(self, actions: List[List[float]]):
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
        info: Mapping[Any, Any]
            A dictionary that may contain additional information regarding the reason for a ``done`` signal.
            `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
            Override :meth"`get_info` to get custom key-value pairs in `info`.
        """

        actions = self.__parse_actions(actions)

        for building, building_actions in zip(self.buildings, actions):
            building.apply_actions(**building_actions)

        self.next_time_step()
        return self.observations, self.get_reward(), self.done, self.get_info()

    def get_reward(self) -> List[float]:
        """Calculate agent(s) reward(s) using :attr:`reward_function`.
        
        Returns
        -------
        reward: List[float]
            Reward for current observations. If `central_agent` is True, `reward` is a list of length = 1 else, `reward` has same length as `buildings`.
        """

        self.reward_function.electricity_consumption = [sum(self.net_electricity_consumption)] if self.central_agent\
            else self.net_electricity_consumption
        self.reward_function.carbon_emission = [sum(self.net_electricity_consumption_emission)] if self.central_agent\
            else self.net_electricity_consumption_emission
        self.reward_function.electricity_price = [sum(self.net_electricity_consumption_price)] if self.central_agent\
            else self.net_electricity_consumption_price
        reward = self.reward_function.calculate()
        return reward

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
            building_actions = actions

        active_actions = [[k for k, v in b.action_metadata.items() if v] for b in self.buildings]
        actions = [{k:a for k, a in zip(active_actions[i],building_actions[i])} for i in range(len(active_actions))]
        actions = [{f'{k}_action':actions[i].get(k, 0.0) for k in b.action_metadata}  for i, b in enumerate(self.buildings)]
        return actions
    
    def get_building_information(self) -> Mapping[str, Any]:
        """Get buildings PV capacity, end-use annual demands, and correlations with other buildings end-use annual demands.

        Returns
        -------
        building_information: Mapping[str, Any]
            Building information summary.
        """

        np.seterr(divide='ignore', invalid='ignore')
        building_info = {}
        n_years = max(1, self.time_steps*self.seconds_per_time_step/8760*3600)

        for building in self.buildings:
            building_info[building.uid] = {}
            building_info[building.uid]['solar_power_capacity'] = round(building.pv.capacity, 3)
            building_info[building.uid]['annual_dhw_demand'] = round(sum(building.energy_simulation.dhw_demand)/n_years, 3)
            building_info[building.uid]['annual_cooling_demand'] = round(sum(building.energy_simulation.cooling_demand)/n_years, 3)
            building_info[building.uid]['annual_heating_demand'] = round(sum(building.energy_simulation.heating_demand)/n_years, 3)
            building_info[building.uid]['annual_nonshiftable_electrical_demand'] = round(sum(building.energy_simulation.non_shiftable_load)/n_years, 3)
            building_info[building.uid]['correlations_dhw'] = {}
            building_info[building.uid]['correlations_cooling_demand'] = {}
            building_info[building.uid]['correlations_non_shiftable_load'] = {}
            
            for corr_building in self.buildings:
                if building.uid != corr_building.uid:
                    building_info[building.uid]['correlations_dhw'][corr_building.uid] = round((np.corrcoef(
                        np.array(building.energy_simulation.dhw_demand), np.array(corr_building.energy_simulation.dhw_demand)
                    ))[0][1], 3)
                    building_info[building.uid]['correlations_cooling_demand'][corr_building.uid] = round((np.corrcoef(
                        np.array(building.energy_simulation.cooling_demand), np.array(corr_building.energy_simulation.cooling_demand)
                    ))[0][1], 3)
                    building_info[building.uid]['correlations_heating_demand'][corr_building.uid] = round((np.corrcoef(
                        np.array(building.energy_simulation.heating_demand), np.array(corr_building.energy_simulation.heating_demand)
                    ))[0][1], 3)
                    building_info[building.uid]['correlations_non_shiftable_load'][corr_building.uid] = round((np.corrcoef(
                        np.array(building.energy_simulation.non_shiftable_load), np.array(corr_building.energy_simulation.non_shiftable_load)
                    ))[0][1], 3)
                else:
                    continue
        
        return building_info

    def next_time_step(self):
        r"""Advance all buildings to next `time_step`."""

        for building in self.buildings:
            building.next_time_step()
            
        super().next_time_step()

    def reset(self):
        r"""Reset `District` to initial state.
        
        Returns
        -------
        observations: List[List[float]]
            :attr:`observations`. 
        """

        super().reset()

        for building in self.buildings:
            building.reset()

        return self.observations

class CityLearn:
    def __init__(self, district: District, agents: List[Agent], episodes: int = None):
        r"""Initialize `CityLearn`.

        Parameters
        ----------
        district : District
            Simulation district.
        agents : List[Agent]
            Simulation agents for `district.buildings` energy storage charging/discharging management.
        episodes : int
            Number of times to simulate until terminal state is reached.
        """

        self.district = district
        self.agents = agents
        self.episodes = episodes

    @property
    def district(self) -> District:
        """Simulation district."""

        return self.__district

    @property
    def agents(self) -> List[Agent]:
        """Simulation agents for `district.buildings` energy storage charging/discharging management."""

        return self.__agents

    @property
    def episodes(self) -> int:
        """Number of times to simulate until terminal state is reached."""

        return self.__episodes

    @district.setter
    def district(self, district: District):
        self.__district = district

    @agents.setter
    def agents(self, agents: List[Agent]):
        if self.district.central_agent:
            assert len(agents) == 1, 'Only 1 agent is expected when `district.central_agent` = True.'
        else:
            assert len(agents) == len(self.district.buildings), 'Length of `agents` and `district.buildings` must be equal when using `district.central_agent` = False.'

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
            observations_list = self.district.reset()

            while not self.district.done:
                logging.debug(f'Timestep: {self.district.time_step}/{self.district.time_steps - 1}, Episode: {episode}')
                actions_list = []

                # select actions
                for agent, observations in zip(self.agents, observations_list):
                    if agent.action_dimension > 0:
                        actions_list.append(agent.select_actions(observations))
                    else:
                        actions_list.append([]) 

                # apply actions to district
                next_observations_list, reward_list, _, _ = self.district.step(actions_list)

                # update
                for agent, observations, actions, reward, next_observations in zip(self.agents, observations_list, actions_list, reward_list, next_observations_list):
                    if agent.action_dimension > 0:
                        agent.add_to_buffer(observations, actions, reward, next_observations, done = self.district.done)
                    else:
                        continue

                observations_list = [o for o in next_observations_list]

    @classmethod
    def load(cls, schema: Union[str, Path, Mapping[str, Any]]) -> Tuple[District, List[Agent]]:
        """Return `District` and `Controller` objects as defined by the `schema`.

        Parameters
        ----------
        schema: Union[str, Path, Mapping[str, Any]]
            Filepath to JSON representation or `dict` object of CityLearn schema.
        
        Returns
        -------
        district: District
            Simulation district.
        agents: List[Agent]
            Simulation agents for `district.buildings` energy storage charging/discharging management.
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
                            autosize_kwargs = {} if building_schema[name].get('autosize_kwargs', None) is None else building_schema[name]['autosize_kwargs']
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
        reward_function = reward_function_constructor(agent_count)
        district = District(list(buildings), timesteps, reward_function, central_agent = central_agent, shared_observations = shared_observations)
        # agent_type = schema['agent']['type']
        # agent_module = '.'.join(agent_type.split('.')[0:-1])
        # agent_name = agent_type.split('.')[-1]
        # agent_constructor = getattr(importlib.import_module(agent_module), agent_name)
        # agent_attributes = schema['agent'].get('attributes', {})
        # agent_attributes = [{
        #     'building_ids':[b.uid for b in buildings],
        #     'action_space':district.action_space[i],
        #     'observation_space':district.observation_space[i],
        #     'encoders':district.observation_encoders[i],
        #     **agent_attributes
        # }  for i in range(agent_count)]
        # agents = [agent_constructor(**agent_attribute) for agent_attribute in agent_attributes]
        agents = None
        episodes = schema['episodes']

        return district, agents, episodes