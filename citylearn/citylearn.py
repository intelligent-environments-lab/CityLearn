import os
from pathlib import Path
from typing import Any, List, Mapping, Union
from gym import Env, spaces
import numpy as np
import pandas as pd
from citylearn.base import Environment
from citylearn.data import EnergySimulation, CarbonIntensity, Weather
from citylearn.energy_model import Battery, ElectricHeater, HeatPump, PV, StorageTank
from citylearn.preprocessing import Normalize, OnehotEncoding, PeriodicNormalization, RemoveFeature
from citylearn.utilities import read_json
    
class Building(Environment):
    def __init__(
        self, energy_simulation: EnergySimulation, weather: Weather, state_metadata: Mapping[str, bool], action_metadata: Mapping[str, bool], carbon_intensity: CarbonIntensity = None, 
        dhw_storage: StorageTank = None, cooling_storage: StorageTank = None, heating_storage: StorageTank = None, electrical_storage: Battery = None, 
        dhw_device: Union[HeatPump, ElectricHeater] = None, cooling_device: HeatPump = None, heating_device: Union[HeatPump, ElectricHeater] = None, pv: PV = None, name: str = None
    ):
        self.name = name
        self.energy_simulation = energy_simulation
        self.weather = weather
        self.carbon_intensity = carbon_intensity
        self.dhw_storage = dhw_storage
        self.cooling_storage = cooling_storage
        self.heating_storage = heating_storage
        self.electrical_storage = electrical_storage
        self.dhw_device = dhw_device
        self.cooling_device = cooling_device
        self.heating_device = heating_device
        self.pv = pv
        self.state_metadata = state_metadata
        self.action_metadata = action_metadata
        super().__init__()

    @property
    def energy_simulation(self) -> EnergySimulation:
        return self.__energy_simulation

    @property
    def weather(self) -> Weather:
        return self.__weather

    @property
    def state_metadata(self) -> Mapping[str, bool]:
        return self.__state_metadata

    @property
    def action_metadata(self) -> Mapping[str, bool]:
        return self.__action_metadata

    @property
    def carbon_intensity(self) -> CarbonIntensity:
        return self.__carbon_intensity

    @property
    def dhw_storage(self) -> StorageTank:
        return self.__dhw_storage

    @property
    def cooling_storage(self) -> StorageTank:
        return self.__cooling_storage

    @property
    def heating_storage(self) -> StorageTank:
        return self.__heating_storage

    @property
    def electrical_storage(self) -> Battery:
        return self.__electrical_storage

    @property
    def dhw_device(self) -> Union[HeatPump, ElectricHeater]:
        return self.__dhw_device

    @property
    def cooling_device(self) -> HeatPump:
        return self.__cooling_device

    @property
    def heating_device(self) -> Union[HeatPump, ElectricHeater]:
        return self.__heating_device

    @property
    def pv(self) -> PV:
        return self.__pv

    @property
    def name(self) -> str:
        return self.__name

    @property
    def observation_encoders(self) -> List[Mapping[str, Any]]:
        active_states = [k for k, v in self.state_metadata.items() if v]
        remove_features = ['net_electricity_consumption']
        remove_features += [
            'solar_generation', 'diffuse_solar_radiation', 'diffuse_solar_radiation_predicted_6h',
            'diffuse_solar_radiation_predicted_12h', 'diffuse_solar_radiation_predicted_24h',
            'direct_solar_radiation', 'direct_solar_radiation_predicted_6h',
            'direct_solar_radiation_predicted_12h', 'direct_solar_radiation_predicted_24h',
        ] if self.pv.capacity == 0 else []
        demand_states = {
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
        remove_features += [k for k, v in demand_states.items() if v == 0]
        remove_features = [f for f in remove_features if f in active_states]
        
        encoders = []

        for i, state in enumerate(active_states):
            if state in ['month', 'hour']:
                encoders.append(PeriodicNormalization(self.observation_spaces.high[i]))
            
            elif state == 'day_type':
                encoders.append(OnehotEncoding([0, 1, 2, 3, 4, 5, 6, 7, 8]))
            
            elif state == "daylight_savings_status":
                encoders.append(OnehotEncoding([0, 1, 2]))
            
            elif state in remove_features:
                encoders.append(RemoveFeature())
            
            else:
                encoders.append(Normalize(self.observation_spaces.low[i], self.observation_spaces.high[i]))

        return encoders

    @property
    def observation_spaces(self) -> spaces.Box:
        # Finding the max and min possible values of all the states, which can then be used 
        # by the RL agent to scale the states and train any function approximators more effectively
        low_limit, high_limit = [], []
        data = {
            'solar_generation':np.array(self.pv.get_generation(self.energy_simulation.solar_generation)),
            **vars(self.energy_simulation),
            **vars(self.weather),
            **vars(self.carbon_intensity),
        }

        for key, value in self.state_metadata.items():
            if value:
                if key == 'net_electricity_consumption':
                    # lower and upper bounds of net electricity consumption are rough estimates and may not be completely accurate. 
                    # Scaling this state-variable using these bounds may result in normalized values above 1 or below 0.
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
            
            else:
                continue

        return spaces.Box(low=np.array(low_limit), high=np.array(high_limit), dtype=np.float32)
    
    @property
    def action_spaces(self) -> spaces.Box:
        
        low_limit, high_limit = [], []
 
        for key, value in self.action_metadata.items():
            if value:
                if key == 'electrical_storage':
                    low_limit.append(-1.0)
                    high_limit.append(1.0)
                
                else:
                    '''The energy storage (tank) capacity indicates how many times bigger the tank is compared to the maximum hourly energy demand of the building (cooling or DHW respectively), which sets a lower bound for the action of 1/tank_capacity, as the energy storage device can't provide the building with more energy than it will ever need for a given hour. The heat pump is sized using approximately the maximum hourly energy demand of the building (after accounting for the COP, see function autosize). Therefore, we make the fair assumption that the action also has an upper bound equal to 1/tank_capacity. This boundaries should speed up the learning process of the agents and make them more stable rather than if we just set them to -1 and 1. I.e. if Chilled_Water_Tank.Capacity is 3 (3 times the max. hourly demand of the building in the entire year), its actions will be bounded between -1/3 and 1/3'''
                    capacity = vars(self)[f'_{self.__class__.__name__}__{key}'].capacity

                    try:
                        low_limit.append(max([-1.0/capacity, -1.0]))
                        high_limit.append(min([1.0/capacity, 1.0]))
                    except ZeroDivisionError:
                        low_limit.append(-1.0)
                        high_limit.append(1.0)

            else:
                continue
                    
        return spaces.Box(low=np.array(low_limit), high=np.array(high_limit), dtype=np.float32)

    @property
    def states(self) -> Mapping[str, float]:
        states = {}
        data = {
            **{k: v[self.time_step] for k, v in vars(self.energy_simulation).items()},
            **{k: v[self.time_step] for k, v in vars(self.weather).items()},
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
        states = {k: data[k] for k, v in self.state_metadata.items() if v and k in data.keys()}
        unknown_states = list(set([k for k, v in self.state_metadata.items() if v]).difference(states.keys()))
        assert len(unknown_states) == 0, f'Unkown states: {unknown_states}'
        return states

    @property
    def net_electricity_consumption_without_storage_and_pv(self) -> List[float]:
        return (np.array(self.net_electricity_consumption_without_storage) - self.solar_generation).tolist()

    @property
    def net_electricity_consumption_without_storage(self) -> List[float]:
        return (self.net_electricity_consumption - np.sum([
            self.cooling_storage_electricity_consumption,
            self.heating_storage_electricity_consumption,
            self.dhw_storage_electricity_consumption,
            self.electrical_storage_electricity_consumption
        ], axis = 0)).tolist()

    @property
    def net_electricity_consumption(self) -> List[float]:
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
        demand = np.sum([self.cooling_demand, self.cooling_storage.energy_balance], axis = 0)
        consumption = self.cooling_device.get_input_power(demand, self.weather.outdoor_drybulb_temperature[:self.time_step], False)
        return list(consumption)

    @property
    def heating_electricity_consumption(self) -> List[float]:
        demand = np.sum([self.heating_demand, self.heating_storage.energy_balance], axis = 0)

        if isinstance(self.heating_device, HeatPump):
            consumption = self.heating_device.get_input_power(demand, self.weather.outdoor_drybulb_temperature[:self.time_step], True)
        else:
            consumption = self.dhw_device.get_input_power(demand)

        return list(consumption)

    @property
    def dhw_electricity_consumption(self) -> List[float]:
        demand = np.sum([self.dhw_demand, self.dhw_storage.energy_balance], axis = 0)

        if isinstance(self.dhw_device, HeatPump):
            consumption = self.dhw_device.get_input_power(demand, self.weather.outdoor_drybulb_temperature[:self.time_step], True)
        else:
            consumption = self.dhw_device.get_input_power(demand)

        return list(consumption)

    @property
    def cooling_storage_electricity_consumption(self) -> List[float]:
        consumption = self.cooling_device.get_input_power(self.cooling_storage.energy_balance, self.weather.outdoor_drybulb_temperature[:self.time_step], False)
        return list(consumption)

    @property
    def heating_storage_electricity_consumption(self) -> List[float]:
        if isinstance(self.heating_device, HeatPump):
            consumption = self.heating_device.get_input_power(self.heating_storage.energy_balance, self.weather.outdoor_drybulb_temperature[:self.time_step], True)
        else:
            consumption = self.heating_device.get_input_power(self.heating_storage.energy_balance)

        return list(consumption)

    @property
    def dhw_storage_electricity_consumption(self) -> List[float]:
        if isinstance(self.dhw_device, HeatPump):
            consumption = self.dhw_device.get_input_power(self.dhw_storage.energy_balance, self.weather.outdoor_drybulb_temperature[:self.time_step], True)
        else:
            consumption = self.dhw_device.get_input_power(self.dhw_storage.energy_balance)

        return list(consumption)

    @property
    def electrical_storage_electricity_consumption(self) -> List[float]:
        return self.electrical_storage.electricity_consumption

    @property
    def energy_from_cooling_device_to_cooling_storage(self) -> List[float]:
        return np.array(self.cooling_storage.energy_balance).clip(min=0).tolist()

    @property
    def energy_from_heating_device_to_heating_storage(self) -> List[float]:
        return np.array(self.heating_storage.energy_balance).clip(min=0).tolist()

    @property
    def energy_from_dhw_device_to_dhw_storage(self) -> List[float]:
        return np.array(self.dhw_storage.energy_balance).clip(min=0).tolist()

    @property
    def energy_to_electrical_storage(self) -> List[float]:
        return np.array(self.electrical_storage.energy_balance).clip(min=0).tolist()

    @property
    def energy_from_cooling_device(self) -> List[float]:
        return (np.array(self.cooling_demand) - self.energy_from_cooling_storage).tolist()

    @property
    def energy_from_heating_device(self) -> List[float]:
        return (np.array(self.heating_demand) - self.energy_from_heating_storage).tolist()

    @property
    def energy_from_dhw_device(self) -> List[float]:
        return (np.array(self.dhw_demand) - self.energy_from_dhw_storage).tolist()

    @property
    def energy_from_cooling_storage(self) -> List[float]:
        return (np.array(self.cooling_storage.energy_balance).clip(max = 0)*-1).tolist()

    @property
    def energy_from_heating_storage(self) -> List[float]:
        return (np.array(self.heating_storage.energy_balance).clip(max = 0)*-1).tolist()

    @property
    def energy_from_dhw_storage(self) -> List[float]:
        return (np.array(self.dhw_storage.energy_balance).clip(max = 0)*-1).tolist()

    @property
    def energy_from_electrical_storage(self) -> List[float]:
        return (np.array(self.electrical_storage.energy_balance).clip(max = 0)*-1).tolist()

    @property
    def cooling_demand(self) -> List[float]:
        return self.energy_simulation.cooling_demand.tolist()[0:self.time_step]

    @property
    def heating_demand(self) -> List[float]:
        return self.energy_simulation.heating_demand.tolist()[0:self.time_step]

    @property
    def dhw_demand(self) -> List[float]:
        return self.energy_simulation.dhw_demand.tolist()[0:self.time_step]

    @property
    def non_shiftable_load_demand(self) -> List[float]:
        return self.energy_simulation.non_shiftable_load.tolist()[0:self.time_step]

    @property
    def solar_generation(self) -> List[float]:
        return (np.array(self.pv.get_generation(self.energy_simulation.solar_generation[0:self.time_step]))*-1).tolist()

    @energy_simulation.setter
    def energy_simulation(self, energy_simulation: EnergySimulation):
        self.__energy_simulation = energy_simulation

    @weather.setter
    def weather(self, weather: Weather):
        self.__weather = weather

    @state_metadata.setter
    def state_metadata(self, state_metadata: Mapping[str, bool]):
        self.__state_metadata = state_metadata

    @action_metadata.setter
    def action_metadata(self, action_metadata: Mapping[str, bool]):
        self.__action_metadata = action_metadata

    @carbon_intensity.setter
    def carbon_intensity(self, carbon_intensity: CarbonIntensity):
        if carbon_intensity is None:
            self.__carbon_intensity = CarbonIntensity(np.zeros(len(self.energy_simulation.hour), dtype = float))
        else:
            self.__carbon_intensity = carbon_intensity

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

    @name.setter
    def name(self, name: str):
        self.__name = self.uid if name is None else name

    def apply_actions(self, cooling_storage_action: float = 0, heating_storage_action: float = 0, dhw_storage_action: float = 0, electrical_storage_action: float = 0):
        self.update_cooling(cooling_storage_action)
        self.update_heating(heating_storage_action)
        self.update_dhw(dhw_storage_action)
        self.update_battery(electrical_storage_action)

    def update_cooling(self, action: float = 0):
        energy = action*self.cooling_storage.capacity
        space_demand = self.energy_simulation.cooling_demand[self.time_step]
        space_demand = 0 if space_demand is None else space_demand # case where space demand is unknown
        max_output = self.cooling_device.get_max_output_power(self.weather.outdoor_drybulb_temperature[self.time_step], False)
        energy = max(-space_demand, min(max_output - space_demand, energy))
        self.cooling_storage.charge(energy)

    def update_heating(self, action: float = 0):
        energy = action*self.heating_storage.capacity
        space_demand = self.energy_simulation.heating_demand[self.time_step]
        space_demand = 0 if space_demand is None else space_demand # case where space demand is unknown
        max_output = self.heating_device.get_max_output_power(self.weather.outdoor_drybulb_temperature[self.time_step], False)\
            if isinstance(self.heating_device, HeatPump) else self.heating_device.get_max_output_power()
        energy = max(-space_demand, min(max_output - space_demand, energy))
        self.heating_storage.charge(energy)

    def update_dhw(self, action: float = 0):
        energy = action*self.dhw_storage.capacity
        space_demand = self.energy_simulation.dhw_demand[self.time_step]
        space_demand = 0 if space_demand is None else space_demand # case where space demand is unknown
        max_output = self.dhw_device.get_max_output_power(self.weather.outdoor_drybulb_temperature[self.time_step], False)\
            if isinstance(self.dhw_device, HeatPump) else self.dhw_device.get_max_output_power()
        energy = max(-space_demand, min(max_output - space_demand, energy))
        self.dhw_storage.charge(energy)

    def update_battery(self, action: float = 0):
        energy = action*self.electrical_storage.capacity
        self.electrical_storage.charge(energy)

    def autosize_cooling_device(self):
        self.cooling_device.autosize(self.weather.outdoor_drybulb_temperature, cooling_demand = self.energy_simulation.cooling_demand)

    def autosize_heating_device(self):
        self.heating_device.autosize(self.weather.outdoor_drybulb_temperature, heating_demand = self.energy_simulation.heating_demand)\
            if isinstance(self.heating_device, HeatPump) else self.heating_device.autosize(self.energy_simulation.heating_demand)

    def autosize_dhw_device(self):
        self.dhw_device.autosize(self.weather.outdoor_drybulb_temperature, heating_demand = self.energy_simulation.dhw_demand)\
            if isinstance(self.dhw_device, HeatPump) else self.dhw_device.autosize(self.energy_simulation.dhw_demand)

    def autosize_cooling_storage(self, **kwargs):
        self.cooling_storage.autosize(self.energy_simulation.cooling_demand, **kwargs)

    def autosize_heating_storage(self, **kwargs):
        self.heating_storage.autosize(self.energy_simulation.heating_demand, **kwargs)

    def autosize_dhw_storage(self, **kwargs):
        self.dhw_storage.autosize(self.energy_simulation.dhw_demand, **kwargs)

    def autosize_electrical_storage(self, **kwargs):
        self.electrical_storage.autosize(self.pv.get_generation(self.energy_simulation.solar_generation), **kwargs)

    def autosize_pv(self):
        self.pv.autosize(self.pv.get_generation(self.energy_simulation.solar_generation))

    def next_time_step(self):
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
        super().reset()
        self.cooling_storage.reset()
        self.heating_storage.reset()
        self.dhw_storage.reset()
        self.electrical_storage.reset()
        self.cooling_device.reset()
        self.heating_device.reset()
        self.dhw_device.reset()
        self.pv.reset()

class City(Environment, Env):
    def __init__(self,buildings: List[Building], time_steps: int, central_agent: bool = False, shared_states: List[str] = None):
        self.buildings = buildings
        self.time_steps = time_steps
        self.central_agent = central_agent
        self.shared_states = shared_states
        super().__init__()

    @property
    def buildings(self) -> List[Building]:
        return self.__buildings

    @property
    def time_steps(self) -> int:
        return self.__time_steps

    @property
    def central_agent(self) -> bool:
        return self.__central_agent

    @property
    def shared_states(self) -> List[str]:
        return self.__shared_states

    @property
    def terminal(self) -> bool:
        return self.time_step == self.time_steps - 1

    @property
    def observation_spaces(self) -> List[spaces.Box]:
        return [b.observation_spaces for b in self.buildings]

    @property
    def action_spaces(self) -> List[spaces.Box]:
        return [b.action_spaces for b in self.buildings]

    @property
    def default_shared_states(self) -> List[str]:
        return [
            'month', 'day_type', 'hour', 'daylight_savings_status',
            'outdoor_drybulb_temperature', 'outdoor_drybulb_temperature_predicted_6h',
            'outdoor_drybulb_temperature_predicted_12h', 'outdoor_drybulb_temperature_predicted_24h',
            'outdoor_relative_humidity', 'outdoor_relative_humidity_predicted_6h',
            'outdoor_relative_humidity_predicted_12h', 'outdoor_relative_humidity_predicted_24h',
            'diffuse_solar_radiation', 'diffuse_solar_radiation_predicted_6h',
            'diffuse_solar_radiation_predicted_12h', 'diffuse_solar_radiation_predicted_24h',
            'direct_solar_radiation', 'direct_solar_radiation_predicted_6h',
            'direct_solar_radiation_predicted_12h', 'direct_solar_radiation_predicted_24h',
            'carbon_intensity',
        ]

    @property
    def states(self) -> Union[List[float], List[List[float]]]:
        states = list(self.buildings[0].states.values()) + [
            v for b in self.buildings[1:] for k, v in b.states if k not in self.shared_states
        ] if self.central_agent else [
            list(b.states.values()) for b in self.buildings
        ]
        return states

    @property
    def net_electricity_consumption_without_storage_and_pv(self) -> List[float]:
        return pd.DataFrame([b.net_electricity_consumption_without_storage_and_pv for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption_without_storage(self) -> List[float]:
        return pd.DataFrame([b.net_electricity_consumption_without_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def net_electricity_consumption(self) -> List[float]:
        return pd.DataFrame([b.net_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def cooling_electricity_consumption(self) -> List[float]:
        return pd.DataFrame([b.cooling_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def heating_electricity_consumption(self) -> List[float]:
        return pd.DataFrame([b.heating_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def dhw_electricity_consumption(self) -> List[float]:
        return pd.DataFrame([b.dhw_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def cooling_storage_electricity_consumption(self) -> List[float]:
        return pd.DataFrame([b.cooling_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def heating_storage_electricity_consumption(self) -> List[float]:
        return pd.DataFrame([b.heating_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def dhw_storage_electricity_consumption(self) -> List[float]:
        return pd.DataFrame([b.dhw_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def electrical_storage_electricity_consumption(self) -> List[float]:
        return pd.DataFrame([b.electrical_storage_electricity_consumption for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_cooling_device_to_cooling_storage(self) -> List[float]:
        return pd.DataFrame([b.energy_from_cooling_device_to_cooling_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_heating_device_to_heating_storage(self) -> List[float]:
        return pd.DataFrame([b.energy_from_heating_device_to_heating_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_dhw_device_to_dhw_storage(self) -> List[float]:
        return pd.DataFrame([b.energy_from_dhw_device_to_dhw_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_to_electrical_storage(self) -> List[float]:
        return pd.DataFrame([b.energy_to_electrical_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_cooling_device(self) -> List[float]:
        return pd.DataFrame([b.energy_from_cooling_device for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_heating_device(self) -> List[float]:
        return pd.DataFrame([b.energy_from_heating_device for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_dhw_device(self) -> List[float]:
        return pd.DataFrame([b.energy_from_dhw_device for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_cooling_storage(self) -> List[float]:
        return pd.DataFrame([b.energy_from_cooling_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_heating_storage(self) -> List[float]:
        return pd.DataFrame([b.energy_from_heating_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_dhw_storage(self) -> List[float]:
        return pd.DataFrame([b.energy_from_dhw_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def energy_from_electrical_storage(self) -> List[float]:
        return pd.DataFrame([b.energy_from_electrical_storage for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def cooling_demand(self) -> List[float]:
        return pd.DataFrame([b.cooling_demand for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def heating_demand(self) -> List[float]:
        return pd.DataFrame([b.heating_demand for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def dhw_demand(self) -> List[float]:
        return pd.DataFrame([b.dhw_demand for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def non_shiftable_load_demand(self) -> List[float]:
        return pd.DataFrame([b.non_shiftable_load_demand for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @property
    def solar_generation(self) -> List[float]:
        return pd.DataFrame([b.solar_generation for b in self.buildings]).sum(axis = 0, min_count = 1).tolist()

    @buildings.setter
    def buildings(self, buildings: List[Building]):
        self.__buildings = buildings

    @time_steps.setter
    def time_steps(self, time_steps: int):
        assert time_steps >= 1, 'time_steps must be >= 1'
        self.__time_steps = time_steps

    @central_agent.setter
    def central_agent(self, central_agent: bool):
        self.__central_agent = central_agent

    @shared_states.setter
    def shared_states(self, shared_states: List[str]):
        self.__shared_states = [] if shared_states is None else shared_states

    def step(self, actions: Union[List[float] ,List[List[float]]]):
        actions = self.__parse_actions(actions)

        for building, building_actions in zip(self.buildings, actions):
            building.apply_actions(**building_actions)

        self.next_time_step()

    def __parse_actions(self, actions: Union[List[float] ,List[List[float]]]) -> List[Mapping[str, float]]:
        actions = list(actions)
        building_actions = []

        if self.central_agent:
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

    def next_time_step(self):
        for building in self.buildings:
            building.next_time_step()
            
        super().next_time_step()

    def reset(self):
        super().reset()

        for building in self.buildings:
            building.reset()

    @staticmethod
    def load(schema: Union[str, Path, Mapping[str, Any]]):
        if not isinstance(schema, dict):
            schema = read_json(schema)
            root_directory = os.path.split(schema) if schema['root_directory'] is None else schema['root_directory']
        else:
            root_directory = '' if schema['root_directory'] is None else schema['root_directory']

        central_agent = schema['central_agent']
        states = {s: v for s, v in schema['states'].items() if v['active']}
        actions = {a: v for a, v in schema['actions'].items() if v['active']}
        shared_states = [k for k, v in states.items() if v['shared_in_central_agent']]
        simulation_start_timestep = schema['simulation_start_timestep']
        simulation_end_timestep = schema['simulation_end_timestep']
        timesteps = simulation_end_timestep - simulation_start_timestep + 1
        constructors = {
            HeatPump.__name__: HeatPump,
            ElectricHeater.__name__: ElectricHeater,
            StorageTank.__name__: StorageTank,
            Battery.__name__: Battery,
            PV.__name__: PV
        }
        buildings = ()
        
        for building_name, building_schema in schema['buildings'].items():
            if building_schema['include']:
                # data
                energy_simulation = pd.read_csv(os.path.join(root_directory,building_schema['energy_simulation'])).iloc[simulation_start_timestep:simulation_end_timestep + 1].copy()
                energy_simulation = EnergySimulation(*energy_simulation.values.T)
                weather = pd.read_csv(os.path.join(root_directory,building_schema['weather'])).iloc[simulation_start_timestep:simulation_end_timestep + 1].copy()
                weather = Weather(*weather.values.T)

                if building_schema.get('carbon_intensity', None) is not None:
                    carbon_intensity = pd.read_csv(os.path.join(root_directory,building_schema['carbon_intensity'])).iloc[simulation_start_timestep:simulation_end_timestep + 1].copy()
                    carbon_intensity = carbon_intensity['kg_CO2/kWh'].tolist()
                    carbon_intensity = CarbonIntensity(carbon_intensity)
                else:
                    carbon_intensity = None

                # state and action metadata
                inactive_states = [] if building_schema.get('inactive_states', None) is None else building_schema['inactive_states']
                inactive_actions = [] if building_schema.get('inactive_actions', None) is None else building_schema['inactive_actions']
                state_metadata = {s: False if s in inactive_states else True for s in states}
                action_metadata = {a: False if a in inactive_actions else True for a in actions}

                # construct building
                building = Building(energy_simulation, weather, state_metadata, action_metadata, carbon_intensity = carbon_intensity, name = building_name)

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
                    autosizer = device_metadata[name]['autosizer']
                    device = constructors[building_schema[name]['type']](**building_schema[name]['attributes']) if building_schema.get(name, None) is not None else None
                    building.__setattr__(name, device)
                    autosize = False if building_schema.get(name, {}).get('autosize', None) is None else building_schema[name]['autosize']
                    autosize_kwargs = {} if building_schema.get(name, {}).get('autosize_kwargs', None) is None else building_schema[name]['autosize_kwargs']

                    if autosize:
                        autosizer(**autosize_kwargs)
                    else:
                        pass

                buildings += (building,)
                
            else:
                continue

        city = City(list(buildings), timesteps, central_agent = central_agent, shared_states = shared_states)
        return city