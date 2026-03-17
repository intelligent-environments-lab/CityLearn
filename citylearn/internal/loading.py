from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import os
from typing import TYPE_CHECKING, Any, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd

from citylearn.base import EpisodeTracker
from citylearn.building import Building
from citylearn.data import (
    CarbonIntensity,
    ChargerSimulation,
    DataSet,
    EnergySimulation,
    LogisticRegressionOccupantParameters,
    Pricing,
    WashingMachineSimulation,
    Weather,
)
from citylearn.electric_vehicle import ElectricVehicle
from citylearn.energy_model import Battery, PV, WashingMachine
from citylearn.reward_function import MultiBuildingRewardFunction, RewardFunction
from citylearn.utilities import parse_bool

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


@dataclass
class LoadContext:
    """Lightweight holder for loading inputs."""

    schema: Mapping[str, Any]
    kwargs: Mapping[str, Any]


class CityLearnLoadingService:
    """Internal loader service that builds env components from schema."""

    def __init__(self, env: "CityLearnEnv"):
        self.env = env

    def load(
        self,
        schema: Mapping[str, Any],
        **kwargs,
    ) -> Tuple[
        Union[os.PathLike, str],
        List[Building],
        List[ElectricVehicle],
        Union[int, List[Tuple[int, int]]],
        bool,
        bool,
        float,
        RewardFunction,
        bool,
        List[str],
        EpisodeTracker,
    ]:
        """Return env objects as defined by schema."""

        schema['root_directory'] = kwargs['root_directory'] if kwargs.get('root_directory') is not None else schema['root_directory']
        schema['random_seed'] = schema.get('random_seed', None) if kwargs.get('random_seed', None) is None else schema.get('random_seed', None)
        schema['central_agent'] = parse_bool(
            kwargs['central_agent'] if kwargs.get('central_agent') is not None else schema['central_agent'],
            default=False,
            path='central_agent',
        )

        schema['chargers_observations_helper'] = {key: value for key, value in schema["observations"].items() if "electric_vehicle_" in key}
        schema['chargers_actions_helper'] = {key: value for key, value in schema["actions"].items() if "electric_vehicle_" in key}
        schema['chargers_shared_observations_helper'] = {
            key: value
            for key, value in schema["observations"].items()
            if "electric_vehicle_" in key and value.get("shared_in_central_agent", True)
        }

        schema['washing_machine_observations_helper'] = {key: value for key, value in schema["observations"].items() if "washing_machine_" in key}
        schema['washing_machine_actions_helper'] = {key: value for key, value in schema["actions"].items() if "washing_machine" in key}

        schema['observations'] = {
            key: value
            for key, value in schema["observations"].items()
            if key not in set(schema['chargers_observations_helper']) | set(schema['washing_machine_observations_helper'])
        }
        schema['actions'] = {
            key: value
            for key, value in schema['actions'].items()
            if key not in set(schema['chargers_actions_helper']) | set(schema['washing_machine_actions_helper'])
        }

        schema['shared_observations'] = (
            kwargs['shared_observations']
            if kwargs.get('shared_observations') is not None
            else [
                k
                for k, v in schema['observations'].items()
                if not k.startswith("electric_vehicle_")
                and "washing_machine" not in k
                and parse_bool(v.get('shared_in_central_agent', False), default=False, path=f'observations.{k}.shared_in_central_agent')
            ]
        )

        schema['episode_time_steps'] = kwargs['episode_time_steps'] if kwargs.get('episode_time_steps') is not None else schema.get('episode_time_steps', None)
        schema['rolling_episode_split'] = kwargs['rolling_episode_split'] if kwargs.get('rolling_episode_split') is not None else schema.get('rolling_episode_split', None)
        schema['random_episode_split'] = kwargs['random_episode_split'] if kwargs.get('random_episode_split') is not None else schema.get('random_episode_split', None)
        schema['seconds_per_time_step'] = kwargs['seconds_per_time_step'] if kwargs.get('seconds_per_time_step') is not None else schema['seconds_per_time_step']

        schema['simulation_start_time_step'] = kwargs['simulation_start_time_step'] if kwargs.get('simulation_start_time_step') is not None else schema['simulation_start_time_step']
        schema['simulation_end_time_step'] = kwargs['simulation_end_time_step'] if kwargs.get('simulation_end_time_step') is not None else schema['simulation_end_time_step']
        episode_tracker = EpisodeTracker(schema['simulation_start_time_step'], schema['simulation_end_time_step'])

        dataset = DataSet()
        pv_sizing_data = dataset.get_pv_sizing_data()
        battery_sizing_data = dataset.get_battery_sizing_data()

        buildings_to_include = list(schema['buildings'].keys())
        buildings: List[Building] = []

        if kwargs.get('buildings') is not None and len(kwargs['buildings']) > 0:
            if isinstance(kwargs['buildings'][0], Building):
                buildings = kwargs['buildings']

                for building in buildings:
                    building.episode_tracker = episode_tracker

                buildings_to_include = []

            elif isinstance(kwargs['buildings'][0], str):
                buildings_to_include = [b for b in buildings_to_include if b in kwargs['buildings']]

            elif isinstance(kwargs['buildings'][0], int):
                buildings_to_include = [buildings_to_include[i] for i in kwargs['buildings']]

            else:
                raise Exception('Unknown buildings type. Allowed types are citylearn.building.Building, int and str.')

        else:
            buildings_to_include = [
                b for b in buildings_to_include
                if parse_bool(schema['buildings'][b].get('include', True), default=True, path=f'buildings.{b}.include')
            ]

        for i, building_name in enumerate(buildings_to_include):
            buildings.append(self.load_building(i, building_name, schema, episode_tracker, pv_sizing_data, battery_sizing_data, **kwargs))

        electric_vehicles: List[ElectricVehicle] = []
        if kwargs.get('electric_vehicles_def') is not None and len(kwargs['electric_vehicles_def']) > 0:
            electric_vehicle_schemas = kwargs['electric_vehicles_def']
        else:
            electric_vehicle_schemas = schema.get('electric_vehicles_def', {})

        for electric_vehicle_name, electric_vehicle_schema in electric_vehicle_schemas.items():
            if parse_bool(electric_vehicle_schema.get('include', True), default=True, path=f'electric_vehicles_def.{electric_vehicle_name}.include'):
                time_step_ratio = buildings[0].time_step_ratio if len(buildings) > 0 else 1.0
                electric_vehicles.append(
                    self.load_electric_vehicle(electric_vehicle_name, schema, electric_vehicle_schema, episode_tracker, time_step_ratio)
                )

        reward_schema = schema['reward_function']
        reward_type = reward_schema['type']
        reward_attrs = reward_schema.get('attributes', {})
        is_multi = isinstance(reward_type, dict)

        if is_multi:
            default_type = reward_type.get('default')
            if default_type is None and reward_type:
                default_type = next(iter(reward_type.values()))

            default_attrs = reward_attrs.get('default')
            if default_attrs is None and reward_attrs:
                default_attrs = next(iter(reward_attrs.values()))

            reward_functions = {}
            for building in buildings:
                name = building.name
                r_type = reward_type.get(name, default_type)
                r_attr = reward_attrs.get(name, default_attrs) or {}

                if r_type is None:
                    raise ValueError(f"No reward function defined for building '{name}' and no default provided")

                module_name = '.'.join(r_type.split('.')[:-1])
                class_name = r_type.split('.')[-1]
                module = importlib.import_module(module_name)
                constructor = getattr(module, class_name)
                reward_functions[name] = constructor(None, **r_attr)

            reward_function = MultiBuildingRewardFunction(None, reward_functions)

        else:
            if 'reward_function' in kwargs and kwargs['reward_function'] is not None:
                reward_function_type = kwargs['reward_function']
                if not isinstance(reward_function_type, str):
                    reward_function_type = f"{reward_function_type.__module__}.{reward_function_type.__name__}"
            else:
                reward_function_type = reward_type

            reward_function_attributes = kwargs.get('reward_function_kwargs') or reward_attrs or {}

            module_name = '.'.join(reward_function_type.split('.')[:-1])
            class_name = reward_function_type.split('.')[-1]
            module = importlib.import_module(module_name)
            constructor = getattr(module, class_name)
            reward_function = constructor(None, **reward_function_attributes)

        return (
            schema['root_directory'],
            buildings,
            electric_vehicles,
            schema['episode_time_steps'],
            schema['rolling_episode_split'],
            schema['random_episode_split'],
            schema['seconds_per_time_step'],
            reward_function,
            schema['central_agent'],
            schema['shared_observations'],
            episode_tracker,
        )

    def load_building(
        self,
        index: int,
        building_name: str,
        schema: dict,
        episode_tracker: EpisodeTracker,
        pv_sizing_data: pd.DataFrame,
        battery_sizing_data: pd.DataFrame,
        **kwargs,
    ) -> Building:
        """Initialize and return a building model."""

        building_schema = schema['buildings'][building_name]
        building_kwargs = {}
        if building_schema.get('charging_constraints') is not None:
            building_kwargs['charging_constraints'] = building_schema['charging_constraints']
        if building_schema.get('electrical_service') is not None:
            building_kwargs['electrical_service'] = building_schema['electrical_service']
        if building_schema.get('equity_group') is not None:
            building_kwargs['equity_group'] = building_schema.get('equity_group')
        electrical_storage_attributes = (building_schema.get('electrical_storage') or {}).get('attributes', {}) or {}
        if electrical_storage_attributes.get('phase_connection') is not None:
            building_kwargs['electrical_storage_phase_connection'] = electrical_storage_attributes.get('phase_connection')
        seconds_per_time_step = schema['seconds_per_time_step']
        noise_std = building_schema.get('noise_std', 0.0)

        energy_simulation = pd.read_csv(os.path.join(schema['root_directory'], building_schema['energy_simulation']))
        energy_simulation = EnergySimulation(**energy_simulation.to_dict('list'), seconds_per_time_step=seconds_per_time_step, noise_std=noise_std)
        ratios = getattr(energy_simulation, 'time_step_ratios', None) or []
        building_kwargs['time_step_ratio'] = ratios[-1] if len(ratios) > 0 else 1.0
        weather = pd.read_csv(os.path.join(schema['root_directory'], building_schema['weather']))
        weather = Weather(**weather.to_dict('list'), noise_std=noise_std)

        if building_schema.get('carbon_intensity', None) is not None:
            carbon_intensity = pd.read_csv(os.path.join(schema['root_directory'], building_schema['carbon_intensity']))
            carbon_intensity = CarbonIntensity(**carbon_intensity.to_dict('list'), noise_std=noise_std)
        else:
            carbon_intensity = CarbonIntensity(np.zeros(energy_simulation.hour.shape[0], dtype='float32'), noise_std=noise_std)

        if building_schema.get('pricing', None) is not None:
            pricing = pd.read_csv(os.path.join(schema['root_directory'], building_schema['pricing']))
            pricing = Pricing(**pricing.to_dict('list'), noise_std=noise_std)
        else:
            pricing = Pricing(
                np.zeros(energy_simulation.hour.shape[0], dtype='float32'),
                np.zeros(energy_simulation.hour.shape[0], dtype='float32'),
                np.zeros(energy_simulation.hour.shape[0], dtype='float32'),
                np.zeros(energy_simulation.hour.shape[0], dtype='float32'),
                noise_std=noise_std,
            )

        building_type = 'citylearn.citylearn.Building' if building_schema.get('type', None) is None else building_schema['type']
        building_type_module = '.'.join(building_type.split('.')[0:-1])
        building_type_name = building_type.split('.')[-1]
        building_constructor = getattr(importlib.import_module(building_type_module), building_type_name)

        if building_schema.get('dynamics', None) is not None:
            dynamics_type = building_schema['dynamics']['type']
            dynamics_module = '.'.join(dynamics_type.split('.')[0:-1])
            dynamics_name = dynamics_type.split('.')[-1]
            dynamics_constructor = getattr(importlib.import_module(dynamics_module), dynamics_name)
            attributes = building_schema['dynamics'].get('attributes', {})
            attributes['filepath'] = os.path.join(schema['root_directory'], attributes['filename'])
            _ = attributes.pop('filename')
            building_kwargs['dynamics'] = dynamics_constructor(**attributes)
        else:
            building_kwargs['dynamics'] = None

        if building_schema.get('occupant', None) is not None:
            building_occupant = building_schema['occupant']
            occupant_type = building_occupant['type']
            occupant_module = '.'.join(occupant_type.split('.')[0:-1])
            occupant_name = occupant_type.split('.')[-1]
            occupant_constructor = getattr(importlib.import_module(occupant_module), occupant_name)
            attributes: dict = building_occupant.get('attributes', {})
            parameters_filepath = os.path.join(schema['root_directory'], building_occupant['parameters_filename'])
            parameters = pd.read_csv(parameters_filepath)
            attributes['parameters'] = LogisticRegressionOccupantParameters(**parameters.to_dict('list'))
            attributes['episode_tracker'] = episode_tracker
            attributes['random_seed'] = schema['random_seed']

            for key in ['increase', 'decrease']:
                attributes[f'setpoint_{key}_model_filepath'] = os.path.join(schema['root_directory'], attributes[f'setpoint_{key}_model_filename'])
                _ = attributes.pop(f'setpoint_{key}_model_filename')

            building_kwargs['occupant'] = occupant_constructor(**attributes)
        else:
            building_kwargs['occupant'] = None

        building_schema_power_outage = building_schema.get('power_outage', {})
        simulate_power_outage = kwargs.get('simulate_power_outage')
        simulate_power_outage = building_schema_power_outage.get('simulate_power_outage') if simulate_power_outage is None else simulate_power_outage
        simulate_power_outage = simulate_power_outage[index] if isinstance(simulate_power_outage, list) else simulate_power_outage
        stochastic_power_outage = building_schema_power_outage.get('stochastic_power_outage')

        if building_schema_power_outage.get('stochastic_power_outage_model', None) is not None:
            stochastic_power_outage_model_type = building_schema_power_outage['stochastic_power_outage_model']['type']
            stochastic_power_outage_model_module = '.'.join(stochastic_power_outage_model_type.split('.')[0:-1])
            stochastic_power_outage_model_name = stochastic_power_outage_model_type.split('.')[-1]
            stochastic_power_outage_model_constructor = getattr(
                importlib.import_module(stochastic_power_outage_model_module),
                stochastic_power_outage_model_name,
            )
            attributes = building_schema_power_outage.get('stochastic_power_outage_model', {}).get('attributes', {})
            stochastic_power_outage_model = stochastic_power_outage_model_constructor(**attributes)
        else:
            stochastic_power_outage_model = None

        chargers_list = []
        if building_schema.get('chargers', None) is not None:
            for charger_name, charger_config in building_schema['chargers'].items():
                noise_std = charger_config.get('noise_std', 0.0)

                charger_simulation_file = pd.read_csv(
                    os.path.join(schema['root_directory'], charger_config['charger_simulation'])
                ).iloc[schema['simulation_start_time_step']:schema['simulation_end_time_step'] + 1].copy()

                charger_simulation = ChargerSimulation(*charger_simulation_file.values.T, noise_std=noise_std)
                if 'electric_vehicle_current_soc' in charger_simulation_file.columns:
                    current_soc_raw = pd.to_numeric(charger_simulation_file['electric_vehicle_current_soc'], errors='coerce').to_numpy(dtype='float32')
                    current_soc = np.full(current_soc_raw.shape[0], -0.1, dtype='float32')
                    valid = ~np.isnan(current_soc_raw)

                    if np.any(valid):
                        normalized = current_soc_raw[valid]
                        normalized = np.where(normalized > 1.0, normalized / 100.0, normalized)
                        normalized = np.clip(normalized, 0.0, 1.0)
                        current_soc[valid] = normalized.astype('float32')

                    charger_simulation.electric_vehicle_current_soc = current_soc

                charger_type = charger_config['type']
                charger_module = '.'.join(charger_type.split('.')[0:-1])
                charger_class_name = charger_type.split('.')[-1]
                charger_class = getattr(importlib.import_module(charger_module), charger_class_name)
                charger_attributes = dict(charger_config.get('attributes', {}) or {})
                charger_attributes['episode_tracker'] = episode_tracker
                charger_object = charger_class(
                    charger_simulation=charger_simulation,
                    charger_id=charger_name,
                    **charger_attributes,
                    seconds_per_time_step=schema['seconds_per_time_step'],
                    time_step_ratio=building_kwargs['time_step_ratio'],
                )
                chargers_list.append(charger_object)

        washing_machines_list = []
        if kwargs.get('washing_machines') is not None and len(kwargs['washing_machines']) > 0:
            washing_machine_schemas = kwargs['washing_machines']
        else:
            washing_machine_schemas = building_schema.get('washing_machines', {})

        for washing_machine_name, washing_machine_schema in washing_machine_schemas.items():
            washing_machines_list.append(self.load_washing_machine(washing_machine_name, schema, washing_machine_schema, episode_tracker))

        observation_metadata, action_metadata = self.process_metadata(
            schema,
            building_schema,
            chargers_list,
            washing_machines_list,
            index,
            energy_simulation,
            **kwargs,
        )

        building: Building = building_constructor(
            energy_simulation=energy_simulation,
            washing_machines=washing_machines_list,
            electric_vehicle_chargers=chargers_list,
            weather=weather,
            observation_metadata=observation_metadata,
            action_metadata=action_metadata,
            carbon_intensity=carbon_intensity,
            pricing=pricing,
            name=building_name,
            seconds_per_time_step=schema['seconds_per_time_step'],
            random_seed=schema['random_seed'],
            episode_tracker=episode_tracker,
            simulate_power_outage=simulate_power_outage,
            stochastic_power_outage=stochastic_power_outage,
            stochastic_power_outage_model=stochastic_power_outage_model,
            **building_kwargs,
        )

        device_metadata = {
            'cooling_device': {'autosizer': building.autosize_cooling_device},
            'heating_device': {'autosizer': building.autosize_heating_device},
            'dhw_device': {'autosizer': building.autosize_dhw_device},
            'dhw_storage': {'autosizer': building.autosize_dhw_storage},
            'cooling_storage': {'autosizer': building.autosize_cooling_storage},
            'heating_storage': {'autosizer': building.autosize_heating_storage},
            'electrical_storage': {'autosizer': building.autosize_electrical_storage},
            'washing_machine': {'autosizer': building.autosize_electrical_storage},
            'pv': {'autosizer': building.autosize_pv},
        }
        solar_generation = kwargs.get('solar_generation')
        solar_generation = True if solar_generation is None else solar_generation
        solar_generation = solar_generation[index] if isinstance(solar_generation, list) else solar_generation

        for device_name in device_metadata:
            if building_schema.get(device_name, None) is None:
                device = None

            elif device_name == 'pv' and not solar_generation:
                device = None

            else:
                device_type: str = building_schema[device_name]['type']
                device_module = '.'.join(device_type.split('.')[0:-1])
                device_type_name = device_type.split('.')[-1]
                constructor = getattr(importlib.import_module(device_module), device_type_name)
                attributes = dict(building_schema[device_name].get('attributes', {}) or {})
                if device_name == 'electrical_storage':
                    attributes.pop('phase_connection', None)
                attributes['seconds_per_time_step'] = schema['seconds_per_time_step']

                md5 = hashlib.md5()
                device_random_seed = 0

                for string in [building_name, building_type, device_name, device_type]:
                    md5.update(string.encode())
                    hash_to_integer_base = 16
                    device_random_seed += int(md5.hexdigest(), hash_to_integer_base)

                device_random_seed = int(str(device_random_seed * (schema['random_seed'] + 1))[:9])

                attributes = {
                    **attributes,
                    'random_seed': attributes['random_seed'] if attributes.get('random_seed', None) is not None else device_random_seed,
                }
                device = constructor(**attributes)
                autosize = parse_bool(
                    building_schema[device_name].get('autosize', False),
                    default=False,
                    path=f'buildings.{building.name}.{device_name}.autosize',
                )
                building.__setattr__(device_name, device)

                if autosize:
                    autosizer = device_metadata[device_name]['autosizer']
                    autosize_kwargs = {} if building_schema[device_name].get('autosize_attributes', None) is None else building_schema[device_name]['autosize_attributes']

                    if isinstance(device, PV):
                        autosize_kwargs['epw_filepath'] = os.path.join(schema['root_directory'], autosize_kwargs['epw_filepath'])
                        autosize_kwargs['sizing_data'] = pv_sizing_data

                    elif isinstance(device, Battery):
                        autosize_kwargs['sizing_data'] = battery_sizing_data

                    autosizer(**autosize_kwargs)

                device.random_seed = schema['random_seed']

        building.observation_space = building.estimate_observation_space()
        building.action_space = building.estimate_action_space()

        return building

    def process_metadata(
        self,
        schema,
        building_schema,
        chargers_list,
        washing_machines_list,
        index,
        energy_simulation: EnergySimulation,
        **kwargs,
    ):
        """Build observation and action metadata for one building."""

        observation_metadata = {
            k: parse_bool(v.get('active', False), default=False, path=f'observations.{k}.active')
            for k, v in schema['observations'].items()
        }
        if 'minutes' in observation_metadata and energy_simulation.minutes is None:
            observation_metadata.pop('minutes', None)

        chargers_observations_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'observations.{k}.active')
            for k, v in schema['chargers_observations_helper'].items()
        }
        washing_machine_observations_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'observations.{k}.active')
            for k, v in schema['washing_machine_observations_helper'].items()
        }

        if kwargs.get('active_observations') is not None:
            active_observations = kwargs['active_observations']
            active_observations = active_observations[index] if isinstance(active_observations[0], list) else active_observations
            observation_metadata = {k: True if k in active_observations else False for k in observation_metadata}
            chargers_observations_metadata_helper = {k: True if k in active_observations else False for k in chargers_observations_metadata_helper}
            washing_machine_observations_metadata_helper = {k: True if k in active_observations else False for k in washing_machine_observations_metadata_helper}

        if kwargs.get('inactive_observations') is not None:
            inactive_observations = kwargs['inactive_observations']
            inactive_observations = inactive_observations[index] if isinstance(inactive_observations[0], list) else inactive_observations
        elif building_schema.get('inactive_observations') is not None:
            inactive_observations = building_schema['inactive_observations']
        else:
            inactive_observations = []

        observation_metadata = {
            k: False if k in inactive_observations else observation_metadata[k]
            for k in observation_metadata
        }
        chargers_observations_metadata_helper = {
            k: False if k in inactive_observations else chargers_observations_metadata_helper[k]
            for k in chargers_observations_metadata_helper
        }
        washing_machine_observations_metadata_helper = {
            k: False if k in inactive_observations else washing_machine_observations_metadata_helper[k]
            for k in washing_machine_observations_metadata_helper
        }

        action_metadata = {
            k: parse_bool(v.get('active', False), default=False, path=f'actions.{k}.active')
            for k, v in schema['actions'].items()
        }
        chargers_actions_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'actions.{k}.active')
            for k, v in schema['chargers_actions_helper'].items()
        }
        washing_machine_actions_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'actions.{k}.active')
            for k, v in schema['washing_machine_actions_helper'].items()
        }

        if kwargs.get('active_actions') is not None:
            active_actions = kwargs['active_actions']
            active_actions = active_actions[index] if isinstance(active_actions[0], list) else active_actions
            action_metadata = {k: True if k in active_actions else False for k in action_metadata}
            chargers_actions_metadata_helper = {k: True if k in active_actions else False for k in chargers_actions_metadata_helper}
            washing_machine_actions_metadata_helper = {k: True if k in active_actions else False for k in washing_machine_actions_metadata_helper}

        if kwargs.get('inactive_actions') is not None:
            inactive_actions = kwargs['inactive_actions']
            inactive_actions = inactive_actions[index] if isinstance(inactive_actions[0], list) else inactive_actions
        elif building_schema.get('inactive_actions') is not None:
            inactive_actions = building_schema['inactive_actions']
        else:
            inactive_actions = []

        action_metadata = {k: False if k in inactive_actions else v for k, v in action_metadata.items()}
        chargers_actions_metadata_helper = {k: False if k in inactive_actions else v for k, v in chargers_actions_metadata_helper.items()}
        washing_machine_actions_metadata_helper = {k: False if k in inactive_actions else v for k, v in washing_machine_actions_metadata_helper.items()}

        if len(chargers_list) > 0:
            for charger in chargers_list:
                charger_id = charger.charger_id

                if chargers_observations_metadata_helper.get('electric_vehicle_charger_connected_state', False):
                    observation_metadata[f'electric_vehicle_charger_{charger_id}_connected_state'] = True

                if chargers_observations_metadata_helper.get('connected_electric_vehicle_at_charger_departure_time', False):
                    observation_metadata[f'connected_electric_vehicle_at_charger_{charger_id}_departure_time'] = True

                if chargers_observations_metadata_helper.get('connected_electric_vehicle_at_charger_required_soc_departure', False):
                    observation_metadata[f'connected_electric_vehicle_at_charger_{charger_id}_required_soc_departure'] = True

                if chargers_observations_metadata_helper.get('connected_electric_vehicle_at_charger_soc', False):
                    observation_metadata[f'connected_electric_vehicle_at_charger_{charger_id}_soc'] = True

                if chargers_observations_metadata_helper.get('connected_electric_vehicle_at_charger_battery_capacity', False):
                    observation_metadata[f'connected_electric_vehicle_at_charger_{charger_id}_battery_capacity'] = True

                if chargers_observations_metadata_helper.get('electric_vehicle_charger_incoming_state', False):
                    observation_metadata[f'electric_vehicle_charger_{charger_id}_incoming_state'] = True

                if chargers_observations_metadata_helper.get('incoming_electric_vehicle_at_charger_estimated_arrival_time', False):
                    observation_metadata[f'incoming_electric_vehicle_at_charger_{charger_id}_estimated_arrival_time'] = True

                if chargers_observations_metadata_helper.get('incoming_electric_vehicle_at_charger_estimated_soc_arrival', False):
                    observation_metadata[f'incoming_electric_vehicle_at_charger_{charger_id}_estimated_soc_arrival'] = True

                if chargers_actions_metadata_helper.get('electric_vehicle_storage', False):
                    action_metadata[f'electric_vehicle_storage_{charger.charger_id}'] = True

        if len(washing_machines_list) > 0:
            for washing_machine in washing_machines_list:
                washing_machine_name = washing_machine.name
                if washing_machine_observations_metadata_helper.get('washing_machine_start_time_step', False):
                    observation_metadata[f'{washing_machine_name}_start_time_step'] = True

                if washing_machine_observations_metadata_helper.get('washing_machine_end_time_step', False):
                    observation_metadata[f'{washing_machine_name}_end_time_step'] = True

                if washing_machine_actions_metadata_helper.get('washing_machine', False):
                    action_metadata[f'{washing_machine_name}'] = True

        return observation_metadata, action_metadata

    def load_electric_vehicle(
        self,
        electric_vehicle_name: str,
        schema: dict,
        electric_vehicle_schema: dict,
        episode_tracker: EpisodeTracker,
        time_step_ratio,
    ) -> ElectricVehicle:
        """Initialize and return an electric vehicle model."""

        capacity = electric_vehicle_schema['battery']['attributes']['capacity']
        nominal_power = electric_vehicle_schema['battery']['attributes']['nominal_power']
        initial_soc = electric_vehicle_schema['battery']['attributes'].get('initial_soc')
        if initial_soc is None:
            seed_source = f"{schema['random_seed']}:{electric_vehicle_name}:initial_soc"
            deterministic_seed = int(hashlib.md5(seed_source.encode('utf-8')).hexdigest()[:8], 16)
            initial_soc = float(np.random.RandomState(deterministic_seed).uniform(0.0, 1.0))
        depth_of_discharge = electric_vehicle_schema['battery']['attributes'].get('depth_of_discharge', 0.10)

        battery = Battery(
            capacity=capacity,
            nominal_power=nominal_power,
            initial_soc=initial_soc,
            seconds_per_time_step=schema['seconds_per_time_step'],
            time_step_ratio=time_step_ratio,
            random_seed=schema['random_seed'],
            episode_tracker=episode_tracker,
            depth_of_discharge=depth_of_discharge,
        )

        electric_vehicle_type = 'citylearn.citylearn.ElectricVehicle' if electric_vehicle_schema.get('type', None) is None else electric_vehicle_schema['type']
        electric_vehicle_type_module = '.'.join(electric_vehicle_type.split('.')[0:-1])
        electric_vehicle_type_name = electric_vehicle_type.split('.')[-1]
        electric_vehicle_constructor = getattr(importlib.import_module(electric_vehicle_type_module), electric_vehicle_type_name)

        electric_vehicle: ElectricVehicle = electric_vehicle_constructor(
            battery=battery,
            name=electric_vehicle_name,
            seconds_per_time_step=schema['seconds_per_time_step'],
            random_seed=schema['random_seed'],
            episode_tracker=episode_tracker,
        )

        return electric_vehicle

    def load_washing_machine(
        self,
        washing_machine_name: str,
        schema: dict,
        washing_machine_schema: dict,
        episode_tracker: EpisodeTracker,
    ) -> WashingMachine:
        """Load simulation data and initialize a `WashingMachine` instance."""

        file_path = os.path.join(schema['root_directory'], washing_machine_schema['washing_machine_energy_simulation'])

        washing_machine_simulation = pd.read_csv(file_path).iloc[
            schema['simulation_start_time_step']:schema['simulation_end_time_step'] + 1
        ].copy()

        washing_machine_simulation = WashingMachineSimulation(*washing_machine_simulation.values.T)

        washing_machine = WashingMachine(
            washing_machine_simulation=washing_machine_simulation,
            episode_tracker=episode_tracker,
            name=washing_machine_name,
            seconds_per_time_step=schema['seconds_per_time_step'],
            random_seed=schema['random_seed'],
        )

        return washing_machine
