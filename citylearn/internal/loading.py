from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import os
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Set, Tuple, Union

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
    DeferrableApplianceSimulation,
    EscalatorSimulation,
    Weather,
)
from citylearn.electric_vehicle import ElectricVehicle
from citylearn.energy_model import Battery, DeferrableAppliance, Escalator, PV
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
        self._dynamic_mode: bool = False
        self._dynamic_expected_rows: int = 0
        self._dynamic_member_windows: Dict[str, Tuple[int, int]] = {}
        self._dynamic_charger_windows: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._dynamic_deferrable_appliance_windows: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._shared_timeseries_cache: Dict[Tuple[Any, ...], Any] = {}

    @staticmethod
    def _dataframe_to_constructor_kwargs(dataframe: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return column arrays without boxing every value into Python lists."""

        return {
            str(column): series.to_numpy(copy=False)
            for column, series in dataframe.items()
        }

    @staticmethod
    def _print_time_step_conversion_notice(
        *,
        building_name: str,
        source: str,
        dataset_seconds_per_time_step,
        seconds_per_time_step,
        time_step_ratio,
    ):
        """Print an explicit notice when runtime energy conversions are active."""

        try:
            ratio = float(time_step_ratio)
        except (TypeError, ValueError):
            return

        if not np.isfinite(ratio) or abs(ratio - 1.0) <= 1.0e-9:
            return

        dataset_seconds = "unknown" if dataset_seconds_per_time_step is None else f"{float(dataset_seconds_per_time_step):g}"
        schema_seconds = "unknown" if seconds_per_time_step is None else f"{float(seconds_per_time_step):g}"
        print(
            "[CityLearn][unit-conversion] "
            f"Building '{building_name}' dataset resolution differs from schema/control step: "
            f"dataset_seconds_per_time_step={dataset_seconds}, "
            f"seconds_per_time_step={schema_seconds}, "
            f"time_step_ratio={ratio:g}. "
            "No automatic resampling is performed. Keep schema seconds_per_time_step aligned with the dataset unless "
            "you intentionally want runtime energy conversion through time_step_ratio. "
            f"source={source}"
        )

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
        schema['random_seed'] = schema.get('random_seed', None) if kwargs.get('random_seed', None) is None else kwargs['random_seed']
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

        schema['deferrable_appliance_observations_helper'] = {
            key: value for key, value in schema["observations"].items()
            if key.startswith("deferrable_appliance_")
        }
        schema['deferrable_appliance_actions_helper'] = {
            key: value for key, value in schema["actions"].items()
            if key == "deferrable_appliance" or key.startswith("deferrable_appliance_")
        }
        schema['escalator_observations_helper'] = {
            key: value for key, value in schema['observations'].items()
            if key.startswith('escalator_')
        }
        schema['escalator_actions_helper'] = {
            key: value for key, value in schema['actions'].items()
            if key == 'escalator' or key.startswith('escalator_')
        }

        schema['observations'] = {
            key: value
            for key, value in schema["observations"].items()
            if key not in (
                set(schema['chargers_observations_helper'])
                | set(schema['deferrable_appliance_observations_helper'])
                | set(schema['escalator_observations_helper'])
            )
        }
        schema['actions'] = {
            key: value
            for key, value in schema['actions'].items()
            if key not in (
                set(schema['chargers_actions_helper'])
                | set(schema['deferrable_appliance_actions_helper'])
                | set(schema['escalator_actions_helper'])
            )
        }

        schema['shared_observations'] = (
            kwargs['shared_observations']
            if kwargs.get('shared_observations') is not None
            else [
                k
                for k, v in schema['observations'].items()
                if not k.startswith("electric_vehicle_")
                and not k.startswith("deferrable_appliance_")
                and not k.startswith("escalator_")
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

        pv_sizing_data = None
        battery_sizing_data = None

        dynamic_mode = getattr(self.env, 'topology_mode', 'static') == 'dynamic'
        self._dynamic_mode = dynamic_mode
        self._dynamic_expected_rows = int(episode_tracker.simulation_time_steps)
        self._dynamic_member_windows = {}
        self._dynamic_charger_windows = {}
        self._dynamic_deferrable_appliance_windows = {}
        self._shared_timeseries_cache = {}

        if dynamic_mode:
            self._dynamic_member_windows = self._build_dynamic_member_windows(schema, self._dynamic_expected_rows)
            self._dynamic_charger_windows = self._build_dynamic_charger_windows(
                schema,
                self._dynamic_member_windows,
                self._dynamic_expected_rows,
            )
            self._dynamic_deferrable_appliance_windows = self._build_dynamic_deferrable_appliance_windows(
                schema,
                self._dynamic_member_windows,
                self._dynamic_expected_rows,
            )

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
            if dynamic_mode:
                # Dynamic topology needs both initially active members and potential templates.
                buildings_to_include = list(schema['buildings'].keys())
            else:
                buildings_to_include = [
                    b for b in buildings_to_include
                    if parse_bool(schema['buildings'][b].get('include', True), default=True, path=f'buildings.{b}.include')
                ]

        if len(buildings_to_include) > 0:
            solar_generation = kwargs.get('solar_generation')
            solar_generation = True if solar_generation is None else solar_generation

            def _is_solar_generation_enabled(index: int) -> bool:
                if isinstance(solar_generation, list):
                    return bool(
                        parse_bool(
                            solar_generation[index],
                            default=True,
                            path=f'solar_generation[{index}]',
                        )
                    )

                return bool(parse_bool(solar_generation, default=True, path='solar_generation'))

            require_pv_sizing_data = False
            require_battery_sizing_data = False

            for i, building_name in enumerate(buildings_to_include):
                building_schema = schema['buildings'][building_name]
                pv_schema = building_schema.get('pv') or {}
                electrical_storage_schema = building_schema.get('electrical_storage') or {}
                pv_autosize = parse_bool(
                    pv_schema.get('autosize', False),
                    default=False,
                    path=f'buildings.{building_name}.pv.autosize',
                )
                battery_autosize = parse_bool(
                    electrical_storage_schema.get('autosize', False),
                    default=False,
                    path=f'buildings.{building_name}.electrical_storage.autosize',
                )
                require_pv_sizing_data = require_pv_sizing_data or (pv_autosize and _is_solar_generation_enabled(i))
                require_battery_sizing_data = require_battery_sizing_data or battery_autosize

                if require_pv_sizing_data and require_battery_sizing_data:
                    break

            if require_pv_sizing_data or require_battery_sizing_data:
                dataset = DataSet(offline=self.env.offline)

                if require_pv_sizing_data:
                    pv_sizing_data = dataset.get_pv_sizing_data()

                if require_battery_sizing_data:
                    battery_sizing_data = dataset.get_battery_sizing_data()

        for i, building_name in enumerate(buildings_to_include):
            buildings.append(self.load_building(i, building_name, schema, episode_tracker, pv_sizing_data, battery_sizing_data, **kwargs))

        if not dynamic_mode:
            self._validate_static_action_asset_consistency(buildings)

        electric_vehicles: List[ElectricVehicle] = []
        if kwargs.get('electric_vehicles_def') is not None and len(kwargs['electric_vehicles_def']) > 0:
            electric_vehicle_schemas = kwargs['electric_vehicles_def']
        else:
            electric_vehicle_schemas = schema.get('electric_vehicles_def', {})

        for electric_vehicle_name, electric_vehicle_schema in electric_vehicle_schemas.items():
            include_ev = True if dynamic_mode else parse_bool(
                electric_vehicle_schema.get('include', True),
                default=True,
                path=f'electric_vehicles_def.{electric_vehicle_name}.include',
            )
            if include_ev:
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
        expected_rows = int(getattr(self, '_dynamic_expected_rows', episode_tracker.simulation_time_steps))
        member_window = self._dynamic_member_windows.get(building_name)

        energy_simulation_filepath = os.path.join(schema['root_directory'], building_schema['energy_simulation'])
        energy_simulation = self._read_simulation_dataframe(schema, energy_simulation_filepath)
        energy_simulation = self._align_dynamic_timeseries_dataframe(
            energy_simulation,
            expected_rows=expected_rows,
            window=member_window,
            source_label=f'buildings.{building_name}.energy_simulation',
        )
        energy_simulation = EnergySimulation(
            **self._dataframe_to_constructor_kwargs(energy_simulation),
            seconds_per_time_step=seconds_per_time_step,
            noise_std=noise_std,
        )
        self._set_time_step_offset(energy_simulation, schema['simulation_start_time_step'])
        ratios = getattr(energy_simulation, 'time_step_ratios', None) or []
        building_kwargs['time_step_ratio'] = ratios[-1] if len(ratios) > 0 else 1.0
        self._print_time_step_conversion_notice(
            building_name=building_name,
            source=energy_simulation_filepath,
            dataset_seconds_per_time_step=getattr(energy_simulation, 'dataset_seconds_per_time_step', None),
            seconds_per_time_step=seconds_per_time_step,
            time_step_ratio=building_kwargs['time_step_ratio'],
        )
        weather_filepath = os.path.join(schema['root_directory'], building_schema['weather'])
        weather = self._load_shared_timeseries(
            Weather,
            schema=schema,
            filepath=weather_filepath,
            expected_rows=expected_rows,
            window=member_window,
            source_label=f'buildings.{building_name}.weather',
            noise_std=noise_std,
        )

        if building_schema.get('carbon_intensity', None) is not None:
            carbon_intensity_filepath = os.path.join(schema['root_directory'], building_schema['carbon_intensity'])
            carbon_intensity = self._load_shared_timeseries(
                CarbonIntensity,
                schema=schema,
                filepath=carbon_intensity_filepath,
                expected_rows=expected_rows,
                window=member_window,
                source_label=f'buildings.{building_name}.carbon_intensity',
                noise_std=noise_std,
            )
        else:
            carbon_intensity = CarbonIntensity(np.zeros(energy_simulation.hour.shape[0], dtype='float32'), noise_std=noise_std)
            self._set_time_step_offset(carbon_intensity, schema['simulation_start_time_step'])

        if building_schema.get('pricing', None) is not None:
            pricing_filepath = os.path.join(schema['root_directory'], building_schema['pricing'])
            pricing = self._load_shared_timeseries(
                Pricing,
                schema=schema,
                filepath=pricing_filepath,
                expected_rows=expected_rows,
                window=member_window,
                source_label=f'buildings.{building_name}.pricing',
                noise_std=noise_std,
            )
        else:
            pricing = Pricing(
                np.zeros(energy_simulation.hour.shape[0], dtype='float32'),
                np.zeros(energy_simulation.hour.shape[0], dtype='float32'),
                np.zeros(energy_simulation.hour.shape[0], dtype='float32'),
                np.zeros(energy_simulation.hour.shape[0], dtype='float32'),
                noise_std=noise_std,
            )
            self._set_time_step_offset(pricing, schema['simulation_start_time_step'])

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

                charger_simulation_filepath = os.path.join(schema['root_directory'], charger_config['charger_simulation'])
                charger_simulation_file = self._read_simulation_dataframe(schema, charger_simulation_filepath)
                charger_window = self._dynamic_charger_windows.get((building_name, charger_name))
                charger_simulation_file = self._align_dynamic_timeseries_dataframe(
                    charger_simulation_file,
                    expected_rows=expected_rows,
                    window=charger_window,
                    source_label=f'buildings.{building_name}.chargers.{charger_name}.charger_simulation',
                )

                # Keep the constructor contract stable when charger datasets add
                # optional telemetry columns.  Passing the complete dataframe
                # positionally would silently reinterpret a seventh column as
                # ``start_time_step``.
                charger_columns = (
                    'electric_vehicle_charger_state',
                    'electric_vehicle_id',
                    'electric_vehicle_departure_time',
                    'electric_vehicle_required_soc_departure',
                    'electric_vehicle_estimated_arrival_time',
                    'electric_vehicle_estimated_soc_arrival',
                )
                missing_charger_columns = [
                    column for column in charger_columns
                    if column not in charger_simulation_file.columns
                ]
                if missing_charger_columns:
                    raise ValueError(
                        f"Charger simulation '{charger_simulation_filepath}' is missing "
                        f"required columns: {missing_charger_columns}."
                    )
                charger_simulation = ChargerSimulation(
                    *(charger_simulation_file[column].to_numpy() for column in charger_columns),
                    electric_vehicle_session_id=(
                        charger_simulation_file['electric_vehicle_session_id'].to_numpy()
                        if 'electric_vehicle_session_id' in charger_simulation_file.columns
                        else None
                    ),
                    noise_std=noise_std,
                )
                self._set_time_step_offset(charger_simulation, schema['simulation_start_time_step'])
                if 'electric_vehicle_current_soc' in charger_simulation_file.columns:
                    current_soc_raw = pd.to_numeric(charger_simulation_file['electric_vehicle_current_soc'], errors='coerce').to_numpy(dtype='float32')
                    charger_simulation.electric_vehicle_current_soc = ChargerSimulation.normalize_soc_series(
                        current_soc_raw,
                        default_soc_value=-0.1,
                        noise_std=0.0,
                    )

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

        deferrable_appliances_list = []
        if kwargs.get('deferrable_appliances') is not None and len(kwargs['deferrable_appliances']) > 0:
            deferrable_appliance_schemas = kwargs['deferrable_appliances']
        else:
            deferrable_appliance_schemas = building_schema.get('deferrable_appliances', {})

        for appliance_name, appliance_schema in deferrable_appliance_schemas.items():
            deferrable_appliances_list.append(
                self.load_deferrable_appliance(
                    appliance_name,
                    building_name,
                    schema,
                    appliance_schema,
                    episode_tracker,
                )
            )

        escalators_list = []
        for escalator_name, escalator_schema in (building_schema.get('escalators', {}) or {}).items():
            escalators_list.append(
                self.load_escalator(
                    escalator_name,
                    building_name,
                    schema,
                    escalator_schema,
                    episode_tracker,
                    expected_rows=expected_rows,
                    window=member_window,
                    time_step_ratio=building_kwargs['time_step_ratio'],
                )
            )

        declared_inactive_actions = self._resolve_inactive_actions(index, building_schema, kwargs)
        observation_metadata, action_metadata = self.process_metadata(
            schema,
            building_schema,
            chargers_list,
            deferrable_appliances_list,
            escalators_list,
            index,
            energy_simulation,
            **kwargs,
        )

        building: Building = building_constructor(
            energy_simulation=energy_simulation,
            deferrable_appliances=deferrable_appliances_list,
            escalators=escalators_list,
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
        building._declared_inactive_actions = set(declared_inactive_actions)

        device_metadata = {
            'cooling_device': {'autosizer': building.autosize_cooling_device},
            'heating_device': {'autosizer': building.autosize_heating_device},
            'dhw_device': {'autosizer': building.autosize_dhw_device},
            'dhw_storage': {'autosizer': building.autosize_dhw_storage},
            'cooling_storage': {'autosizer': building.autosize_cooling_storage},
            'heating_storage': {'autosizer': building.autosize_heating_storage},
            'electrical_storage': {'autosizer': building.autosize_electrical_storage},
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
        deferrable_appliances_list,
        escalators_list,
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
        if 'seconds' in observation_metadata and getattr(energy_simulation, 'seconds', None) is None:
            observation_metadata.pop('seconds', None)

        chargers_observations_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'observations.{k}.active')
            for k, v in schema['chargers_observations_helper'].items()
        }
        deferrable_appliance_observations_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'observations.{k}.active')
            for k, v in schema['deferrable_appliance_observations_helper'].items()
        }
        escalator_observations_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'observations.{k}.active')
            for k, v in schema['escalator_observations_helper'].items()
        }

        if kwargs.get('active_observations') is not None:
            active_observations = kwargs['active_observations']
            active_observations = active_observations[index] if isinstance(active_observations[0], list) else active_observations
            observation_metadata = {k: True if k in active_observations else False for k in observation_metadata}
            chargers_observations_metadata_helper = {k: True if k in active_observations else False for k in chargers_observations_metadata_helper}
            deferrable_appliance_observations_metadata_helper = {k: True if k in active_observations else False for k in deferrable_appliance_observations_metadata_helper}
            escalator_observations_metadata_helper = {k: True if k in active_observations else False for k in escalator_observations_metadata_helper}

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
        deferrable_appliance_observations_metadata_helper = {
            k: False if k in inactive_observations else deferrable_appliance_observations_metadata_helper[k]
            for k in deferrable_appliance_observations_metadata_helper
        }
        escalator_observations_metadata_helper = {
            k: False if k in inactive_observations else escalator_observations_metadata_helper[k]
            for k in escalator_observations_metadata_helper
        }

        action_metadata = {
            k: parse_bool(v.get('active', False), default=False, path=f'actions.{k}.active')
            for k, v in schema['actions'].items()
        }
        chargers_actions_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'actions.{k}.active')
            for k, v in schema['chargers_actions_helper'].items()
        }
        deferrable_appliance_actions_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'actions.{k}.active')
            for k, v in schema['deferrable_appliance_actions_helper'].items()
        }
        escalator_actions_metadata_helper = {
            k: parse_bool(v.get('active', False), default=False, path=f'actions.{k}.active')
            for k, v in schema['escalator_actions_helper'].items()
        }

        if kwargs.get('active_actions') is not None:
            active_actions = kwargs['active_actions']
            active_actions = active_actions[index] if isinstance(active_actions[0], list) else active_actions
            action_metadata = {k: True if k in active_actions else False for k in action_metadata}
            chargers_actions_metadata_helper = {k: True if k in active_actions else False for k in chargers_actions_metadata_helper}
            deferrable_appliance_actions_metadata_helper = {k: True if k in active_actions else False for k in deferrable_appliance_actions_metadata_helper}
            escalator_actions_metadata_helper = {k: True if k in active_actions else False for k in escalator_actions_metadata_helper}

        inactive_actions = self._resolve_inactive_actions(index, building_schema, kwargs)

        action_metadata = {k: False if k in inactive_actions else v for k, v in action_metadata.items()}
        chargers_actions_metadata_helper = {k: False if k in inactive_actions else v for k, v in chargers_actions_metadata_helper.items()}
        deferrable_appliance_actions_metadata_helper = {k: False if k in inactive_actions else v for k, v in deferrable_appliance_actions_metadata_helper.items()}
        escalator_actions_metadata_helper = {k: False if k in inactive_actions else v for k, v in escalator_actions_metadata_helper.items()}

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

        if len(deferrable_appliances_list) > 0:
            for appliance in deferrable_appliances_list:
                for helper_name, active in deferrable_appliance_observations_metadata_helper.items():
                    if active:
                        feature_name = helper_name.replace('deferrable_appliance_', '', 1)
                        observation_metadata[f'deferrable_appliance_{appliance.name}_{feature_name}'] = True

                if deferrable_appliance_actions_metadata_helper.get('deferrable_appliance', False):
                    action_metadata[f'deferrable_appliance_{appliance.name}'] = True

        if len(escalators_list) > 0:
            for escalator in escalators_list:
                for helper_name, active in escalator_observations_metadata_helper.items():
                    if active:
                        feature_name = helper_name.replace('escalator_', '', 1)
                        observation_metadata[f'escalator_{escalator.name}_{feature_name}'] = True

                if escalator_actions_metadata_helper.get('escalator', False):
                    action_metadata[f'escalator_{escalator.name}'] = True

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

        attrs = electric_vehicle_schema['battery']['attributes']
        capacity = attrs['capacity']
        nominal_power = attrs['nominal_power']
        initial_soc = attrs.get('initial_soc')
        if initial_soc is None:
            seed_source = f"{schema['random_seed']}:{electric_vehicle_name}:initial_soc"
            deterministic_seed = int(hashlib.md5(seed_source.encode('utf-8')).hexdigest()[:8], 16)
            initial_soc = float(np.random.RandomState(deterministic_seed).uniform(0.0, 1.0))
        depth_of_discharge = attrs.get('depth_of_discharge', 1.0)
        loss_coefficient = attrs.get('loss_coefficient', 0.0)

        if loss_coefficient is None:
            loss_coefficient = 0.0
        elif isinstance(loss_coefficient, (list, tuple)):
            loss_coefficient = tuple(float(value) for value in loss_coefficient)
        else:
            loss_coefficient = float(loss_coefficient)

        battery_kwargs = {
            'capacity': capacity,
            'nominal_power': nominal_power,
            'initial_soc': initial_soc,
            'loss_coefficient': loss_coefficient,
            'seconds_per_time_step': schema['seconds_per_time_step'],
            'time_step_ratio': time_step_ratio,
            'random_seed': schema['random_seed'],
            'episode_tracker': episode_tracker,
            'depth_of_discharge': depth_of_discharge,
        }

        for attribute in [
            'efficiency',
            'capacity_loss_coefficient',
            'power_efficiency_curve',
            'capacity_power_curve',
        ]:
            if attribute in attrs:
                battery_kwargs[attribute] = attrs[attribute]

        battery = Battery(
            **battery_kwargs,
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

    def load_deferrable_appliance(
        self,
        appliance_name: str,
        building_name: str,
        schema: dict,
        appliance_schema: dict,
        episode_tracker: EpisodeTracker,
    ) -> DeferrableAppliance:
        """Load sparse profile/schedule data and initialize a deferrable appliance."""

        profiles_file = os.path.join(schema['root_directory'], appliance_schema['cycle_profiles_file'])
        schedule_file = os.path.join(schema['root_directory'], appliance_schema['flexibility_schedule_file'])

        profiles = self._read_dataframe(profiles_file)
        schedule = self._read_dataframe(schedule_file)
        window = self._dynamic_deferrable_appliance_windows.get((building_name, appliance_name))
        if self._dynamic_mode and window is not None and not schedule.empty:
            start, end = window
            schedule = schedule[
                (pd.to_numeric(schedule['deadline_time_step'], errors='coerce') >= int(start))
                & (pd.to_numeric(schedule['earliest_start_time_step'], errors='coerce') < int(end))
            ].reset_index(drop=True)

        source_label = f'buildings.{building_name}.deferrable_appliances.{appliance_name}'
        simulation = DeferrableApplianceSimulation.from_dataframes(
            cycle_profiles=profiles,
            flexibility_schedule=schedule,
            source_label=source_label,
        )

        appliance_type = appliance_schema.get('type', 'citylearn.energy_model.DeferrableAppliance')
        appliance_module = '.'.join(appliance_type.split('.')[0:-1])
        appliance_class_name = appliance_type.split('.')[-1]
        appliance_class = getattr(importlib.import_module(appliance_module), appliance_class_name)
        attributes = dict(appliance_schema.get('attributes', {}) or {})

        appliance = appliance_class(
            deferrable_appliance_simulation=simulation,
            episode_tracker=episode_tracker,
            name=appliance_name,
            seconds_per_time_step=schema['seconds_per_time_step'],
            random_seed=schema['random_seed'],
            **attributes,
        )

        return appliance

    def load_escalator(
        self,
        escalator_name: str,
        building_name: str,
        schema: dict,
        escalator_schema: dict,
        episode_tracker: EpisodeTracker,
        *,
        expected_rows: int,
        window: Optional[Tuple[int, int]],
        time_step_ratio: float,
    ) -> Escalator:
        """Load one aggregate escalator demand series and operating specification."""

        simulation_file = escalator_schema.get('simulation') or escalator_schema.get('simulation_file')
        if not simulation_file:
            raise ValueError(f'buildings.{building_name}.escalators.{escalator_name}.simulation is required.')
        filepath = os.path.join(schema['root_directory'], simulation_file)
        dataframe = self._read_simulation_dataframe(schema, filepath)
        dataframe = self._align_dynamic_timeseries_dataframe(
            dataframe,
            expected_rows=expected_rows,
            window=window,
            source_label=f'buildings.{building_name}.escalators.{escalator_name}.simulation',
        )
        source_label = f'buildings.{building_name}.escalators.{escalator_name}'
        simulation = EscalatorSimulation.from_dataframe(dataframe, source_label=source_label)
        self._set_time_step_offset(simulation, schema['simulation_start_time_step'])

        escalator_type = escalator_schema.get('type', 'citylearn.energy_model.Escalator')
        module_name = '.'.join(escalator_type.split('.')[:-1])
        class_name = escalator_type.split('.')[-1]
        constructor = getattr(importlib.import_module(module_name), class_name)
        attributes = dict(escalator_schema.get('attributes', {}) or {})
        return constructor(
            escalator_simulation=simulation,
            episode_tracker=episode_tracker,
            name=escalator_name,
            seconds_per_time_step=schema['seconds_per_time_step'],
            time_step_ratio=time_step_ratio,
            random_seed=schema['random_seed'],
            **attributes,
        )

    def load_washing_machine(self, *args, **kwargs):
        raise ValueError(
            "Legacy 'washing_machines' schemas are no longer supported. "
            "Use 'deferrable_appliances' with cycle_profiles_file and flexibility_schedule_file."
        )

    @staticmethod
    def _set_time_step_offset(time_series_data: Any, offset: int):
        """Record the original file index represented by row 0 after windowed loading."""

        setattr(time_series_data, 'time_step_offset', int(offset or 0))

    @staticmethod
    def _file_format(filepath: Union[str, os.PathLike]) -> str:
        suffix = os.path.splitext(str(filepath))[1].lower()
        if suffix in {'.parquet', '.pq', '.parq'}:
            return 'parquet'

        return 'csv'

    def _read_simulation_dataframe(self, schema: Mapping[str, Any], filepath: Union[str, os.PathLike]) -> pd.DataFrame:
        """Read only the configured simulation window from a CSV or Parquet time series file."""

        dataframe = self._read_timeseries_dataframe(
            filepath,
            start_time_step=int(schema['simulation_start_time_step']),
            end_time_step=int(schema['simulation_end_time_step']),
        )
        if not parse_bool(
            schema.get('terminal_observation_padding', False),
            default=False,
            path='terminal_observation_padding',
        ):
            return dataframe

        expected_rows = (
            int(schema['simulation_end_time_step'])
            - int(schema['simulation_start_time_step'])
            + 1
        )
        if len(dataframe) == expected_rows:
            return dataframe
        if len(dataframe) != expected_rows - 1 or dataframe.empty:
            raise ValueError(
                'terminal_observation_padding may add exactly one terminal state '
                f'after the available data; expected {expected_rows - 1} source rows '
                f'but read {len(dataframe)} from {filepath}.'
            )

        terminal = dataframe.iloc[-1].copy()
        # The padded row is a terminal observation, not an additional physical
        # interval. Calendar fields describe the next quarter-hour boundary.
        if {'month', 'hour', 'minutes', 'day_type'}.issubset(dataframe.columns):
            minutes = int(terminal['minutes']) + int(
                round(float(schema.get('seconds_per_time_step', 3600)) / 60.0)
            )
            hour_increment, terminal['minutes'] = divmod(minutes, 60)
            hour = int(terminal['hour']) + hour_increment
            day_increment, terminal['hour'] = divmod(hour, 24)
            if day_increment:
                terminal['day_type'] = (int(terminal['day_type']) % 7) + 1
                if int(terminal['month']) == 12:
                    terminal['month'] = 1

        # A vehicle whose countdown reaches one leaves at the boundary after
        # the final controlled interval. Make that transition observable to the
        # KPI layer without inventing a 2024 charging interval.
        if {
            'electric_vehicle_charger_state',
            'electric_vehicle_departure_time',
        }.issubset(dataframe.columns):
            state = pd.to_numeric(
                pd.Series([terminal['electric_vehicle_charger_state']]),
                errors='coerce',
            ).iloc[0]
            departure = pd.to_numeric(
                pd.Series([terminal['electric_vehicle_departure_time']]),
                errors='coerce',
            ).iloc[0]
            if state == 1 and np.isfinite(departure) and float(departure) <= 1.0:
                terminal['electric_vehicle_charger_state'] = 3
                for column in (
                    'electric_vehicle_session_id',
                    'electric_vehicle_departure_time',
                    'electric_vehicle_required_soc_departure',
                    'electric_vehicle_current_soc',
                ):
                    if column in dataframe.columns:
                        terminal[column] = np.nan

        terminal_frame = pd.DataFrame(
            {
                column: pd.Series([terminal[column]], dtype=dtype)
                for column, dtype in dataframe.dtypes.items()
            }
        )
        return pd.concat(
            [dataframe, terminal_frame],
            ignore_index=True,
        )

    def _read_timeseries_dataframe(
        self,
        filepath: Union[str, os.PathLike],
        *,
        start_time_step: Optional[int] = None,
        end_time_step: Optional[int] = None,
    ) -> pd.DataFrame:
        if start_time_step is not None and end_time_step is not None and end_time_step < start_time_step:
            raise ValueError(
                f'end_time_step ({end_time_step}) must be >= start_time_step ({start_time_step}) '
                f'for source={filepath}.'
            )

        if self._file_format(filepath) == 'parquet':
            return self._read_parquet_timeseries_dataframe(
                filepath,
                start_time_step=start_time_step,
                end_time_step=end_time_step,
            )

        nrows = None if end_time_step is None else int(end_time_step) - int(start_time_step or 0) + 1

        if start_time_step is None or int(start_time_step) <= 0:
            return pd.read_csv(filepath, nrows=nrows)

        return pd.read_csv(filepath, skiprows=range(1, int(start_time_step) + 1), nrows=nrows)

    @staticmethod
    def _read_parquet_timeseries_dataframe(
        filepath: Union[str, os.PathLike],
        *,
        start_time_step: Optional[int] = None,
        end_time_step: Optional[int] = None,
    ) -> pd.DataFrame:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet CityLearn datasets requires the optional 'pyarrow' dependency. "
                "Install pyarrow or use CSV files in the schema."
            ) from exc

        start = 0 if start_time_step is None else max(0, int(start_time_step))
        end = None if end_time_step is None else int(end_time_step)
        parquet_file = pq.ParquetFile(filepath)

        if start == 0 and end is None:
            return parquet_file.read().to_pandas()

        batches = []
        row_cursor = 0

        for batch in parquet_file.iter_batches(batch_size=65_536):
            batch_rows = batch.num_rows
            batch_start = row_cursor
            batch_end = row_cursor + batch_rows

            if batch_end <= start:
                row_cursor = batch_end
                continue

            if end is not None and batch_start > end:
                break

            local_start = max(0, start - batch_start)
            local_end = batch_rows if end is None else min(batch_rows, end + 1 - batch_start)

            if local_end > local_start:
                batches.append(batch.slice(local_start, local_end - local_start))

            row_cursor = batch_end

        if len(batches) == 0:
            schema = parquet_file.schema_arrow
            arrays = [pa.array([], type=field.type) for field in schema]
            return pa.Table.from_arrays(arrays, names=schema.names).to_pandas()

        return pa.Table.from_batches(batches).to_pandas()

    def _read_dataframe(self, filepath: Union[str, os.PathLike]) -> pd.DataFrame:
        if self._file_format(filepath) == 'parquet':
            return self._read_parquet_timeseries_dataframe(filepath)
        return pd.read_csv(filepath)

    def _load_shared_timeseries(
        self,
        constructor: Any,
        *,
        schema: Mapping[str, Any],
        filepath: Union[str, os.PathLike],
        expected_rows: int,
        window: Optional[Tuple[int, int]],
        source_label: str,
        noise_std: float,
    ) -> Any:
        cacheable = float(noise_std or 0.0) == 0.0
        base_cache_key = (
            constructor.__name__,
            os.path.abspath(str(filepath)),
            int(schema['simulation_start_time_step']),
            int(schema['simulation_end_time_step']),
            int(expected_rows),
        )
        full_horizon_cache_key = (*base_cache_key, None)
        window_cache_key = (*base_cache_key, tuple(window) if window is not None else None)

        if cacheable:
            if full_horizon_cache_key in self._shared_timeseries_cache:
                return self._shared_timeseries_cache[full_horizon_cache_key]
            if window_cache_key in self._shared_timeseries_cache:
                return self._shared_timeseries_cache[window_cache_key]

        dataframe = self._read_simulation_dataframe(schema, filepath)
        full_horizon_source = len(dataframe) >= int(expected_rows)
        dataframe = self._align_dynamic_timeseries_dataframe(
            dataframe,
            expected_rows=expected_rows,
            window=window,
            source_label=source_label,
        )
        time_series = constructor(**self._dataframe_to_constructor_kwargs(dataframe), noise_std=noise_std)
        self._set_time_step_offset(time_series, schema['simulation_start_time_step'])

        if cacheable:
            cache_key = full_horizon_cache_key if full_horizon_source else window_cache_key
            self._shared_timeseries_cache[cache_key] = time_series

        return time_series

    @staticmethod
    def _collect_topology_events(schema: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        raw_events = schema.get('topology_events', []) if isinstance(schema, Mapping) else []
        events = [event for event in raw_events if isinstance(event, Mapping)]
        events.sort(
            key=lambda event: (
                int(event.get('time_step', 0)),
                str(event.get('id', '')),
            )
        )
        return events

    @staticmethod
    def _normalize_action_tokens(values: Any) -> List[str]:
        if values is None:
            return []
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        return [str(value) for value in values]

    def _resolve_inactive_actions(
        self,
        index: int,
        building_schema: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> List[str]:
        inactive_actions_override = kwargs.get('inactive_actions')
        if inactive_actions_override is not None:
            if isinstance(inactive_actions_override, list) and len(inactive_actions_override) > 0 and isinstance(inactive_actions_override[0], list):
                if index < len(inactive_actions_override):
                    inactive_actions_override = inactive_actions_override[index]
                else:
                    inactive_actions_override = []
            return self._normalize_action_tokens(inactive_actions_override)

        inactive_actions_schema = None if building_schema is None else building_schema.get('inactive_actions')
        return self._normalize_action_tokens(inactive_actions_schema)

    @staticmethod
    def _building_has_electrical_storage_asset(building: Building) -> bool:
        battery = getattr(building, 'electrical_storage', None)
        if battery is None:
            return False
        capacity = float(getattr(battery, 'capacity', 0.0) or 0.0)
        nominal_power = float(getattr(battery, 'nominal_power', 0.0) or 0.0)
        return capacity > 0.0 and nominal_power > 0.0

    def _declared_inactive_actions_for_building(self, building: Building) -> Set[str]:
        declared = getattr(building, '_declared_inactive_actions', None)
        if isinstance(declared, (set, list, tuple)):
            return {str(value) for value in declared}

        schema = self.env.schema if isinstance(getattr(self.env, 'schema', None), Mapping) else {}
        building_schema = (schema.get('buildings', {}) or {}).get(getattr(building, 'name', ''), {})
        return set(self._normalize_action_tokens((building_schema or {}).get('inactive_actions')))

    def _validate_static_action_asset_consistency(self, buildings: List[Building]) -> None:
        inconsistent: List[str] = []

        for building in buildings:
            action_enabled = bool(getattr(building, 'action_metadata', {}).get('electrical_storage', False))
            if not action_enabled:
                continue

            if self._building_has_electrical_storage_asset(building):
                continue

            if 'electrical_storage' in self._declared_inactive_actions_for_building(building):
                continue

            inconsistent.append(str(getattr(building, 'name', '<unknown>')))

        if inconsistent:
            joined = ', '.join(inconsistent)
            raise ValueError(
                'Schema/action inconsistency in static topology: electrical_storage action is active '
                f'for building(s) without electrical_storage asset declaration or positive capacity/power: {joined}. '
                "Fix by declaring electrical_storage for those buildings or adding 'electrical_storage' to each building's inactive_actions."
            )

    def _build_dynamic_member_windows(
        self,
        schema: Mapping[str, Any],
        expected_rows: int,
    ) -> Dict[str, Tuple[int, int]]:
        windows: Dict[str, Tuple[int, int]] = {}
        events = self._collect_topology_events(schema)
        buildings = schema.get('buildings', {}) if isinstance(schema, Mapping) else {}

        for member_id, member_schema in buildings.items():
            include = parse_bool(
                (member_schema or {}).get('include', True),
                default=True,
                path=f'buildings.{member_id}.include',
            )
            start = 0 if include else None
            end = int(expected_rows)

            if start is None:
                for event in events:
                    if str(event.get('operation', '')).strip().lower() != 'add_member':
                        continue
                    if event.get('target_member_id') != member_id:
                        continue
                    start = int(event.get('time_step', 0))
                    break

            if start is None:
                continue

            for event in events:
                if str(event.get('operation', '')).strip().lower() != 'remove_member':
                    continue
                if event.get('target_member_id') != member_id:
                    continue
                candidate = int(event.get('time_step', 0))
                if candidate >= start:
                    end = min(end, candidate)
                    break

            start = max(0, min(int(start), int(expected_rows)))
            end = max(start, min(int(end), int(expected_rows)))

            if end > start:
                windows[str(member_id)] = (start, end)

        return windows

    def _build_dynamic_charger_windows(
        self,
        schema: Mapping[str, Any],
        member_windows: Mapping[str, Tuple[int, int]],
        expected_rows: int,
    ) -> Dict[Tuple[str, str], Tuple[int, int]]:
        windows: Dict[Tuple[str, str], Tuple[int, int]] = {}
        events = self._collect_topology_events(schema)
        buildings = schema.get('buildings', {}) if isinstance(schema, Mapping) else {}

        for member_id, member_schema in buildings.items():
            member_window = member_windows.get(str(member_id))
            if member_window is None:
                continue

            base_start, base_end = member_window
            chargers = (member_schema or {}).get('chargers', {}) or {}

            for charger_id in chargers.keys():
                start, end = int(base_start), int(base_end)

                for event in events:
                    if str(event.get('operation', '')).strip().lower() != 'remove_asset':
                        continue
                    if str(event.get('target_asset_type', '')).strip().lower() != 'charger':
                        continue
                    if event.get('target_member_id') != member_id:
                        continue
                    if event.get('target_asset_id') != charger_id:
                        continue
                    candidate = int(event.get('time_step', 0))
                    if candidate >= start:
                        end = min(end, candidate)
                        break

                start = max(0, min(start, int(expected_rows)))
                end = max(start, min(end, int(expected_rows)))
                if end > start:
                    windows[(str(member_id), str(charger_id))] = (start, end)

        return windows

    @staticmethod
    def _build_dynamic_deferrable_appliance_windows(
        schema: Mapping[str, Any],
        member_windows: Mapping[str, Tuple[int, int]],
        expected_rows: int,
    ) -> Dict[Tuple[str, str], Tuple[int, int]]:
        windows: Dict[Tuple[str, str], Tuple[int, int]] = {}
        buildings = schema.get('buildings', {}) if isinstance(schema, Mapping) else {}

        for member_id, member_schema in buildings.items():
            member_window = member_windows.get(str(member_id))
            if member_window is None:
                continue

            start, end = member_window
            start = max(0, min(int(start), int(expected_rows)))
            end = max(start, min(int(end), int(expected_rows)))
            if end <= start:
                continue

            appliances = (member_schema or {}).get('deferrable_appliances', {}) or {}
            for appliance_id in appliances.keys():
                windows[(str(member_id), str(appliance_id))] = (start, end)

        return windows

    def _align_dynamic_timeseries_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        expected_rows: int,
        window: Optional[Tuple[int, int]],
        source_label: str,
    ) -> pd.DataFrame:
        if not self._dynamic_mode or window is None:
            return dataframe

        expected_rows = int(expected_rows)
        if expected_rows <= 0:
            return dataframe

        row_count = len(dataframe)

        if row_count == expected_rows:
            index = dataframe.index
            if (
                isinstance(index, pd.RangeIndex)
                and int(index.start) == 0
                and int(index.stop) == expected_rows
                and int(index.step) == 1
            ):
                return dataframe

            return dataframe.reset_index(drop=True)

        df = dataframe.iloc[:expected_rows].copy() if row_count > expected_rows else dataframe.copy()

        start, end = window
        start = max(0, min(int(start), expected_rows))
        end = max(start, min(int(end), expected_rows))
        active_rows = end - start

        row_count = len(df)

        if active_rows <= 0:
            raise ValueError(
                f'{source_label} has no active window in dynamic mode. '
                f'Provide full-horizon data ({expected_rows} rows) for this source.'
            )

        if row_count <= 0:
            raise ValueError(
                f'{source_label} has 0 rows in dynamic mode. '
                f'Provide at least 1 row or a full-horizon file ({expected_rows} rows).'
            )

        aligned = pd.DataFrame(index=np.arange(expected_rows), columns=df.columns)
        window_rows = min(active_rows, row_count)
        aligned.iloc[start:start + window_rows, :] = df.iloc[:window_rows].to_numpy()
        for column in aligned.columns:
            with pd.option_context('future.no_silent_downcasting', True):
                aligned[column] = aligned[column].ffill().bfill()
        return aligned.reset_index(drop=True)
