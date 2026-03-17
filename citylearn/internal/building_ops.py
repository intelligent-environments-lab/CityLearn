from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from citylearn.energy_model import HeatPump
from citylearn.preprocessing import Normalize, PeriodicNormalization

if TYPE_CHECKING:
    from citylearn.building import Building

LOGGER = logging.getLogger()


class BuildingOpsService:
    """Internal observation/action operations for `Building`."""

    def __init__(self, building: "Building"):
        self.building = building

    def observations(
        self,
        include_all: bool = None,
        normalize: bool = None,
        periodic_normalization: bool = None,
        check_limits: bool = None,
    ) -> Mapping[str, float]:
        """Observations at current time step."""

        building = self.building

        normalize = False if normalize is None else normalize
        periodic_normalization = False if periodic_normalization is None else periodic_normalization
        include_all = False if include_all is None else include_all
        check_limits = False if check_limits is None else check_limits

        data = self.get_observations_data(include_all=include_all)

        if include_all:
            valid_observations = list(set(data.keys()) | set(building.active_observations))
        else:
            valid_observations = building.active_observations

        observations = {k: data[k] for k in valid_observations if k in data.keys()}

        observations = self.update_ev_charger_observations(
            observations,
            valid_observations,
            building.electric_vehicle_chargers,
            include_all=include_all,
        )

        observations = self.update_washing_machine_observations(
            observations,
            valid_observations,
            building.washing_machines,
        )

        unknown_observations = set(observations.keys()).difference(set(valid_observations))
        assert len(unknown_observations) == 0, f'Unknown observations: {unknown_observations}'

        non_periodic_low_limit, non_periodic_high_limit = building.non_periodic_normalized_observation_space_limits
        periodic_low_limit, periodic_high_limit = building.periodic_normalized_observation_space_limits
        periodic_observations = building.get_periodic_observation_metadata()

        if check_limits:
            for key in building.active_observations:
                value = observations[key]
                lower = non_periodic_low_limit[key]
                upper = non_periodic_high_limit[key]
                if not lower <= value <= upper:
                    report = {
                        'Building': building.name,
                        'episode': building.episode_tracker.episode,
                        'time_step': f'{building.time_step + 1}/{building.episode_tracker.episode_time_steps}',
                        'observation': key,
                        'value': value,
                        'lower': lower,
                        'upper': upper,
                    }
                    LOGGER.debug(f'Observation outside space limit: {report}')

        if periodic_normalization:
            observations_copy = {k: v for k, v in observations.items()}
            observations = {}
            periodic_normalizer = PeriodicNormalization(x_max=0)

            for key, value in observations_copy.items():
                if key in periodic_observations:
                    periodic_normalizer.x_max = max(periodic_observations[key])
                    sin_x, cos_x = value * periodic_normalizer
                    observations[f'{key}_cos'] = cos_x
                    observations[f'{key}_sin'] = sin_x
                else:
                    observations[key] = value

        if normalize:
            normalizer = Normalize(0.0, 1.0)

            for key, value in observations.items():
                normalizer.x_min = periodic_low_limit[key]
                normalizer.x_max = periodic_high_limit[key]
                observations[key] = value * normalizer

        return observations

    def update_ev_charger_observations(self, observations, valid_observations, ev_chargers, include_all: bool = False):
        """Update observations for each electric vehicle charger."""

        building = self.building

        for charger in ev_chargers:
            charger_id = charger.charger_id
            sim = charger.charger_simulation
            t = building.time_step
            endogenous_t = t if include_all else max(t - 1, 0)

            connected_state_key = f'electric_vehicle_charger_{charger_id}_connected_state'
            incoming_state_key = f'electric_vehicle_charger_{charger_id}_incoming_state'
            departure_key = f'connected_electric_vehicle_at_charger_{charger_id}_departure_time'
            req_soc_key = f'connected_electric_vehicle_at_charger_{charger_id}_required_soc_departure'
            soc_key = f'connected_electric_vehicle_at_charger_{charger_id}_soc'
            capacity_key = f'connected_electric_vehicle_at_charger_{charger_id}_battery_capacity'
            arrival_key = f'incoming_electric_vehicle_at_charger_{charger_id}_estimated_arrival_time'
            soc_arrival_key = f'incoming_electric_vehicle_at_charger_{charger_id}_estimated_soc_arrival'

            state = sim.electric_vehicle_charger_state[t] if t < len(sim.electric_vehicle_charger_state) else np.nan

            if charger.connected_electric_vehicle and state == 1:
                if connected_state_key in valid_observations:
                    observations[connected_state_key] = 1
                if departure_key in valid_observations:
                    observations[departure_key] = int(sim.electric_vehicle_departure_time[t])
                if req_soc_key in valid_observations:
                    observations[req_soc_key] = float(sim.electric_vehicle_required_soc_departure[t])
                if soc_key in valid_observations:
                    observations[soc_key] = charger.connected_electric_vehicle.battery.soc[endogenous_t]
                if capacity_key in valid_observations:
                    observations[capacity_key] = float(charger.connected_electric_vehicle.battery.capacity)
            else:
                if connected_state_key in valid_observations:
                    observations[connected_state_key] = 0
                if departure_key in valid_observations:
                    observations[departure_key] = -1
                if req_soc_key in valid_observations:
                    observations[req_soc_key] = -0.1
                if soc_key in valid_observations:
                    observations[soc_key] = -0.1
                if capacity_key in valid_observations:
                    observations[capacity_key] = -1.0

            if charger.incoming_electric_vehicle and state == 2:
                if incoming_state_key in valid_observations:
                    observations[incoming_state_key] = 1
                if arrival_key in valid_observations:
                    observations[arrival_key] = int(sim.electric_vehicle_estimated_arrival_time[t])
                if soc_arrival_key in valid_observations:
                    observations[soc_arrival_key] = float(sim.electric_vehicle_estimated_soc_arrival[t])
            else:
                if incoming_state_key in valid_observations:
                    observations[incoming_state_key] = 0
                if arrival_key in valid_observations:
                    observations[arrival_key] = -1
                if soc_arrival_key in valid_observations:
                    observations[soc_arrival_key] = -0.1

        return observations

    def update_washing_machine_observations(self, observations, valid_observations, washing_machines):
        """Update observations for each washing machine."""

        for washing_machine in washing_machines:
            washing_machine_name = washing_machine.name
            washing_machine_observations = washing_machine.observations()

            start_key = f'{washing_machine_name}_start_time_step'
            if start_key in valid_observations:
                observations[start_key] = next(
                    (value for key, value in washing_machine_observations.items() if '_start_time_step' in key),
                    -1,
                )

            end_key = f'{washing_machine_name}_end_time_step'
            if end_key in valid_observations:
                observations[end_key] = next(
                    (value for key, value in washing_machine_observations.items() if '_end_time_step' in key),
                    -1,
                )
        return observations

    def get_observations_data(self, include_all: bool = False) -> Mapping[str, Union[float, int]]:
        """Build base observation dictionary without normalization."""

        building = self.building

        electric_vehicle_chargers_dict = {}
        washing_machines_dict = {}
        t = building.time_step
        endogenous_t = t if include_all else max(t - 1, 0)

        for charger in building.electric_vehicle_chargers or []:
            charger_id = charger.charger_id
            connected_car = charger.connected_electric_vehicle

            if connected_car is not None:
                last_charged_kwh = 0.0
                if 0 <= endogenous_t < len(charger.past_charging_action_values_kwh):
                    last_charged_kwh = float(charger.past_charging_action_values_kwh[endogenous_t])

                battery_soc = connected_car.battery.soc[endogenous_t]
                previous_battery_soc = connected_car.battery.initial_soc if endogenous_t == 0 else connected_car.battery.soc[endogenous_t - 1]

                required_soc = charger.charger_simulation.electric_vehicle_required_soc_departure[t]
                hours_until_departure = charger.charger_simulation.electric_vehicle_departure_time[t]

                battery_capacity = connected_car.battery.capacity
                min_capacity = (1 - connected_car.battery.depth_of_discharge) * battery_capacity

                electric_vehicle_chargers_dict[charger_id] = {
                    'connected': True,
                    'last_charged_kwh': last_charged_kwh,
                    'previous_battery_soc': previous_battery_soc,
                    'battery_soc': battery_soc,
                    'battery_capacity': battery_capacity,
                    'min_capacity': min_capacity,
                    'required_soc': required_soc,
                    'hours_until_departure': hours_until_departure,
                    'max_charging_power': charger.max_charging_power,
                    'max_discharging_power': charger.max_discharging_power,
                }

            else:
                electric_vehicle_chargers_dict[charger_id] = {
                    'connected': False,
                    'last_charged_kwh': 0.0,
                    'previous_battery_soc': None,
                    'battery_soc': None,
                    'battery_capacity': None,
                    'min_capacity': None,
                    'required_soc': None,
                    'hours_until_departure': None,
                    'max_charging_power': charger.max_charging_power,
                    'max_discharging_power': charger.max_discharging_power,
                }

        for washing_machine in building.washing_machines or []:
            washing_machine_name = washing_machine.name

            def _safe(arr, idx, default):
                try:
                    return arr[idx]
                except Exception:
                    return default

            start_time_step = _safe(washing_machine.washing_machine_simulation.wm_start_time_step, t, -1)
            end_time_step = _safe(washing_machine.washing_machine_simulation.wm_end_time_step, t, -1)
            load_profile = _safe(washing_machine.washing_machine_simulation.load_profile, t, 0.0)

            washing_machines_dict[washing_machine_name] = {
                'wm_start_time_step': start_time_step,
                'wm_end_time_step': end_time_step,
                'load_profile': load_profile,
            }

        observations = {}
        for key, series in building._energy_simulation_observation_sources:
            if t < len(series):
                observations[key] = series[t]

        for key, series in building._weather_observation_sources:
            if t < len(series):
                observations[key] = series[t]

        for key, series in building._pricing_observation_sources:
            if t < len(series):
                observations[key] = series[t]

        for key, series in building._carbon_observation_sources:
            if t < len(series):
                observations[key] = series[t]

        observations.update({
            'solar_generation': abs(building.solar_generation[t]),
            **{
                'cooling_storage_soc': building.cooling_storage.soc[endogenous_t],
                'heating_storage_soc': building.heating_storage.soc[endogenous_t],
                'dhw_storage_soc': building.dhw_storage.soc[endogenous_t],
                'electrical_storage_soc': building.electrical_storage.soc[endogenous_t],
            },
            'cooling_demand': building.energy_from_cooling_device[endogenous_t] + abs(min(building.cooling_storage.energy_balance[endogenous_t], 0.0)),
            'heating_demand': building.energy_from_heating_device[endogenous_t] + abs(min(building.heating_storage.energy_balance[endogenous_t], 0.0)),
            'dhw_demand': building.energy_from_dhw_device[endogenous_t] + abs(min(building.dhw_storage.energy_balance[endogenous_t], 0.0)),
            'net_electricity_consumption': building.net_electricity_consumption[endogenous_t],
            'cooling_electricity_consumption': building.cooling_electricity_consumption[endogenous_t],
            'heating_electricity_consumption': building.heating_electricity_consumption[endogenous_t],
            'dhw_electricity_consumption': building.dhw_electricity_consumption[endogenous_t],
            'cooling_storage_electricity_consumption': building.cooling_storage_electricity_consumption[endogenous_t],
            'heating_storage_electricity_consumption': building.heating_storage_electricity_consumption[endogenous_t],
            'dhw_storage_electricity_consumption': building.dhw_storage_electricity_consumption[endogenous_t],
            'electrical_storage_electricity_consumption': building.electrical_storage_electricity_consumption[endogenous_t],
            'washing_machine_electricity_consumption': building.washing_machines_electricity_consumption[endogenous_t],
            'cooling_device_efficiency': building.cooling_device.get_cop(building.weather.outdoor_dry_bulb_temperature[t], heating=False),
            'heating_device_efficiency': building.heating_device.get_cop(building.weather.outdoor_dry_bulb_temperature[t], heating=True)
            if isinstance(building.heating_device, HeatPump) else building.heating_device.efficiency,
            'dhw_device_efficiency': building.dhw_device.get_cop(building.weather.outdoor_dry_bulb_temperature[t], heating=True)
            if isinstance(building.dhw_device, HeatPump) else building.dhw_device.efficiency,
            'indoor_dry_bulb_temperature_cooling_set_point': building.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[t],
            'indoor_dry_bulb_temperature_heating_set_point': building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[t],
            'indoor_dry_bulb_temperature_cooling_delta': building.energy_simulation.indoor_dry_bulb_temperature[t] - building.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[t],
            'indoor_dry_bulb_temperature_heating_delta': building.energy_simulation.indoor_dry_bulb_temperature[t] - building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[t],
            'comfort_band': building.energy_simulation.comfort_band[t],
            'occupant_count': building.energy_simulation.occupant_count[t],
            'power_outage': building.power_outage_signal[t],
            'electric_vehicles_chargers_dict': electric_vehicle_chargers_dict,
            'washing_machines_dict': washing_machines_dict,
        })

        if (
            getattr(building, '_charging_constraints_enabled', False)
            and getattr(building, '_expose_charging_constraints', False)
            and isinstance(building._charging_constraints_state, dict)
        ):
            state = building._charging_constraints_state
            headroom = state.get('building_headroom_kw')
            if headroom is not None:
                observations['charging_building_headroom_kw'] = headroom
            export_headroom = state.get('building_export_headroom_kw')
            if export_headroom is not None:
                observations['charging_building_export_headroom_kw'] = export_headroom
            for phase_name, value in (state.get('phase_headroom_kw') or {}).items():
                if value is not None:
                    observations[f'charging_phase_{phase_name}_headroom_kw'] = value
            for phase_name, value in (state.get('phase_export_headroom_kw') or {}).items():
                if value is not None:
                    observations[f'charging_phase_{phase_name}_export_headroom_kw'] = value

        if getattr(building, '_charging_constraints_enabled', False):
            if getattr(building, '_expose_charging_violation', False):
                observations['charging_constraint_violation_kwh'] = building._charging_constraint_last_penalty_kwh
            if getattr(building, '_phase_encoding_observations', None):
                observations.update(building._phase_encoding_observations)

        return observations

    def apply_actions(
        self,
        cooling_or_heating_device_action: float = None,
        cooling_device_action: float = None,
        heating_device_action: float = None,
        cooling_storage_action: float = None,
        heating_storage_action: float = None,
        dhw_storage_action: float = None,
        electrical_storage_action: float = None,
        washing_machine_actions: dict = None,
        electric_vehicle_storage_actions: dict = None,
    ):
        """Update demand and charge/discharge storage devices."""

        building = self.building

        if electric_vehicle_storage_actions is not None:
            electric_vehicle_storage_actions = dict(electric_vehicle_storage_actions)

        if 'cooling_or_heating_device' in building.active_actions:
            assert 'cooling_device' not in building.active_actions and 'heating_device' not in building.active_actions, \
                'cooling_device and heating_device actions must be set to False when cooling_or_heating_device is True.' \
                ' They will be implicitly set based on the polarity of cooling_or_heating_device.'
            cooling_device_action = abs(min(cooling_or_heating_device_action, 0.0))
            heating_device_action = abs(max(cooling_or_heating_device_action, 0.0))

        else:
            assert not ('cooling_device' in building.active_actions and 'heating_device' in building.active_actions), \
                'cooling_device and heating_device actions cannot both be set to True to avoid both actions having' \
                ' values > 0.0 in the same time step. Use cooling_or_heating_device action instead to control' \
                ' both cooling_device and heating_device in a building.'
            cooling_device_action = np.nan if 'cooling_device' not in building.active_actions else cooling_device_action
            heating_device_action = np.nan if 'heating_device' not in building.active_actions else heating_device_action

        cooling_storage_action = 0.0 if 'cooling_storage' not in building.active_actions else cooling_storage_action
        heating_storage_action = 0.0 if 'heating_storage' not in building.active_actions else heating_storage_action
        dhw_storage_action = 0.0 if 'dhw_storage' not in building.active_actions else dhw_storage_action
        electrical_storage_action = 0.0 if 'electrical_storage' not in building.active_actions else electrical_storage_action

        electric_vehicle_storage_actions, electrical_storage_action = self.apply_charging_constraints_to_actions(
            electric_vehicle_storage_actions,
            electrical_storage_action,
        )

        actions = {
            'cooling_demand': (building.update_cooling_demand, (cooling_device_action,)),
            'heating_demand': (building.update_heating_demand, (heating_device_action,)),
            'cooling_device': (building.update_energy_from_cooling_device, ()),
            'cooling_storage': (building.update_cooling_storage, (cooling_storage_action,)),
            'heating_device': (building.update_energy_from_heating_device, ()),
            'heating_storage': (building.update_heating_storage, (heating_storage_action,)),
            'dhw_device': (building.update_energy_from_dhw_device, ()),
            'dhw_storage': (building.update_dhw_storage, (dhw_storage_action,)),
            'non_shiftable_load': (building.update_non_shiftable_load, ()),
            'electrical_storage': (building.update_electrical_storage, (electrical_storage_action,)),
        }

        priority_list = list(actions.keys())

        if electric_vehicle_storage_actions is not None:
            electric_vehicle_priority_list = []
            for charger_id, action in electric_vehicle_storage_actions.items():
                action_key = f'electric_vehicle_storage_{charger_id}'
                if action_key not in building.active_actions:
                    raise ValueError('This action should not be applied. Verify')
                for charger in building.electric_vehicle_chargers:
                    if charger.charger_id == charger_id:
                        actions[action_key] = (charger.update_connected_electric_vehicle_soc, (action,))
                        electric_vehicle_priority_list.append(action_key)
            priority_list = priority_list + electric_vehicle_priority_list

        if washing_machine_actions is not None:
            washing_machine_priority_list = []
            for washing_machine_name, action in washing_machine_actions.items():
                action_key = f'{washing_machine_name}'
                if action_key not in building.active_actions:
                    raise ValueError('This action should not be applied. Verify')
                for washing_machine in building.washing_machines:
                    if washing_machine.name == washing_machine_name:
                        actions[action_key] = (washing_machine.start_cycle, (action,))
                        washing_machine_priority_list.append(action_key)
            priority_list = priority_list + washing_machine_priority_list

        if electrical_storage_action < 0.0:
            key = 'electrical_storage'
            priority_list.remove(key)
            priority_list = [key] + priority_list

        for key in ['cooling', 'heating', 'dhw']:
            storage = f'{key}_storage'
            device = f'{key}_device'

            if actions[storage][1][0] < 0.0:
                storage_ix = priority_list.index(storage)
                device_ix = priority_list.index(device)
                priority_list[storage_ix] = device
                priority_list[device_ix] = storage

        for key in priority_list:
            func, args = actions[key]

            try:
                func(*args)
            except NotImplementedError:
                pass

    def _safe_scalar(self, value, default: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(scalar):
            return float(default)

        return scalar

    def _safe_index(self, values, idx: int, default: float = 0.0) -> float:
        try:
            return self._safe_scalar(values[idx], default)
        except Exception:
            return float(default)

    def _current_phase_names(self):
        building = self.building
        if getattr(building, '_electrical_service_mode', 'single_phase') == 'three_phase':
            return ['L1', 'L2', 'L3']
        return ['L1']

    def _split_unassigned_power(self, power_kw: float) -> Dict[str, float]:
        building = self.building
        phase_names = self._current_phase_names()

        if len(phase_names) == 1:
            return {'L1': float(power_kw)}

        split_mode = str(getattr(building, '_electrical_service_default_split', 'balanced')).strip().lower()
        if split_mode in {'l1', 'l2', 'l3'}:
            return {phase: float(power_kw if phase.lower() == split_mode else 0.0) for phase in phase_names}

        share = float(power_kw) / len(phase_names)
        return {phase: share for phase in phase_names}

    def _split_power_by_connection(self, power_kw: float, phase_connection: Optional[str]) -> Dict[str, float]:
        phase_names = self._current_phase_names()

        if len(phase_names) == 1:
            return {'L1': float(power_kw)}

        if phase_connection in {'L1', 'L2', 'L3'}:
            return {phase: float(power_kw if phase == phase_connection else 0.0) for phase in phase_names}

        if phase_connection == 'all_phases':
            share = float(power_kw) / len(phase_names)
            return {phase: share for phase in phase_names}

        return self._split_unassigned_power(power_kw)

    def _estimate_non_controllable_base_power(self) -> Tuple[float, Dict[str, float]]:
        building = self.building
        t = building.time_step
        phase_names = self._current_phase_names()

        if building.power_outage:
            return 0.0, {phase: 0.0 for phase in phase_names}

        temperature = self._safe_index(building.weather.outdoor_dry_bulb_temperature, t, 0.0)

        cooling_demand = self._safe_index(building.energy_from_cooling_device, t, 0.0) + self._safe_index(
            building.cooling_storage.energy_balance, t, 0.0
        )
        cooling_kw = self._safe_scalar(building.cooling_device.get_input_power(cooling_demand, temperature, heating=False), 0.0)

        heating_demand = self._safe_index(building.energy_from_heating_device, t, 0.0) + self._safe_index(
            building.heating_storage.energy_balance, t, 0.0
        )
        if isinstance(building.heating_device, HeatPump):
            heating_kw = self._safe_scalar(building.heating_device.get_input_power(heating_demand, temperature, heating=True), 0.0)
        else:
            heating_kw = self._safe_scalar(building.dhw_device.get_input_power(heating_demand), 0.0)

        dhw_demand = self._safe_index(building.energy_from_dhw_device, t, 0.0) + self._safe_index(
            building.dhw_storage.energy_balance, t, 0.0
        )
        if isinstance(building.dhw_device, HeatPump):
            dhw_kw = self._safe_scalar(building.dhw_device.get_input_power(dhw_demand, temperature, heating=True), 0.0)
        else:
            dhw_kw = self._safe_scalar(building.dhw_device.get_input_power(dhw_demand), 0.0)

        non_shiftable_kw = self._safe_index(building.energy_to_non_shiftable_load, t, 0.0)
        solar_kw = self._safe_index(building.solar_generation, t, 0.0)
        washing_kw = sum(self._safe_index(wm.electricity_consumption, t, 0.0) for wm in building.washing_machines or [])

        base_total_kw = cooling_kw + heating_kw + dhw_kw + non_shiftable_kw + solar_kw + washing_kw
        base_phase_kw = self._split_unassigned_power(base_total_kw)

        return float(base_total_kw), base_phase_kw

    def _charger_requested_power_kw(self, charger, action: float) -> float:
        if action is None:
            return 0.0

        action = self._safe_scalar(action, 0.0)
        action = float(np.clip(action, -1.0, 1.0))
        if action > 0.0:
            max_power = self._safe_scalar(getattr(charger, 'max_charging_power', 0.0), 0.0)
            return action * max_power if max_power > 0.0 else 0.0
        if action < 0.0:
            max_power = self._safe_scalar(getattr(charger, 'max_discharging_power', 0.0), 0.0)
            return -abs(action) * max_power if max_power > 0.0 else 0.0
        return 0.0

    def _charger_action_from_power_kw(self, charger, target_power_kw: float) -> float:
        target_power_kw = self._safe_scalar(target_power_kw, 0.0)

        if target_power_kw > 0.0:
            max_power = self._safe_scalar(getattr(charger, 'max_charging_power', 0.0), 0.0)
            min_power = self._safe_scalar(getattr(charger, 'min_charging_power', 0.0), 0.0)
            if max_power <= 0.0:
                return 0.0
            if min_power > 0.0 and target_power_kw < min_power:
                return 0.0
            return float(np.clip(target_power_kw / max_power, 0.0, 1.0))

        if target_power_kw < 0.0:
            max_power = self._safe_scalar(getattr(charger, 'max_discharging_power', 0.0), 0.0)
            min_power = self._safe_scalar(getattr(charger, 'min_discharging_power', 0.0), 0.0)
            requested = abs(target_power_kw)
            if max_power <= 0.0:
                return 0.0
            if min_power > 0.0 and requested < min_power:
                return 0.0
            return float(-np.clip(requested / max_power, 0.0, 1.0))

        return 0.0

    def _storage_requested_power_kw(self, action: Optional[float]) -> float:
        building = self.building
        if action is None:
            return 0.0
        action = self._safe_scalar(action, 0.0)
        action = float(np.clip(action, -1.0, 1.0))
        nominal_power = self._safe_scalar(getattr(building.electrical_storage, 'nominal_power', 0.0), 0.0)
        return action * nominal_power if nominal_power > 0.0 else 0.0

    def _storage_action_from_power_kw(self, target_power_kw: float) -> float:
        building = self.building
        nominal_power = self._safe_scalar(getattr(building.electrical_storage, 'nominal_power', 0.0), 0.0)
        if nominal_power <= 0.0:
            return 0.0
        return float(np.clip(target_power_kw / nominal_power, -1.0, 1.0))

    def _compute_totals(self, base_total_kw: float, base_phase_kw: Mapping[str, float], controls, scales):
        total_kw = float(base_total_kw)
        phase_kw = {phase: float(value) for phase, value in base_phase_kw.items()}

        for control_id, control in controls.items():
            scale = self._safe_scalar(scales.get(control_id, 1.0), 1.0)
            total_kw += control['request_total_kw'] * scale
            for phase_name, value in control['request_phase_kw'].items():
                phase_kw[phase_name] = phase_kw.get(phase_name, 0.0) + (value * scale)

        return total_kw, phase_kw

    def _scale_for_import_scope(self, current_value_kw, limit_kw, controls, scales, component_getter) -> bool:
        limit_kw = self._safe_scalar(limit_kw, np.nan)
        current_value_kw = self._safe_scalar(current_value_kw, 0.0)
        if not np.isfinite(limit_kw):
            return False
        if limit_kw is None or current_value_kw <= limit_kw + 1e-9:
            return False

        relevant = []
        for control_id, control in controls.items():
            component_kw = component_getter(control)
            if component_kw > 0.0 and self._safe_scalar(scales.get(control_id, 0.0), 0.0) > 0.0:
                relevant.append((control_id, component_kw))

        if not relevant:
            return False

        current_relevant_kw = sum(scales[control_id] * component_kw for control_id, component_kw in relevant)
        if current_relevant_kw <= 1e-9:
            return False

        fixed_kw = current_value_kw - current_relevant_kw
        allowed_kw = limit_kw - fixed_kw
        factor = 0.0 if allowed_kw <= 0.0 else min(1.0, allowed_kw / current_relevant_kw)
        if factor >= 1.0 - 1e-9:
            return False

        for control_id, _ in relevant:
            scales[control_id] *= factor

        return True

    def _scale_for_export_scope(self, current_value_kw, limit_kw, controls, scales, component_getter) -> bool:
        limit_kw = self._safe_scalar(limit_kw, np.nan)
        current_value_kw = self._safe_scalar(current_value_kw, 0.0)
        if not np.isfinite(limit_kw):
            return False
        if limit_kw is None:
            return False

        current_export_kw = max(-current_value_kw, 0.0)
        if current_export_kw <= limit_kw + 1e-9:
            return False

        relevant = []
        for control_id, control in controls.items():
            component_kw = component_getter(control)
            if component_kw < 0.0 and self._safe_scalar(scales.get(control_id, 0.0), 0.0) > 0.0:
                relevant.append((control_id, component_kw))

        if not relevant:
            return False

        current_relevant_export_kw = sum(scales[control_id] * abs(component_kw) for control_id, component_kw in relevant)
        if current_relevant_export_kw <= 1e-9:
            return False

        fixed_kw = current_value_kw + current_relevant_export_kw
        allowed_export_kw = limit_kw + fixed_kw
        factor = 0.0 if allowed_export_kw <= 0.0 else min(1.0, allowed_export_kw / current_relevant_export_kw)
        if factor >= 1.0 - 1e-9:
            return False

        for control_id, _ in relevant:
            scales[control_id] *= factor

        return True

    def _apply_legacy_charging_constraints(self, actions: Optional[Mapping[str, float]]) -> Optional[Mapping[str, float]]:
        building = self.building

        if not actions:
            building._set_default_charging_headroom()
            return actions

        positive_requests = {}
        scales = {}
        for charger_id, action in actions.items():
            if action is None or action <= 0.0:
                continue
            charger = building._charger_lookup.get(charger_id)
            if charger is None:
                continue
            max_power = getattr(charger, 'max_charging_power', 0.0) or 0.0
            if max_power <= 0.0:
                continue
            positive_requests[charger_id] = action * max_power
            scales[charger_id] = 1.0

        violation_kw = 0.0

        if positive_requests:
            total_kw = sum(positive_requests.values())
            building_limit = building._building_charger_limit_kw
            building_limit = self._safe_scalar(building_limit, np.nan)
            if np.isfinite(building_limit) and building_limit >= 0.0 and total_kw > building_limit:
                scale = 0.0 if building_limit == 0 else building_limit / total_kw
                for charger_id in scales:
                    scales[charger_id] *= scale
                violation_kw += total_kw - building_limit

            for phase in building._phase_limits:
                limit = phase.get('limit_kw')
                limit = self._safe_scalar(limit, np.nan)
                if not np.isfinite(limit) or limit < 0.0:
                    continue
                chargers = phase.get('chargers', []) or []
                phase_sum = sum(
                    positive_requests.get(charger_id, 0.0) * scales.get(charger_id, 1.0)
                    for charger_id in chargers
                    if charger_id in positive_requests
                )
                if phase_sum > limit:
                    phase_scale = 0.0 if limit == 0 else limit / phase_sum
                    for charger_id in chargers:
                        if charger_id in scales:
                            scales[charger_id] *= phase_scale
                    violation_kw += phase_sum - limit

            scaled_positive_kw = {
                charger_id: positive_requests[charger_id] * scales.get(charger_id, 1.0)
                for charger_id in positive_requests
            }
            used_kw = sum(scaled_positive_kw.values())

            actions = dict(actions)
            for charger_id, action in list(actions.items()):
                if action is None or action <= 0.0:
                    continue
                charger = building._charger_lookup.get(charger_id)
                if charger is None:
                    continue
                max_power = getattr(charger, 'max_charging_power', 0.0) or 0.0
                if max_power <= 0.0:
                    actions[charger_id] = 0.0
                    continue
                target_kw = scaled_positive_kw.get(charger_id, 0.0)
                actions[charger_id] = max(0.0, min(action, target_kw / max_power))

            if getattr(building, '_expose_charging_constraints', False):
                building_limit = self._safe_scalar(building._building_charger_limit_kw, np.nan)
                building_headroom = None if not np.isfinite(building_limit) else building_limit - used_kw
                phase_headroom = {}
                for phase in building._phase_limits:
                    limit = phase.get('limit_kw')
                    limit = self._safe_scalar(limit, np.nan)
                    if not np.isfinite(limit):
                        phase_headroom[phase['name']] = None
                    else:
                        used = sum(scaled_positive_kw.get(charger_id, 0.0) for charger_id in phase.get('chargers', []))
                        phase_headroom[phase['name']] = limit - used

                building._charging_constraints_state = {
                    'building_headroom_kw': building_headroom,
                    'building_export_headroom_kw': None,
                    'phase_headroom_kw': phase_headroom,
                    'phase_export_headroom_kw': {},
                    'total_power_kw': used_kw,
                    'phase_power_kw': {},
                }

            penalty_kwh = self._safe_scalar(violation_kw * (building.seconds_per_time_step / 3600), 0.0)
            building._charging_constraint_penalty_kwh = penalty_kwh
            building._charging_constraint_last_penalty_kwh = penalty_kwh
            phase_power = {}
            if getattr(building, '_electrical_service_enabled', False):
                phase_power = dict((building._charging_constraints_state or {}).get('phase_power_kw') or {})
            building._record_charging_constraint_state(
                violation_kwh=penalty_kwh,
                total_power_kw=float((building._charging_constraints_state or {}).get('total_power_kw', used_kw)),
                phase_power_kw=phase_power,
            )

        else:
            building._set_default_charging_headroom()
            building._record_charging_constraint_state(
                violation_kwh=0.0,
                total_power_kw=0.0,
                phase_power_kw={},
            )

        return actions

    def _apply_electrical_service_constraints(
        self,
        actions: Optional[Mapping[str, float]],
        electrical_storage_action: Optional[float],
    ) -> Tuple[Optional[Mapping[str, float]], Optional[float]]:
        building = self.building
        phase_names = self._current_phase_names()
        base_total_kw, base_phase_kw = self._estimate_non_controllable_base_power()
        base_phase_kw = {phase: base_phase_kw.get(phase, 0.0) for phase in phase_names}

        controls = {}
        adjusted_actions = None if actions is None else dict(actions)

        for charger_id, action in (actions or {}).items():
            charger = building._charger_lookup.get(charger_id)
            if charger is None:
                continue

            request_total_kw = self._charger_requested_power_kw(charger, action)
            if abs(request_total_kw) <= 1e-9:
                continue

            phase_connection = building._charger_phase_map.get(charger_id)
            request_phase_kw = self._split_power_by_connection(request_total_kw, phase_connection)
            controls[charger_id] = {
                'request_total_kw': request_total_kw,
                'request_phase_kw': request_phase_kw,
            }

        storage_control_id = '__electrical_storage__'
        request_storage_kw = self._storage_requested_power_kw(electrical_storage_action)
        if abs(request_storage_kw) > 1e-9:
            request_phase_kw = self._split_power_by_connection(request_storage_kw, building.electrical_storage_phase_connection)
            controls[storage_control_id] = {
                'request_total_kw': request_storage_kw,
                'request_phase_kw': request_phase_kw,
            }

        scales = {control_id: 1.0 for control_id in controls}
        total_limits = building._electrical_service_limits.get('total', {})
        per_phase_limits = building._electrical_service_limits.get('per_phase', {})

        for _ in range(8):
            changed = False
            total_kw, phase_kw = self._compute_totals(base_total_kw, base_phase_kw, controls, scales)

            changed |= self._scale_for_import_scope(
                total_kw,
                total_limits.get('import_kw'),
                controls,
                scales,
                component_getter=lambda c: c['request_total_kw'],
            )
            changed |= self._scale_for_export_scope(
                total_kw,
                total_limits.get('export_kw'),
                controls,
                scales,
                component_getter=lambda c: c['request_total_kw'],
            )

            for phase_name in phase_names:
                phase_limit = per_phase_limits.get(phase_name, {})
                changed |= self._scale_for_import_scope(
                    phase_kw.get(phase_name, 0.0),
                    phase_limit.get('import_kw'),
                    controls,
                    scales,
                    component_getter=lambda c, p=phase_name: c['request_phase_kw'].get(p, 0.0),
                )
                changed |= self._scale_for_export_scope(
                    phase_kw.get(phase_name, 0.0),
                    phase_limit.get('export_kw'),
                    controls,
                    scales,
                    component_getter=lambda c, p=phase_name: c['request_phase_kw'].get(p, 0.0),
                )

            if not changed:
                break

        total_kw, phase_kw = self._compute_totals(base_total_kw, base_phase_kw, controls, scales)
        total_kw = self._safe_scalar(total_kw, 0.0)
        phase_kw = {phase: self._safe_scalar(value, 0.0) for phase, value in phase_kw.items()}

        if adjusted_actions is not None:
            for charger_id in adjusted_actions:
                charger = building._charger_lookup.get(charger_id)
                if charger is None:
                    continue
                control = controls.get(charger_id)
                target_kw = 0.0 if control is None else control['request_total_kw'] * scales.get(charger_id, 1.0)
                adjusted_actions[charger_id] = self._charger_action_from_power_kw(charger, target_kw)

        adjusted_storage_action = electrical_storage_action
        if electrical_storage_action is not None:
            storage_control = controls.get(storage_control_id)
            target_kw = 0.0 if storage_control is None else storage_control['request_total_kw'] * scales.get(storage_control_id, 1.0)
            adjusted_storage_action = self._storage_action_from_power_kw(target_kw)

        violation_kw = 0.0
        import_limit = self._safe_scalar(total_limits.get('import_kw'), np.nan)
        export_limit = self._safe_scalar(total_limits.get('export_kw'), np.nan)
        if np.isfinite(import_limit):
            violation_kw += max(total_kw - import_limit, 0.0)
        if np.isfinite(export_limit):
            violation_kw += max(-total_kw - export_limit, 0.0)

        phase_headroom = {}
        phase_export_headroom = {}
        for phase_name in phase_names:
            phase_total = self._safe_scalar(phase_kw.get(phase_name, 0.0), 0.0)
            phase_limit = per_phase_limits.get(phase_name, {})
            phase_import_limit = self._safe_scalar(phase_limit.get('import_kw'), np.nan)
            phase_export_limit = self._safe_scalar(phase_limit.get('export_kw'), np.nan)

            phase_headroom[phase_name] = None if not np.isfinite(phase_import_limit) else (phase_import_limit - phase_total)
            phase_export_headroom[phase_name] = None if not np.isfinite(phase_export_limit) else (phase_export_limit + phase_total)

            if np.isfinite(phase_import_limit):
                violation_kw += max(phase_total - phase_import_limit, 0.0)
            if np.isfinite(phase_export_limit):
                violation_kw += max(-phase_total - phase_export_limit, 0.0)

        building_headroom = None if not np.isfinite(import_limit) else (import_limit - total_kw)
        building_export_headroom = None if not np.isfinite(export_limit) else (export_limit + total_kw)
        building._charging_constraints_state = {
            'building_headroom_kw': building_headroom,
            'building_export_headroom_kw': building_export_headroom,
            'phase_headroom_kw': phase_headroom,
            'phase_export_headroom_kw': phase_export_headroom,
            'total_power_kw': total_kw,
            'phase_power_kw': phase_kw,
        }

        penalty_kwh = self._safe_scalar(violation_kw * (building.seconds_per_time_step / 3600.0), 0.0)
        building._charging_constraint_penalty_kwh = penalty_kwh
        building._charging_constraint_last_penalty_kwh = penalty_kwh
        building._record_charging_constraint_state(
            violation_kwh=penalty_kwh,
            total_power_kw=float(total_kw),
            phase_power_kw=phase_kw,
        )

        return adjusted_actions, adjusted_storage_action

    def apply_charging_constraints_to_actions(
        self,
        actions: Optional[Mapping[str, float]],
        electrical_storage_action: Optional[float] = None,
    ) -> Tuple[Optional[Mapping[str, float]], Optional[float]]:
        """Apply configured electrical constraints and return adjusted EV/storage actions."""

        building = self.building

        building._charging_constraint_penalty_kwh = 0.0
        building._charging_constraint_last_penalty_kwh = 0.0

        if not building._charging_constraints_enabled:
            return actions, electrical_storage_action

        if getattr(building, '_electrical_service_enabled', False):
            return self._apply_electrical_service_constraints(actions, electrical_storage_action)

        adjusted_actions = self._apply_legacy_charging_constraints(actions)
        return adjusted_actions, electrical_storage_action
