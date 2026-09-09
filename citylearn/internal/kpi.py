from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from citylearn.cost_function import CostFunction
from citylearn.data import EnergySimulation, ZERO_DIVISION_PLACEHOLDER

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class CityLearnKPIService:
    """Internal KPI/evaluation service for `CityLearnEnv`."""

    EV_DEPARTURE_WITHIN_TOLERANCE_DEFAULT = 0.05
    EV_DEPARTURE_SERVICE_TOLERANCE_DEFAULT = 0.05
    EV_DEPARTURE_EPS = 1.0e-6
    ELECTRICAL_RESIDUAL_EPS_KW = 1.0e-5
    SCORECARD_DEFAULT_KPIS: Tuple[str, ...] = (
        # Cost scorecard values and BAU reference.
        'district_cost_total_control_eur',
        'district_cost_total_business_as_usual_eur',
        'district_cost_total_delta_to_business_as_usual_eur',
        'building_cost_total_control_eur',
        'building_cost_total_business_as_usual_eur',
        'building_cost_total_delta_to_business_as_usual_eur',

        # EV service and grid-safety values are judged directly, not vs BAU.
        'district_ev_performance_departure_min_acceptable_feasible_ratio',
        'district_ev_performance_departure_within_tolerance_feasible_ratio',
        'district_electrical_service_phase_violations_energy_total_kwh',
        'district_electrical_service_phase_requested_pressure_energy_total_kwh',
        'building_ev_performance_departure_min_acceptable_feasible_ratio',
        'building_ev_performance_departure_within_tolerance_feasible_ratio',
        'building_electrical_service_phase_violations_energy_total_kwh',
        'building_electrical_service_phase_requested_pressure_energy_total_kwh',

        # Battery and V2G totals with BAU reference rows.
        'district_battery_total_throughput_kwh',
        'district_battery_total_throughput_business_as_usual_kwh',
        'district_battery_total_throughput_delta_to_business_as_usual_kwh',
        'district_ev_total_v2g_export_kwh',
        'district_ev_total_v2g_export_business_as_usual_kwh',
        'district_ev_total_v2g_export_delta_to_business_as_usual_kwh',
        'building_battery_total_throughput_kwh',
        'building_battery_total_throughput_business_as_usual_kwh',
        'building_battery_total_throughput_delta_to_business_as_usual_kwh',
        'building_ev_total_v2g_export_kwh',
        'building_ev_total_v2g_export_business_as_usual_kwh',
        'building_ev_total_v2g_export_delta_to_business_as_usual_kwh',

        # Solar self-consumption with BAU reference rows.
        'district_solar_self_consumption_ratio_self_consumption_ratio',
        'district_solar_self_consumption_ratio_self_consumption_business_as_usual_ratio',
        'district_solar_self_consumption_ratio_self_consumption_delta_to_business_as_usual_ratio',
        'building_solar_self_consumption_ratio_self_consumption_ratio',
        'building_solar_self_consumption_ratio_self_consumption_business_as_usual_ratio',
        'building_solar_self_consumption_ratio_self_consumption_delta_to_business_as_usual_ratio',

        # Community/building grid exchange values with BAU reference rows.
        'district_energy_grid_total_import_control_kwh',
        'district_energy_grid_total_import_business_as_usual_kwh',
        'district_energy_grid_total_import_delta_to_business_as_usual_kwh',
        'district_energy_grid_total_export_control_kwh',
        'district_energy_grid_total_export_business_as_usual_kwh',
        'district_energy_grid_total_export_delta_to_business_as_usual_kwh',
        'district_energy_grid_total_net_exchange_control_kwh',
        'district_energy_grid_total_net_exchange_business_as_usual_kwh',
        'district_energy_grid_total_net_exchange_delta_to_business_as_usual_kwh',
        'building_energy_grid_total_import_control_kwh',
        'building_energy_grid_total_import_business_as_usual_kwh',
        'building_energy_grid_total_import_delta_to_business_as_usual_kwh',
        'building_energy_grid_total_export_control_kwh',
        'building_energy_grid_total_export_business_as_usual_kwh',
        'building_energy_grid_total_export_delta_to_business_as_usual_kwh',
        'building_energy_grid_total_net_exchange_control_kwh',
        'building_energy_grid_total_net_exchange_business_as_usual_kwh',
        'building_energy_grid_total_net_exchange_delta_to_business_as_usual_kwh',

        # District-only grid-shape ratios used by the community scorecard.
        'district_energy_grid_shape_quality_peak_daily_average_to_business_as_usual_ratio',
        'district_energy_grid_shape_quality_peak_all_time_average_to_business_as_usual_ratio',
        'district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_business_as_usual_ratio',
    )
    LEGACY_COST_FUNCTIONS = {
        'all_time_peak_average',
        'annual_normalized_unserved_energy_total',
        'carbon_emissions_total',
        'cost_total',
        'daily_one_minus_load_factor_average',
        'daily_peak_average',
        'discomfort_cold_delta_average',
        'discomfort_cold_delta_maximum',
        'discomfort_cold_delta_minimum',
        'discomfort_cold_proportion',
        'discomfort_hot_delta_average',
        'discomfort_hot_delta_maximum',
        'discomfort_hot_delta_minimum',
        'discomfort_hot_proportion',
        'discomfort_proportion',
        'electricity_consumption_total',
        'monthly_one_minus_load_factor_average',
        'one_minus_thermal_resilience_proportion',
        'power_outage_normalized_unserved_energy_total',
        'ramping_average',
        'zero_net_energy',
    }

    def __init__(self, env: "CityLearnEnv"):
        self.env = env

    @staticmethod
    def _to_scalar(value, default: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(scalar):
            return float(default)

        return scalar

    @staticmethod
    def _safe_div(control_value: float, baseline_value: float):
        c = CityLearnKPIService._to_scalar(control_value, 0.0)
        b = CityLearnKPIService._to_scalar(baseline_value, 0.0)
        eps = float(ZERO_DIVISION_PLACEHOLDER)

        if abs(b) <= eps:
            return 1.0 if abs(c) <= eps else None

        return c / b

    @staticmethod
    def _window_steps(window_seconds: float, seconds_per_time_step: float) -> int:
        step_seconds = max(float(seconds_per_time_step), 1.0)
        return max(1, int(round(float(window_seconds) / step_seconds)))

    @staticmethod
    def _simulated_days(env: "CityLearnEnv") -> float:
        steps = max(int(getattr(env, 'time_step', 0)), 1)
        step_seconds = max(float(getattr(env, 'seconds_per_time_step', 0) or 0), 1.0)
        return (steps * step_seconds) / (24.0 * 3600.0)

    @staticmethod
    def _window_days(t_start: int, t_final: int, seconds_per_time_step: float) -> float:
        if t_final < t_start:
            return 0.0
        steps = max(int(t_final) - int(t_start) + 1, 0)
        step_seconds = max(float(seconds_per_time_step or 0.0), 1.0)
        return (steps * step_seconds) / (24.0 * 3600.0)

    @staticmethod
    def _daily_average(total_value: float, simulated_days: float) -> Optional[float]:
        value = CityLearnKPIService._to_scalar(total_value, np.nan)

        if not np.isfinite(value):
            return None

        if simulated_days <= float(ZERO_DIVISION_PLACEHOLDER):
            return None

        return float(value / simulated_days)

    @staticmethod
    def _normalize_soc_target(value) -> Optional[float]:
        try:
            target = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(target):
            return None

        if target > 1.0 and target <= 100.0:
            target = target / 100.0

        if target < 0.0 or target > 1.0:
            return None

        return float(target)

    @staticmethod
    def _safe_sequence_value(sequence, index: int, default=None):
        if sequence is None or index is None or index < 0:
            return default

        try:
            if index >= len(sequence):
                return default
            return sequence[index]
        except (TypeError, IndexError):
            return default

    @staticmethod
    def _normal_ev_id(value) -> Optional[str]:
        if not isinstance(value, str):
            return None

        value = value.strip()
        if value == '' or value.lower() == 'nan':
            return None

        return value

    @staticmethod
    def _interp_curve(curve, x: float, default: Optional[float] = None) -> Optional[float]:
        try:
            arr = np.array(curve, dtype='float64')
        except (TypeError, ValueError):
            return default

        if arr.ndim != 2:
            return default

        if arr.shape[0] != 2 and arr.shape[1] == 2:
            arr = arr.T

        if arr.shape[0] != 2:
            return default

        xs = arr[0]
        ys = arr[1]
        finite = np.isfinite(xs) & np.isfinite(ys)

        if not np.any(finite):
            return default

        xs = xs[finite]
        ys = ys[finite]
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        return float(np.interp(float(x), xs, ys))

    def _ev_connected_interval_start(self, sim, states: np.ndarray, departure_index: int) -> int:
        ev_ids = getattr(sim, 'electric_vehicle_id', None)
        session_ids = getattr(sim, 'electric_vehicle_session_id', None)
        current_ev_id = self._normal_ev_id(self._safe_sequence_value(ev_ids, departure_index))
        current_session_id = self._normal_ev_id(
            self._safe_sequence_value(session_ids, departure_index)
        )
        start = int(departure_index)

        while start > 0:
            previous_state = self._to_scalar(states[start - 1], np.nan)

            if previous_state != 1:
                break

            previous_ev_id = self._normal_ev_id(self._safe_sequence_value(ev_ids, start - 1))
            previous_session_id = self._normal_ev_id(
                self._safe_sequence_value(session_ids, start - 1)
            )

            if (
                current_ev_id is not None
                and previous_ev_id is not None
                and previous_ev_id != current_ev_id
            ):
                break
            if (
                current_session_id is not None
                and previous_session_id is not None
                and previous_session_id != current_session_id
            ):
                break

            start -= 1

        return start

    def _ev_arrival_soc_for_interval(self, sim, ev, states: np.ndarray, interval_start: int) -> Optional[float]:
        estimated_soc_arrival = getattr(sim, 'electric_vehicle_estimated_soc_arrival', None)
        current_soc_reference = getattr(sim, 'electric_vehicle_current_soc', None)
        ev_ids = getattr(sim, 'electric_vehicle_id', None)
        current_ev_id = self._normal_ev_id(self._safe_sequence_value(ev_ids, interval_start))
        # This is the same boundary reference used by the runtime for a normal
        # arrival, a partial-window reset, or a topology activation that exposes
        # an already occupied charger.  It is a boundary value only, not an
        # exogenous trajectory that may overwrite subsequent control.
        candidates = [self._safe_sequence_value(current_soc_reference, interval_start)]

        if interval_start > 0:
            previous_state = self._to_scalar(states[interval_start - 1], np.nan)
            previous_ev_id = self._normal_ev_id(self._safe_sequence_value(ev_ids, interval_start - 1))

            if (
                previous_state in (2, 3)
                and (
                    current_ev_id is None
                    or previous_ev_id is None
                    or previous_ev_id == current_ev_id
                )
            ):
                candidates.append(self._safe_sequence_value(estimated_soc_arrival, interval_start - 1))

        candidates.append(self._safe_sequence_value(estimated_soc_arrival, interval_start))

        battery = getattr(ev, 'battery', None)
        battery_soc = getattr(battery, 'soc', None)

        if interval_start == 0:
            candidates.append(getattr(battery, 'initial_soc', None))
        else:
            candidates.append(self._safe_sequence_value(battery_soc, interval_start - 1))

        candidates.append(self._safe_sequence_value(battery_soc, interval_start))

        for candidate in candidates:
            soc = self._normalize_soc_target(candidate)
            if soc is not None:
                return soc

        return None

    def _ev_battery_max_input_power_kw(self, battery, charger_power_kw: float, soc: float) -> float:
        charger_power_kw = max(float(charger_power_kw), 0.0)
        nominal_power_kw = self._to_scalar(getattr(battery, 'nominal_power', np.nan), np.nan)

        if not np.isfinite(nominal_power_kw) or nominal_power_kw <= 0.0:
            return charger_power_kw

        multiplier = self._interp_curve(getattr(battery, 'capacity_power_curve', None), soc, default=1.0)
        multiplier = 1.0 if multiplier is None or not np.isfinite(multiplier) else max(float(multiplier), 0.0)
        return min(charger_power_kw, nominal_power_kw * multiplier)

    def _ev_battery_charge_round_trip_efficiency(self, battery, power_kw: float) -> float:
        nominal_power_kw = self._to_scalar(getattr(battery, 'nominal_power', np.nan), np.nan)
        normalized_power = (
            float(power_kw) / nominal_power_kw
            if np.isfinite(nominal_power_kw) and nominal_power_kw > 0.0
            else 1.0
        )
        efficiency = self._interp_curve(
            getattr(battery, 'power_efficiency_curve', None),
            np.clip(normalized_power, 0.0, 1.0),
            default=None,
        )

        if efficiency is None or not np.isfinite(efficiency):
            efficiency = self._to_scalar(getattr(battery, 'efficiency', np.nan), np.nan)

        if efficiency is None or not np.isfinite(efficiency):
            round_trip = self._to_scalar(getattr(battery, 'round_trip_efficiency', np.nan), np.nan)
            return float(np.clip(round_trip, 0.0, 1.0)) if np.isfinite(round_trip) else 1.0

        return float(np.sqrt(np.clip(efficiency, 0.0, 1.0)))

    def _ev_charger_charge_efficiency(self, charger, power_kw: float, max_power_kw: float) -> float:
        normalized_power = float(power_kw) / max(float(max_power_kw), ZERO_DIVISION_PLACEHOLDER)
        get_efficiency = getattr(charger, 'get_efficiency', None)

        if callable(get_efficiency):
            try:
                return float(np.clip(get_efficiency(np.clip(normalized_power, 0.0, 1.0), True), 0.0, 1.0))
            except (TypeError, ValueError):
                pass

        return float(np.clip(self._to_scalar(getattr(charger, 'efficiency', 1.0), 1.0), 0.0, 1.0))

    @staticmethod
    def _normalize_phase_connection_label(value) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        if text == '':
            return None

        normalized = text.lower().replace('-', '_')
        aliases = {
            'l1': 'L1',
            'phase_1': 'L1',
            'l2': 'L2',
            'phase_2': 'L2',
            'l3': 'L3',
            'phase_3': 'L3',
            'all_phases': 'all_phases',
            'three_phase': 'all_phases',
        }
        return aliases.get(normalized, text)

    def _electrical_service_phase_names(self, building) -> List[str]:
        if getattr(building, '_electrical_service_mode', 'single_phase') == 'three_phase':
            return ['L1', 'L2', 'L3']

        return ['L1']

    def _split_unassigned_service_power(self, building, power_kw: float) -> Dict[str, float]:
        phase_names = self._electrical_service_phase_names(building)

        if len(phase_names) == 1:
            return {'L1': float(power_kw)}

        split_mode = str(getattr(building, '_electrical_service_default_split', 'balanced')).strip().lower()
        if split_mode in {'l1', 'l2', 'l3'}:
            return {phase: float(power_kw if phase.lower() == split_mode else 0.0) for phase in phase_names}

        share = float(power_kw) / len(phase_names)
        return {phase: share for phase in phase_names}

    def _split_service_power_by_connection(
        self,
        building,
        power_kw: float,
        phase_connection: Optional[str],
    ) -> Dict[str, float]:
        phase_names = self._electrical_service_phase_names(building)

        if len(phase_names) == 1:
            return {'L1': float(power_kw)}

        phase_connection = self._normalize_phase_connection_label(phase_connection)
        if phase_connection in {'L1', 'L2', 'L3'}:
            return {phase: float(power_kw if phase == phase_connection else 0.0) for phase in phase_names}

        if phase_connection == 'all_phases':
            share = float(power_kw) / len(phase_names)
            return {phase: share for phase in phase_names}

        return self._split_unassigned_service_power(building, power_kw)

    def _ev_power_outage_at_step(self, building, step: int) -> bool:
        ops_service = getattr(building, '_ops_service', None)
        outage_at_step = getattr(ops_service, '_power_outage_at_step', None)

        if callable(outage_at_step):
            try:
                return bool(outage_at_step(step))
            except Exception:
                pass

        signal = getattr(building, 'power_outage_signal', None)
        if signal is None:
            energy_simulation = getattr(building, 'energy_simulation', None)
            signal = getattr(energy_simulation, 'power_outage', None)

        return bool(self._safe_sequence_value(signal, step, 0.0))

    def _estimate_feasibility_base_power_at_step(self, building, step: int) -> Tuple[float, Dict[str, float]]:
        ops_service = getattr(building, '_ops_service', None)
        estimate = getattr(ops_service, '_estimate_base_power_at_step', None)

        if callable(estimate):
            try:
                total_kw, phase_kw = estimate(step)
                total_kw = self._to_scalar(total_kw, 0.0)
                phase_kw = {
                    phase: self._to_scalar(value, 0.0)
                    for phase, value in (phase_kw or {}).items()
                }
                return total_kw, phase_kw
            except Exception:
                pass

        phase_kw = {phase: 0.0 for phase in self._electrical_service_phase_names(building)}
        return 0.0, phase_kw

    def _ev_apply_charger_charge_deadband(self, charger, power_kw: float) -> float:
        power_kw = max(self._to_scalar(power_kw, 0.0), 0.0)
        if power_kw <= self.EV_DEPARTURE_EPS:
            return 0.0

        max_power_kw = max(self._to_scalar(getattr(charger, 'max_charging_power', power_kw), power_kw), 0.0)
        min_power_kw = max(self._to_scalar(getattr(charger, 'min_charging_power', 0.0), 0.0), 0.0)
        if max_power_kw > 0.0:
            min_power_kw = min(min_power_kw, max_power_kw)

        if min_power_kw > 0.0 and power_kw < min_power_kw:
            return 0.0

        return power_kw

    def _ev_constraint_limited_charge_power_kw(
        self,
        building,
        charger,
        step: int,
        requested_power_kw: float,
    ) -> float:
        requested_power_kw = max(self._to_scalar(requested_power_kw, 0.0), 0.0)

        if requested_power_kw <= self.EV_DEPARTURE_EPS:
            return 0.0

        if building is None or not getattr(building, '_charging_constraints_enabled', False):
            return self._ev_apply_charger_charge_deadband(charger, requested_power_kw)

        if self._ev_power_outage_at_step(building, step):
            return 0.0

        if getattr(building, '_electrical_service_enabled', False):
            base_total_kw, base_phase_kw = self._estimate_feasibility_base_power_at_step(building, step)
            total_limits = getattr(building, '_electrical_service_limits', {}).get('total', {}) or {}
            per_phase_limits = getattr(building, '_electrical_service_limits', {}).get('per_phase', {}) or {}
            cap_kw = requested_power_kw

            import_limit = self._to_scalar(total_limits.get('import_kw'), np.nan)
            if np.isfinite(import_limit):
                cap_kw = min(cap_kw, max(import_limit - base_total_kw, 0.0))

            charger_id = getattr(charger, 'charger_id', None)
            phase_connection = (getattr(building, '_charger_phase_map', {}) or {}).get(
                charger_id,
                getattr(charger, 'phase_connection', None),
            )
            request_phase_fraction = self._split_service_power_by_connection(building, 1.0, phase_connection)

            for phase_name, fraction in request_phase_fraction.items():
                fraction = self._to_scalar(fraction, 0.0)
                if fraction <= self.EV_DEPARTURE_EPS:
                    continue

                phase_limit = self._to_scalar(
                    (per_phase_limits.get(phase_name, {}) or {}).get('import_kw'),
                    np.nan,
                )
                if np.isfinite(phase_limit):
                    phase_base_kw = self._to_scalar(base_phase_kw.get(phase_name, 0.0), 0.0)
                    cap_kw = min(cap_kw, max((phase_limit - phase_base_kw) / fraction, 0.0))

            return self._ev_apply_charger_charge_deadband(charger, min(requested_power_kw, cap_kw))

        cap_kw = requested_power_kw
        building_limit = self._to_scalar(getattr(building, '_building_charger_limit_kw', None), np.nan)
        if np.isfinite(building_limit):
            cap_kw = min(cap_kw, max(building_limit, 0.0))

        charger_id = getattr(charger, 'charger_id', None)
        charger_phase_map = getattr(building, '_charger_phase_map', {}) or {}
        charger_phase = charger_phase_map.get(charger_id)

        for phase in getattr(building, '_phase_limits', []) or []:
            chargers = phase.get('chargers', []) or []
            if charger_id not in chargers and phase.get('name') != charger_phase:
                continue

            phase_limit = self._to_scalar(phase.get('limit_kw', phase.get('import_kw')), np.nan)
            if np.isfinite(phase_limit):
                cap_kw = min(cap_kw, max(phase_limit, 0.0))

        return self._ev_apply_charger_charge_deadband(charger, min(requested_power_kw, cap_kw))

    def _ev_departure_threshold_feasible(
        self,
        building,
        charger,
        sim,
        ev,
        states: np.ndarray,
        departure_index: int,
        threshold_soc: float,
    ) -> bool:
        """Return whether a departure threshold was reachable with max charging.

        Missing charger/battery/arrival data is treated as feasible to preserve
        existing KPI semantics for legacy datasets that cannot support this audit.
        """

        threshold_soc = self._normalize_soc_target(threshold_soc)

        if threshold_soc is None or threshold_soc <= 0.0:
            return True

        battery = getattr(ev, 'battery', None)

        if battery is None:
            return True

        capacity_kwh = self._to_scalar(getattr(battery, 'capacity', np.nan), np.nan)
        charger_power_kw = self._to_scalar(getattr(charger, 'max_charging_power', np.nan), np.nan)

        if (
            not np.isfinite(capacity_kwh)
            or capacity_kwh <= 0.0
            or not np.isfinite(charger_power_kw)
            or charger_power_kw < 0.0
        ):
            return True

        interval_start = self._ev_connected_interval_start(sim, states, departure_index)
        arrival_soc = self._ev_arrival_soc_for_interval(sim, ev, states, interval_start)

        if arrival_soc is None:
            return True

        if arrival_soc + self.EV_DEPARTURE_EPS >= threshold_soc:
            return True

        seconds_per_time_step = self._to_scalar(
            getattr(
                charger,
                'seconds_per_time_step',
                getattr(battery, 'seconds_per_time_step', getattr(self.env, 'seconds_per_time_step', 3600.0)),
            ),
            3600.0,
        )
        step_hours = max(seconds_per_time_step / 3600.0, 0.0)

        if step_hours <= 0.0:
            return True

        soc = float(arrival_soc)
        connected_steps = max(int(departure_index) - int(interval_start) + 1, 0)

        for offset in range(connected_steps):
            if soc + self.EV_DEPARTURE_EPS >= threshold_soc:
                return True

            step = int(interval_start) + offset
            battery_limited_power_kw = self._ev_battery_max_input_power_kw(battery, charger_power_kw, soc)
            power_kw = self._ev_constraint_limited_charge_power_kw(
                building,
                charger,
                step,
                battery_limited_power_kw,
            )

            if power_kw <= self.EV_DEPARTURE_EPS:
                break

            charger_efficiency = self._ev_charger_charge_efficiency(charger, power_kw, charger_power_kw)
            battery_round_trip_efficiency = self._ev_battery_charge_round_trip_efficiency(battery, power_kw)
            delta_soc = (
                power_kw
                * step_hours
                * charger_efficiency
                * battery_round_trip_efficiency
                / capacity_kwh
            )

            if delta_soc <= self.EV_DEPARTURE_EPS:
                break

            soc = float(np.clip(soc + delta_soc, 0.0, 1.0))

        return bool(soc + self.EV_DEPARTURE_EPS >= threshold_soc)

    def _ev_departure_within_tolerance(self) -> float:
        return max(
            self._to_scalar(
                getattr(self.env, 'ev_departure_within_tolerance', self.EV_DEPARTURE_WITHIN_TOLERANCE_DEFAULT),
                self.EV_DEPARTURE_WITHIN_TOLERANCE_DEFAULT,
            ),
            0.0,
        )

    def _ev_departure_service_tolerance(self) -> float:
        return max(
            self._to_scalar(
                getattr(self.env, 'ev_departure_service_tolerance', self.EV_DEPARTURE_SERVICE_TOLERANCE_DEFAULT),
                self.EV_DEPARTURE_SERVICE_TOLERANCE_DEFAULT,
            ),
            0.0,
        )

    @staticmethod
    def _metric(cost_function: str, value, name: str, level: str) -> Dict[str, object]:
        return {
            'cost_function': cost_function,
            'value': value,
            'name': name,
            'level': level,
        }

    @staticmethod
    def _sum_finite(values) -> float:
        try:
            series = np.array(values, dtype='float64').flatten()
        except (TypeError, ValueError):
            return 0.0

        if series.size == 0:
            return 0.0

        finite = series[np.isfinite(series)]
        if finite.size == 0:
            return 0.0

        return float(finite.sum())

    @staticmethod
    def _as_float_array(values) -> np.ndarray:
        try:
            return np.array(values, dtype='float64').flatten()
        except (TypeError, ValueError):
            return np.zeros((0,), dtype='float64')

    @classmethod
    def _positive_total(cls, values) -> float:
        series = cls._as_float_array(values)
        if series.size == 0:
            return 0.0
        return cls._sum_finite(np.clip(series, 0.0, None))

    @classmethod
    def _net_total(cls, values) -> float:
        return cls._sum_finite(values)

    @classmethod
    def _ramping_final(cls, values, down_ramp: bool = None, net_export: bool = None) -> float:
        series = cls._as_float_array(values)
        if series.size == 0:
            return 0.0

        down_ramp = False if down_ramp is None else down_ramp
        net_export = True if net_export is None else net_export
        ramp = np.empty_like(series, dtype='float64')
        ramp[0] = np.nan
        if series.size > 1:
            ramp[1:] = series[1:] - series[:-1]

        if down_ramp:
            ramp = np.abs(ramp)
        else:
            ramp = np.clip(ramp, 0.0, None)

        if not net_export:
            ramp = np.where(series < 0.0, 0.0, ramp)

        return cls._sum_finite(ramp)

    @staticmethod
    def _group_slices(values: np.ndarray, window: int):
        window = max(int(window), 1)
        for start in range(0, values.size, window):
            yield values[start:start + window]

    @classmethod
    def _nan_mean_preserve_inf(cls, values: np.ndarray) -> float:
        values = cls._as_float_array(values)
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            return float(np.nan)
        return float(np.mean(valid))

    @classmethod
    def _one_minus_load_factor_final(cls, values, window: int = None) -> float:
        series = cls._as_float_array(values)
        if series.size == 0:
            return 0.0

        window = 730 if window is None else window
        penalties = []
        for group in cls._group_slices(series, window):
            finite = group[np.isfinite(group)]
            if finite.size == 0:
                penalties.append(np.nan)
                continue
            peak = float(np.max(finite))
            mean = float(np.mean(finite))
            with np.errstate(divide='ignore', invalid='ignore'):
                penalties.append(float(1.0 - np.divide(mean, peak)))

        return cls._nan_mean_preserve_inf(np.array(penalties, dtype='float64'))

    @classmethod
    def _peak_final(cls, values, window: int = None) -> float:
        series = cls._as_float_array(values)
        if series.size == 0:
            return 0.0

        window = 24 if window is None else window
        peaks = []
        for group in cls._group_slices(series, window):
            finite = group[np.isfinite(group)]
            peaks.append(np.nan if finite.size == 0 else float(np.max(finite)))

        return cls._nan_mean_preserve_inf(np.array(peaks, dtype='float64'))

    @staticmethod
    def _equity_relative_benefit_percent(cost_scenario: float, cost_baseline: float) -> Optional[float]:
        scenario = CityLearnKPIService._to_scalar(cost_scenario, np.nan)
        baseline = CityLearnKPIService._to_scalar(cost_baseline, np.nan)

        if not np.isfinite(scenario) or not np.isfinite(baseline) or baseline <= 0.0:
            return None

        return float(100.0 * (baseline - scenario) / baseline)

    @staticmethod
    def _equity_distribution_metrics(relative_benefits: np.ndarray) -> Dict[str, Optional[float]]:
        benefits = np.array(relative_benefits, dtype='float64')
        benefits = benefits[np.isfinite(benefits)]

        if benefits.size == 0:
            return {
                'equity_gini_benefit': None,
                'equity_cr20_benefit': None,
                'equity_losers_percent': None,
            }

        losers_percent = float(100.0 * np.count_nonzero(benefits < 0.0) / benefits.size)
        benefits_plus = np.clip(benefits, 0.0, None)
        total_plus = float(benefits_plus.sum())

        if total_plus <= 0.0:
            return {
                'equity_gini_benefit': None,
                'equity_cr20_benefit': None,
                'equity_losers_percent': losers_percent,
            }

        n = benefits_plus.size
        diff_sum = float(np.abs(benefits_plus[:, None] - benefits_plus[None, :]).sum())
        gini = float(diff_sum / (2.0 * n * total_plus))

        k = max(1, int(np.ceil(0.2 * n)))
        top_sum = float(np.sort(benefits_plus)[::-1][:k].sum())
        cr20 = float(top_sum / total_plus)

        return {
            'equity_gini_benefit': gini,
            'equity_cr20_benefit': cr20,
            'equity_losers_percent': losers_percent,
        }

    @staticmethod
    def _equity_bpr(
        non_negative_relative_benefits: Mapping[str, float],
        groups: Mapping[str, Optional[str]],
    ) -> Optional[float]:
        if len(non_negative_relative_benefits) == 0:
            return None

        asset_poor_values = []
        asset_rich_values = []

        for building_name, value in non_negative_relative_benefits.items():
            group = groups.get(building_name)

            if group == 'asset_poor':
                asset_poor_values.append(float(value))
            elif group == 'asset_rich':
                asset_rich_values.append(float(value))
            else:
                return None

        if len(asset_poor_values) == 0 or len(asset_rich_values) == 0:
            return None

        rich_mean = float(np.mean(asset_rich_values))
        poor_mean = float(np.mean(asset_poor_values))

        if rich_mean <= 0.0:
            return None

        return float(poor_mean / rich_mean)

    def _compute_ev_metrics(self, building, *, t_start: int = 0, t_final: Optional[int] = None) -> Dict[str, float]:
        upper = int(max(building.time_step, 0))
        t_start = int(max(t_start, 0))
        t_final = upper if t_final is None else int(min(max(t_final, 0), upper))
        within_tolerance = self._ev_departure_within_tolerance()
        service_tolerance = self._ev_departure_service_tolerance()
        if t_final < t_start:
            return {
                'departures_total': 0.0,
                'departures_met': 0.0,
                'departures_min_acceptable': 0.0,
                'departures_within_tolerance': 0.0,
                'departures_target_feasible': 0.0,
                'departures_target_infeasible': 0.0,
                'departures_min_acceptable_feasible': 0.0,
                'departures_min_acceptable_infeasible': 0.0,
                'departures_within_tolerance_feasible': 0.0,
                'departures_within_tolerance_infeasible': 0.0,
                '_departures_met_target_feasible': 0.0,
                '_departures_min_acceptable_feasible_met': 0.0,
                '_departures_within_tolerance_feasible_met': 0.0,
                'departure_deficit_sum': 0.0,
                'departure_shortfall_beyond_tolerance_sum': 0.0,
                'departure_surplus_sum': 0.0,
                'departure_absolute_error_sum': 0.0,
                'ev_departure_success_rate': None,
                'ev_departure_min_acceptable_rate': None,
                'ev_departure_within_tolerance_rate': None,
                'ev_departure_success_feasible_rate': None,
                'ev_departure_min_acceptable_feasible_rate': None,
                'ev_departure_within_tolerance_feasible_rate': None,
                'ev_departure_soc_deficit_mean': None,
                'ev_departure_shortfall_beyond_tolerance_mean': None,
                'ev_departure_soc_surplus_mean': None,
                'ev_departure_soc_absolute_error_mean': None,
                'ev_departure_tolerance_ratio': service_tolerance,
                'ev_charge_total_kwh': 0.0,
                'ev_v2g_export_total_kwh': 0.0,
                'ev_connected_soc_gain_total_kwh': 0.0,
                'ev_energy_accounting_shortfall_kwh': 0.0,
            }

        departures_total = 0
        departures_met = 0
        departures_min_acceptable = 0
        departures_within_tolerance = 0
        departures_target_feasible = 0
        departures_target_infeasible = 0
        departures_min_acceptable_feasible = 0
        departures_min_acceptable_infeasible = 0
        departures_within_tolerance_feasible = 0
        departures_within_tolerance_infeasible = 0
        departures_met_target_feasible = 0
        departures_min_acceptable_feasible_met = 0
        departures_within_tolerance_feasible_met = 0
        departure_deficit_sum = 0.0
        departure_shortfall_beyond_tolerance_sum = 0.0
        departure_surplus_sum = 0.0
        departure_absolute_error_sum = 0.0
        charge_total_kwh = 0.0
        v2g_export_total_kwh = 0.0
        connected_soc_gain_total_kwh = 0.0
        eps = self.EV_DEPARTURE_EPS

        for charger in self._chargers_for_metrics(building):
            consumption = np.array(charger.electricity_consumption[t_start:t_final + 1], dtype='float64')
            charge_total_kwh += self._sum_finite(np.clip(consumption, 0.0, None))
            v2g_export_total_kwh += self._sum_finite(np.clip(-consumption, 0.0, None))

            sim = charger.charger_simulation
            states = np.array(sim.electric_vehicle_charger_state, dtype='float64')
            required_soc = np.array(sim.electric_vehicle_required_soc_departure, dtype='float64')
            history_limit = min(
                t_final,
                len(states) - 1,
                len(required_soc) - 1,
                len(charger.past_connected_evs) - 1,
            )

            if history_limit < t_start:
                continue

            for t in range(t_start, history_limit + 1):
                current_state = states[t]
                next_state = states[t + 1] if t + 1 < len(states) else np.nan
                ev_ids = getattr(sim, 'electric_vehicle_id', None)
                session_ids = getattr(sim, 'electric_vehicle_session_id', None)
                current_ev_id = self._normal_ev_id(
                    self._safe_sequence_value(ev_ids, t)
                )
                next_ev_id = self._normal_ev_id(
                    self._safe_sequence_value(ev_ids, t + 1)
                )
                current_session_id = self._normal_ev_id(
                    self._safe_sequence_value(session_ids, t)
                )
                next_session_id = self._normal_ev_id(
                    self._safe_sequence_value(session_ids, t + 1)
                )
                departure_countdown = self._to_scalar(
                    self._safe_sequence_value(
                        getattr(sim, 'electric_vehicle_departure_time', None),
                        t,
                    ),
                    np.nan,
                )
                next_departure_countdown = self._to_scalar(
                    self._safe_sequence_value(
                        getattr(sim, 'electric_vehicle_departure_time', None),
                        t + 1,
                    ),
                    np.nan,
                )
                terminal_departure = (
                    t == len(states) - 1
                    and np.isfinite(departure_countdown)
                    and departure_countdown <= 1.0 + self.EV_DEPARTURE_EPS
                )
                departure_transition = t + 1 < len(states) and (
                    next_state != 1
                    or (
                        current_session_id is not None
                        and next_session_id is not None
                        and next_session_id != current_session_id
                    )
                    or (
                        current_ev_id is not None
                        and next_ev_id is not None
                        and next_ev_id != current_ev_id
                    )
                    or (
                        current_session_id is None
                        and next_session_id is None
                        and np.isfinite(departure_countdown)
                        # Legacy schedules without explicit session identity
                        # expose a back-to-back boundary through a countdown
                        # reset while the charger remains connected.  A plain
                        # ``countdown <= 1`` check also counts ordinary
                        # departures one interval early, so require the next
                        # session's countdown to increase explicitly.
                        and departure_countdown <= 1.0 + self.EV_DEPARTURE_EPS
                        and np.isfinite(next_departure_countdown)
                        and next_departure_countdown
                        > departure_countdown + self.EV_DEPARTURE_EPS
                    )
                )

                if current_state != 1 or not (departure_transition or terminal_departure):
                    continue

                ev = charger.past_connected_evs[t]
                if ev is None:
                    continue

                target_soc = self._normalize_soc_target(required_soc[t])
                if target_soc is None:
                    continue

                if t >= len(ev.battery.soc):
                    continue

                actual_soc = self._to_scalar(ev.battery.soc[t], np.nan)
                if not np.isfinite(actual_soc):
                    continue

                interval_start = self._ev_connected_interval_start(sim, states, t)
                arrival_soc = self._ev_arrival_soc_for_interval(
                    sim,
                    ev,
                    states,
                    interval_start,
                )
                capacity_kwh = self._to_scalar(
                    getattr(ev.battery, 'capacity', np.nan),
                    np.nan,
                )
                if (
                    arrival_soc is not None
                    and np.isfinite(capacity_kwh)
                    and capacity_kwh > 0.0
                ):
                    connected_soc_gain_total_kwh += (
                        max(actual_soc - float(arrival_soc), 0.0)
                        * capacity_kwh
                    )

                departures_total += 1
                deficit = max(target_soc - actual_soc, 0.0)
                surplus = max(actual_soc - target_soc, 0.0)
                absolute_error = abs(actual_soc - target_soc)
                shortfall_beyond_tolerance = max(target_soc - service_tolerance - actual_soc, 0.0)
                min_acceptable_threshold = max(target_soc - service_tolerance, 0.0)
                within_tolerance_threshold = max(target_soc - within_tolerance, 0.0)
                target_feasible = self._ev_departure_threshold_feasible(
                    building,
                    charger,
                    sim,
                    ev,
                    states,
                    t,
                    target_soc,
                )
                min_acceptable_feasible = self._ev_departure_threshold_feasible(
                    building,
                    charger,
                    sim,
                    ev,
                    states,
                    t,
                    min_acceptable_threshold,
                )
                within_tolerance_feasible = self._ev_departure_threshold_feasible(
                    building,
                    charger,
                    sim,
                    ev,
                    states,
                    t,
                    within_tolerance_threshold,
                )

                if absolute_error <= within_tolerance + eps:
                    departures_within_tolerance += 1
                if actual_soc + service_tolerance + eps >= target_soc:
                    departures_min_acceptable += 1
                departure_deficit_sum += deficit
                departure_surplus_sum += surplus
                departure_absolute_error_sum += absolute_error
                departure_shortfall_beyond_tolerance_sum += shortfall_beyond_tolerance
                if deficit <= eps:
                    departures_met += 1

                if target_feasible:
                    departures_target_feasible += 1
                    if deficit <= eps:
                        departures_met_target_feasible += 1
                else:
                    departures_target_infeasible += 1

                if min_acceptable_feasible:
                    departures_min_acceptable_feasible += 1
                    if actual_soc + service_tolerance + eps >= target_soc:
                        departures_min_acceptable_feasible_met += 1
                else:
                    departures_min_acceptable_infeasible += 1

                if within_tolerance_feasible:
                    departures_within_tolerance_feasible += 1
                    if absolute_error <= within_tolerance + eps:
                        departures_within_tolerance_feasible_met += 1
                else:
                    departures_within_tolerance_infeasible += 1

        success_rate = None if departures_total == 0 else departures_met / departures_total
        min_acceptable_rate = None if departures_total == 0 else departures_min_acceptable / departures_total
        within_tolerance_rate = None if departures_total == 0 else departures_within_tolerance / departures_total
        success_feasible_rate = None if departures_target_feasible == 0 else departures_met_target_feasible / departures_target_feasible
        min_acceptable_feasible_rate = (
            None if departures_min_acceptable_feasible == 0
            else departures_min_acceptable_feasible_met / departures_min_acceptable_feasible
        )
        within_tolerance_feasible_rate = (
            None if departures_within_tolerance_feasible == 0
            else departures_within_tolerance_feasible_met / departures_within_tolerance_feasible
        )
        deficit_mean = None if departures_total == 0 else departure_deficit_sum / departures_total
        shortfall_beyond_tolerance_mean = None if departures_total == 0 else departure_shortfall_beyond_tolerance_sum / departures_total
        surplus_mean = None if departures_total == 0 else departure_surplus_sum / departures_total
        absolute_error_mean = None if departures_total == 0 else departure_absolute_error_sum / departures_total

        return {
            'departures_total': float(departures_total),
            'departures_met': float(departures_met),
            'departures_min_acceptable': float(departures_min_acceptable),
            'departures_within_tolerance': float(departures_within_tolerance),
            'departures_target_feasible': float(departures_target_feasible),
            'departures_target_infeasible': float(departures_target_infeasible),
            'departures_min_acceptable_feasible': float(departures_min_acceptable_feasible),
            'departures_min_acceptable_infeasible': float(departures_min_acceptable_infeasible),
            'departures_within_tolerance_feasible': float(departures_within_tolerance_feasible),
            'departures_within_tolerance_infeasible': float(departures_within_tolerance_infeasible),
            '_departures_met_target_feasible': float(departures_met_target_feasible),
            '_departures_min_acceptable_feasible_met': float(departures_min_acceptable_feasible_met),
            '_departures_within_tolerance_feasible_met': float(departures_within_tolerance_feasible_met),
            'departure_deficit_sum': float(departure_deficit_sum),
            'departure_shortfall_beyond_tolerance_sum': float(departure_shortfall_beyond_tolerance_sum),
            'departure_surplus_sum': float(departure_surplus_sum),
            'departure_absolute_error_sum': float(departure_absolute_error_sum),
            'ev_departure_success_rate': success_rate,
            'ev_departure_min_acceptable_rate': min_acceptable_rate,
            'ev_departure_within_tolerance_rate': within_tolerance_rate,
            'ev_departure_success_feasible_rate': success_feasible_rate,
            'ev_departure_min_acceptable_feasible_rate': min_acceptable_feasible_rate,
            'ev_departure_within_tolerance_feasible_rate': within_tolerance_feasible_rate,
            'ev_departure_soc_deficit_mean': deficit_mean,
            'ev_departure_shortfall_beyond_tolerance_mean': shortfall_beyond_tolerance_mean,
            'ev_departure_soc_surplus_mean': surplus_mean,
            'ev_departure_soc_absolute_error_mean': absolute_error_mean,
            'ev_departure_tolerance_ratio': service_tolerance,
            'ev_charge_total_kwh': float(charge_total_kwh),
            'ev_v2g_export_total_kwh': float(v2g_export_total_kwh),
            'ev_connected_soc_gain_total_kwh': float(connected_soc_gain_total_kwh),
            'ev_energy_accounting_shortfall_kwh': float(max(
                connected_soc_gain_total_kwh - charge_total_kwh,
                0.0,
            )),
        }

    def _chargers_for_metrics(self, building) -> Sequence:
        topology_service = getattr(self.env, '_topology_service', None)
        if (
            getattr(self.env, 'topology_mode', 'static') == 'dynamic'
            and topology_service is not None
            and hasattr(topology_service, 'historical_chargers')
        ):
            chargers = list(topology_service.historical_chargers(building.name))
            if chargers:
                return chargers

        return list(building.electric_vehicle_chargers or [])

    def _deferrable_appliances_for_metrics(self, building) -> Sequence:
        topology_service = getattr(self.env, '_topology_service', None)
        if (
            getattr(self.env, 'topology_mode', 'static') == 'dynamic'
            and topology_service is not None
            and hasattr(topology_service, 'historical_deferrable_appliances')
        ):
            appliances = list(topology_service.historical_deferrable_appliances(building.name))
            if appliances:
                return appliances

        return list(getattr(building, 'deferrable_appliances', []) or [])

    def _storages_for_metrics(self, building) -> Sequence:
        topology_service = getattr(self.env, '_topology_service', None)
        if (
            getattr(self.env, 'topology_mode', 'static') == 'dynamic'
            and topology_service is not None
            and hasattr(topology_service, 'historical_storages')
        ):
            storages = list(topology_service.historical_storages(building.name))
            if storages:
                return storages

        storage = getattr(building, 'electrical_storage', None)
        return [] if storage is None else [storage]

    def _compute_bess_metrics(self, building, *, t_start: int = 0, t_final: Optional[int] = None) -> Dict[str, float]:
        upper = int(max(building.time_step, 0))
        t_start = int(max(t_start, 0))
        t_final = upper if t_final is None else int(min(max(t_final, 0), upper))
        if t_final < t_start:
            capacity = self._to_scalar(getattr(getattr(building, 'electrical_storage', None), 'capacity', 0.0), 0.0)
            return {
                'bess_charge_total_kwh': 0.0,
                'bess_discharge_total_kwh': 0.0,
                'bess_throughput_total_kwh': 0.0,
                'bess_equivalent_full_cycles': None,
                'bess_capacity_fade_ratio': None,
                '_bess_capacity_kwh': capacity,
                '_bess_degraded_capacity_kwh': capacity,
            }

        storages = [
            storage for storage in self._storages_for_metrics(building)
            if self._to_scalar(getattr(storage, 'capacity', 0.0), 0.0) > 0.0
        ]
        storage_series = []
        for storage in storages:
            values = getattr(storage, 'electricity_consumption', None)
            if values is None and len(storages) == 1:
                # Compatibility for legacy/custom building facades that expose
                # only the aggregate storage series on the building.
                values = getattr(
                    building,
                    'electrical_storage_electricity_consumption',
                    [],
                )
            storage_series.append(
                np.array(values[t_start:t_final + 1], dtype='float64')
            )
        charge_total = sum(
            self._sum_finite(np.clip(values, 0.0, None))
            for values in storage_series
        )
        discharge_total = sum(
            self._sum_finite(np.clip(-values, 0.0, None))
            for values in storage_series
        )
        throughput_total = charge_total + discharge_total

        capacity = sum(
            self._to_scalar(getattr(storage, 'capacity', 0.0), 0.0)
            for storage in storages
        )
        degraded_capacity = sum(
            self._to_scalar(
                getattr(storage, 'degraded_capacity', getattr(storage, 'capacity', 0.0)),
                self._to_scalar(getattr(storage, 'capacity', 0.0), 0.0),
            )
            for storage in storages
        )
        equivalent_cycles = None if capacity <= 0.0 else throughput_total / (2.0 * capacity)
        fade_ratio = None if capacity <= 0.0 else (capacity - degraded_capacity) / capacity

        if fade_ratio is not None:
            fade_ratio = float(np.clip(fade_ratio, 0.0, 1.0))

        return {
            'bess_charge_total_kwh': charge_total,
            'bess_discharge_total_kwh': discharge_total,
            'bess_throughput_total_kwh': throughput_total,
            'bess_equivalent_full_cycles': equivalent_cycles,
            'bess_capacity_fade_ratio': fade_ratio,
            '_bess_capacity_kwh': capacity,
            '_bess_degraded_capacity_kwh': degraded_capacity,
        }

    def _compute_pv_metrics(self, building, *, t_start: int = 0, t_final: Optional[int] = None) -> Dict[str, float]:
        upper = int(max(building.time_step, 0))
        t_start = int(max(t_start, 0))
        t_final = upper if t_final is None else int(min(max(t_final, 0), upper))
        if t_final < t_start:
            return {
                'pv_generation_total_kwh': 0.0,
                'pv_export_total_kwh': 0.0,
                'pv_self_consumption_ratio': None,
            }

        solar = np.array(building.solar_generation[t_start:t_final + 1], dtype='float64')
        net = np.array(building.net_electricity_consumption[t_start:t_final + 1], dtype='float64')

        generation = np.clip(-solar, 0.0, None)
        export = np.clip(-net, 0.0, None)
        pv_generation_total = self._sum_finite(generation)
        pv_export_total = self._sum_finite(np.minimum(generation, export))
        self_consumption_ratio = None if pv_generation_total <= 0.0 else (pv_generation_total - pv_export_total) / pv_generation_total

        return {
            'pv_generation_total_kwh': pv_generation_total,
            'pv_export_total_kwh': pv_export_total,
            'pv_self_consumption_ratio': self_consumption_ratio,
        }

    def _compute_district_pv_metrics(
        self,
        buildings: Sequence,
        building_windows: Mapping[str, Tuple[int, int]],
        *,
        final_t: int,
    ) -> Dict[str, float]:
        series_by_building = []
        for building in buildings:
            t_start, t_end = building_windows.get(building.name, (0, -1))
            if t_end < t_start:
                continue

            series_by_building.append(
                (
                    building.name,
                    int(t_start),
                    int(t_end),
                    np.array(building.solar_generation, dtype='float64'),
                    np.array(building.net_electricity_consumption, dtype='float64'),
                )
            )

        pv_generation_total = 0.0
        pv_export_total = 0.0

        for t in range(int(max(final_t, 0)) + 1):
            generation_t = 0.0
            district_net_t = 0.0

            for _, t_start, t_end, solar, net in series_by_building:
                if not (t_start <= t <= t_end):
                    continue

                solar_value = self._to_scalar(solar[t] if t < len(solar) else np.nan, 0.0)
                net_value = self._to_scalar(net[t] if t < len(net) else np.nan, 0.0)
                generation_t += max(-solar_value, 0.0)
                district_net_t += net_value

            pv_generation_total += generation_t
            pv_export_total += min(generation_t, max(-district_net_t, 0.0))

        self_consumption_ratio = None if pv_generation_total <= 0.0 else (pv_generation_total - pv_export_total) / pv_generation_total

        return {
            'pv_generation_total_kwh': float(pv_generation_total),
            'pv_export_total_kwh': float(pv_export_total),
            'pv_self_consumption_ratio': self_consumption_ratio,
        }

    def _compute_phase_metrics(self, building, *, t_start: int = 0, t_final: Optional[int] = None) -> Dict[str, object]:
        if not getattr(building, '_electrical_service_enabled', False):
            return {
                'electrical_service_violation_total_kwh': 0.0,
                'electrical_service_violation_time_step_count': 0.0,
                'electrical_service_requested_pressure_total_kwh': 0.0,
                'electrical_service_requested_pressure_time_step_count': 0.0,
                'phase_imbalance_ratio_average': None,
                'phase_import_peak_kw': {},
                'phase_export_peak_kw': {},
                '_imbalance_sum': 0.0,
                '_imbalance_count': 0.0,
            }

        upper = int(max(building.time_step, 0))
        t_start = int(max(t_start, 0))
        t_final = upper if t_final is None else int(min(max(t_final, 0), upper))
        if t_final < t_start:
            return {
                'electrical_service_violation_total_kwh': 0.0,
                'electrical_service_violation_time_step_count': 0.0,
                'electrical_service_requested_pressure_total_kwh': 0.0,
                'electrical_service_requested_pressure_time_step_count': 0.0,
                'phase_imbalance_ratio_average': None,
                'phase_import_peak_kw': {},
                'phase_export_peak_kw': {},
                '_imbalance_sum': 0.0,
                '_imbalance_count': 0.0,
            }

        # The constraint penalty records requested-action pressure before
        # projection. It is useful controller feedback, but it is not evidence
        # that post-projection power still violates the declared service limits.
        pressure_history = np.array(
            getattr(building, '_charging_constraint_violation_history', [0.0]),
            dtype='float64',
        )[t_start:t_final + 1]
        requested_pressure_total = float(np.clip(pressure_history, 0.0, None).sum())
        requested_pressure_count = float(np.count_nonzero(pressure_history > 1e-9))

        all_total_history = np.array(
            getattr(building, '_charging_total_power_history_kw', [0.0]),
            dtype='float64',
        )
        total_history = all_total_history[t_start:t_final + 1]
        phase_history = getattr(building, '_charging_phase_power_history_kw', {}) or {}
        total_limits = getattr(building, '_electrical_service_limits', {}).get('total', {}) or {}
        per_phase_limits = getattr(building, '_electrical_service_limits', {}).get('per_phase', {}) or {}
        step_hours = max(
            self._to_scalar(getattr(building, 'seconds_per_time_step', 3600.0), 3600.0) / 3600.0,
            0.0,
        )
        residual_violation_kw = np.zeros(len(total_history), dtype='float64')

        def accumulate_limit(values: np.ndarray, limits: Mapping[str, float]) -> None:
            import_limit = self._to_scalar(limits.get('import_kw'), np.nan)
            export_limit = self._to_scalar(limits.get('export_kw'), np.nan)
            if np.isfinite(import_limit):
                excess = np.clip(values - import_limit, 0.0, None)
                excess[excess <= self.ELECTRICAL_RESIDUAL_EPS_KW] = 0.0
                residual_violation_kw[:] += excess
            if np.isfinite(export_limit):
                excess = np.clip(-values - export_limit, 0.0, None)
                excess[excess <= self.ELECTRICAL_RESIDUAL_EPS_KW] = 0.0
                residual_violation_kw[:] += excess

        accumulate_limit(total_history, total_limits)
        for phase_name in self._electrical_service_phase_names(building):
            full_values = np.array(
                phase_history.get(phase_name, np.zeros(len(all_total_history))),
                dtype='float64',
            )
            values = full_values[t_start:t_final + 1]
            if len(values) != len(residual_violation_kw):
                aligned = np.zeros(len(residual_violation_kw), dtype='float64')
                count = min(len(values), len(aligned))
                aligned[:count] = values[:count]
                values = aligned
            accumulate_limit(values, per_phase_limits.get(phase_name, {}) or {})

        violation_history = residual_violation_kw * step_hours
        violation_total = float(np.clip(violation_history, 0.0, None).sum())
        violation_count = float(np.count_nonzero(violation_history > 1e-9))

        phase_history = getattr(building, '_charging_phase_power_history_kw', {}) or {}
        phase_import_peak = {}
        phase_export_peak = {}

        for phase_name, values in phase_history.items():
            series = np.array(values[t_start:t_final + 1], dtype='float64')
            phase_import_peak[phase_name] = float(np.clip(series, 0.0, None).max(initial=0.0))
            phase_export_peak[phase_name] = float(np.clip(-series, 0.0, None).max(initial=0.0))

        imbalance_sum = 0.0
        imbalance_count = 0.0
        imbalance_average = None

        if getattr(building, '_electrical_service_mode', 'single_phase') == 'three_phase':
            names = [n for n in ['L1', 'L2', 'L3'] if n in phase_history]
            if len(names) == 3:
                stacked = np.stack([np.array(phase_history[n][t_start:t_final + 1], dtype='float64') for n in names], axis=1)
                for row in stacked:
                    abs_row = np.abs(row)
                    mean_abs = float(abs_row.mean())
                    if mean_abs <= 1e-9:
                        ratio = 0.0
                    else:
                        ratio = float((abs_row.max() - abs_row.min()) / mean_abs)
                    imbalance_sum += ratio
                    imbalance_count += 1.0

                if imbalance_count > 0:
                    imbalance_average = imbalance_sum / imbalance_count

        return {
            'electrical_service_violation_total_kwh': violation_total,
            'electrical_service_violation_time_step_count': violation_count,
            'electrical_service_requested_pressure_total_kwh': requested_pressure_total,
            'electrical_service_requested_pressure_time_step_count': requested_pressure_count,
            'phase_imbalance_ratio_average': imbalance_average,
            'phase_import_peak_kw': phase_import_peak,
            'phase_export_peak_kw': phase_export_peak,
            '_imbalance_sum': imbalance_sum,
            '_imbalance_count': imbalance_count,
        }

    def _collect_market_totals(self, building_names: List[str]) -> Tuple[Mapping[str, Mapping[str, float]], Mapping[str, float]]:
        history = getattr(self.env, '_community_market_settlement_history', []) or []
        by_building = {
            name: {
                'community_local_import_total_kwh': 0.0,
                'community_local_export_total_kwh': 0.0,
                'community_grid_import_after_local_total_kwh': 0.0,
                'community_grid_export_after_local_total_kwh': 0.0,
                'community_settled_cost_total_eur': 0.0,
                'community_counterfactual_cost_total_eur': 0.0,
                'community_market_savings_total_eur': 0.0,
            }
            for name in building_names
        }

        for rows in history:
            for row in rows:
                name = row.get('building')
                if name not in by_building:
                    continue

                target = by_building[name]
                target['community_local_import_total_kwh'] += self._to_scalar(row.get('local_import_kwh'), 0.0)
                target['community_local_export_total_kwh'] += self._to_scalar(row.get('local_export_kwh'), 0.0)
                target['community_grid_import_after_local_total_kwh'] += self._to_scalar(row.get('grid_import_kwh'), 0.0)
                target['community_grid_export_after_local_total_kwh'] += self._to_scalar(row.get('grid_export_kwh'), 0.0)
                target['community_settled_cost_total_eur'] += self._to_scalar(row.get('settled_cost_eur', row.get('settled_cost')), 0.0)
                target['community_counterfactual_cost_total_eur'] += self._to_scalar(row.get('counterfactual_cost_eur'), 0.0)
                target['community_market_savings_total_eur'] += self._to_scalar(row.get('market_savings_eur'), 0.0)

        district = {
            key: float(sum(values[key] for values in by_building.values()))
            for key in [
                'community_local_import_total_kwh',
                'community_local_export_total_kwh',
                'community_grid_import_after_local_total_kwh',
                'community_grid_export_after_local_total_kwh',
                'community_settled_cost_total_eur',
                'community_counterfactual_cost_total_eur',
                'community_market_savings_total_eur',
            ]
        }

        return by_building, district

    @staticmethod
    def _resolve_step_value(value, time_step: int, default: float = 0.0) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return float(default)
            index = min(max(time_step, 0), len(value) - 1)
            return CityLearnKPIService._to_scalar(value[index], default)

        return CityLearnKPIService._to_scalar(value, default)

    @staticmethod
    def _allocate_weighted_share_import(imports: np.ndarray, traded_kwh: float, weights: np.ndarray) -> np.ndarray:
        allocations = np.zeros_like(imports, dtype='float64')
        remaining = max(float(traded_kwh), 0.0)
        eps = 1e-9
        weights = np.array(weights, dtype='float64')
        weights = np.clip(weights, 0.0, None)

        while remaining > eps:
            needs = imports - allocations
            active = needs > eps
            if not np.any(active):
                break

            active_weights = weights[active]
            sum_weights = float(active_weights.sum())

            if sum_weights <= eps:
                # Fallback to equal-share for active importers when weights are invalid.
                share = remaining / float(np.count_nonzero(active))
                granted = np.minimum(share, needs[active])
            else:
                granted = np.minimum(remaining * (active_weights / sum_weights), needs[active])

            granted_total = float(granted.sum())
            if granted_total <= eps:
                break

            allocations[active] += granted
            remaining -= granted_total

        return allocations

    def _default_building_conditions(self, building, control_condition, baseline_condition, *, evaluation_condition_cls, dynamics_building_cls):
        if isinstance(building, dynamics_building_cls):
            building_control_condition = (
                evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
                if control_condition is None else control_condition
            )
            building_baseline_condition = (
                evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
                if baseline_condition is None else baseline_condition
            )
        else:
            building_control_condition = (
                evaluation_condition_cls.WITH_STORAGE_AND_PV
                if control_condition is None else control_condition
            )
            building_baseline_condition = (
                evaluation_condition_cls.WITHOUT_STORAGE_BUT_WITH_PV
                if baseline_condition is None else baseline_condition
            )

        return building_control_condition, building_baseline_condition

    def _get_kpi_buildings(self) -> List:
        env = self.env

        if getattr(env, 'topology_mode', 'static') != 'dynamic':
            return list(env.buildings)

        topology_service = getattr(env, '_topology_service', None)
        if topology_service is None:
            return list(env.buildings)

        lifecycle = getattr(env, 'topology_member_lifecycle', {}) or {}
        pool = dict(getattr(topology_service, 'member_pool', {}) or {})
        if len(pool) == 0:
            return list(env.buildings)

        active_or_historical = []
        current_active_names = {b.name for b in env.buildings}
        for member_id, building in pool.items():
            state = lifecycle.get(member_id, {})
            if state.get('born_at') is None and member_id not in current_active_names:
                continue
            active_or_historical.append(building)

        return active_or_historical

    def _building_active_window(self, building) -> Tuple[int, int]:
        env = self.env
        final_t = int(max(getattr(env, 'time_step', 0), 0))

        if getattr(env, 'topology_mode', 'static') != 'dynamic':
            return 0, int(min(max(getattr(building, 'time_step', 0), 0), final_t))

        lifecycle = getattr(env, 'topology_member_lifecycle', {}) or {}
        state = lifecycle.get(getattr(building, 'name', ''), {}) or {}

        born_at = state.get('born_at', 0)
        removed_at = state.get('removed_at', None)
        if born_at is None:
            return 1, 0

        start = int(max(born_at, 0))
        if removed_at is None:
            end = final_t
        else:
            end = min(final_t, int(removed_at) - 1)

        return start, end

    @staticmethod
    def _slice_window(values, t_start: int, t_final: int) -> np.ndarray:
        array = np.array(values, dtype='float64')
        if array.size == 0:
            return np.zeros((0,), dtype='float64')

        lo = max(int(t_start), 0)
        hi = min(int(t_final), array.size - 1)
        if hi < lo:
            return np.zeros((0,), dtype='float64')
        return array[lo:hi + 1]

    @staticmethod
    def _cost_last(values: np.ndarray, fn, **kwargs) -> float:
        if values.size == 0:
            return 0.0
        out = fn(values, **kwargs) if kwargs else fn(values)
        if len(out) == 0:
            return 0.0
        return float(out[-1])

    def _condition_settled_cost_totals(
        self,
        *,
        condition_by_building: Mapping[str, object],
    ) -> Tuple[Mapping[str, float], float]:
        env = self.env
        buildings = self._get_kpi_buildings()
        building_names = [b.name for b in buildings]
        windows = {b.name: self._building_active_window(b) for b in buildings}
        totals = {name: 0.0 for name in building_names}
        if len(building_names) == 0:
            return totals, 0.0

        final_t = int(max(getattr(env, 'time_step', 0), 0))
        ratio = self._to_scalar(getattr(env, 'community_market_sell_ratio', 0.8), 0.8)
        ratio = min(max(ratio, 0.0), 1.0)
        weights_config = getattr(env, 'community_market_import_member_weights', {}) or {}
        weights = np.array(
            [
                self._to_scalar(weights_config.get(name, 1.0), 1.0)
                for name in building_names
            ],
            dtype='float64',
        )
        weight_by_name = {name: float(weight) for name, weight in zip(building_names, weights)}
        net_series_by_name: Dict[str, np.ndarray] = {}
        price_series_by_name: Dict[str, np.ndarray] = {}

        for building in buildings:
            condition = condition_by_building.get(building.name)
            if condition is None:
                continue

            net_series_by_name[building.name] = np.asarray(
                getattr(building, f'net_electricity_consumption{condition.value}'),
                dtype='float64',
            )
            price_series_by_name[building.name] = np.asarray(
                building.pricing.electricity_pricing,
                dtype='float64',
            )

        for t in range(final_t + 1):
            net_values = []
            active_buildings_t = []
            active_weights_t = []
            for building in buildings:
                t_start, t_end = windows.get(building.name, (0, -1))
                if not (t_start <= t <= t_end):
                    continue

                net_series = net_series_by_name.get(building.name)
                if net_series is None:
                    continue

                net_values.append(self._to_scalar(net_series[t] if t < len(net_series) else np.nan, 0.0))
                active_buildings_t.append(building)
                active_weights_t.append(weight_by_name.get(building.name, 1.0))

            if len(active_buildings_t) == 0:
                continue

            net_values = np.array(net_values, dtype='float64')
            imports = np.clip(net_values, 0.0, None)
            exports = np.clip(-net_values, 0.0, None)
            total_import = float(imports.sum())
            total_export = float(exports.sum())
            traded_kwh = min(total_import, total_export)

            if total_import > 0.0 and traded_kwh > 0.0:
                local_import = self._allocate_weighted_share_import(
                    imports,
                    traded_kwh,
                    np.array(active_weights_t, dtype='float64'),
                )
            else:
                local_import = np.zeros_like(imports, dtype='float64')

            if total_export > 0.0 and traded_kwh > 0.0:
                local_export = exports * (traded_kwh / total_export)
            else:
                local_export = np.zeros_like(exports, dtype='float64')

            for idx, building in enumerate(active_buildings_t):
                price_series = price_series_by_name.get(building.name)
                grid_import_price = self._to_scalar(
                    price_series[t] if price_series is not None and t < len(price_series) else np.nan,
                    0.0,
                )
                local_price = ratio * grid_import_price
                grid_import_remaining = max(imports[idx] - local_import[idx], 0.0)

                cost = (
                    grid_import_remaining * grid_import_price
                    + local_import[idx] * local_price
                    - local_export[idx] * local_price
                )
                totals[building.name] += float(cost)

        district_total = float(sum(totals.values()))
        return totals, district_total

    def evaluate(
        self,
        control_condition=None,
        baseline_condition=None,
        comfort_band: float = None,
        *,
        evaluation_condition_cls,
        dynamics_building_cls,
    ) -> pd.DataFrame:
        """Evaluate cost functions at current time step."""

        env = self.env

        get_net_electricity_consumption = lambda x, c: getattr(x, f'net_electricity_consumption{c.value}')
        get_net_electricity_consumption_cost = lambda x, c: getattr(x, f'net_electricity_consumption_cost{c.value}')
        get_net_electricity_consumption_emission = lambda x, c: getattr(x, f'net_electricity_consumption_emission{c.value}')

        comfort_band = EnergySimulation.DEFUALT_COMFORT_BAND if comfort_band is None else comfort_band
        daily_steps = self._window_steps(24.0 * 3600.0, env.seconds_per_time_step)
        monthly_steps = self._window_steps(730.0 * 3600.0, env.seconds_per_time_step)
        simulated_days = self._simulated_days(env)

        legacy_building_frames: List[pd.DataFrame] = []
        extended_building_rows: List[Dict[str, object]] = []

        ev_departures_total = 0.0
        ev_departures_met = 0.0
        ev_departures_min_acceptable = 0.0
        ev_departures_within_tolerance = 0.0
        ev_departures_target_feasible = 0.0
        ev_departures_target_infeasible = 0.0
        ev_departures_min_acceptable_feasible = 0.0
        ev_departures_min_acceptable_infeasible = 0.0
        ev_departures_within_tolerance_feasible = 0.0
        ev_departures_within_tolerance_infeasible = 0.0
        ev_departures_met_target_feasible = 0.0
        ev_departures_min_acceptable_feasible_met = 0.0
        ev_departures_within_tolerance_feasible_met = 0.0
        ev_deficit_sum = 0.0
        ev_shortfall_beyond_tolerance_sum = 0.0
        ev_surplus_sum = 0.0
        ev_absolute_error_sum = 0.0
        ev_charge_total = 0.0
        ev_v2g_total = 0.0
        ev_connected_soc_gain_total = 0.0
        ev_energy_accounting_shortfall_total = 0.0

        bess_charge_total = 0.0
        bess_discharge_total = 0.0
        bess_throughput_total = 0.0
        bess_capacity_total = 0.0
        bess_capacity_loss_total = 0.0

        phase_violation_total = 0.0
        phase_violation_count = 0.0
        phase_requested_pressure_total = 0.0
        phase_requested_pressure_count = 0.0
        phase_imbalance_sum = 0.0
        phase_imbalance_count = 0.0

        kpi_buildings = self._get_kpi_buildings()
        building_windows = {building.name: self._building_active_window(building) for building in kpi_buildings}
        building_names = [building.name for building in kpi_buildings]
        equity_group_by_building = {building.name: getattr(building, 'equity_group', None) for building in kpi_buildings}
        equity_relative_benefit_by_building: Dict[str, Optional[float]] = {}
        equity_valid_benefits: Dict[str, float] = {}

        for building in kpi_buildings:
            t_start, t_end = building_windows.get(building.name, (0, -1))
            if t_end < t_start:
                continue
            building_days = self._window_days(t_start, t_end, env.seconds_per_time_step)

            if isinstance(building, dynamics_building_cls):
                building_control_condition = (
                    evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
                    if control_condition is None else control_condition
                )
                building_baseline_condition = (
                    evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
                    if baseline_condition is None else baseline_condition
                )
            else:
                building_control_condition = (
                    evaluation_condition_cls.WITH_STORAGE_AND_PV
                    if control_condition is None else control_condition
                )
                building_baseline_condition = (
                    evaluation_condition_cls.WITHOUT_STORAGE_BUT_WITH_PV
                    if baseline_condition is None else baseline_condition
                )

            indoor = self._slice_window(building.indoor_dry_bulb_temperature, t_start, t_end)
            cooling_setpoint = self._slice_window(building.indoor_dry_bulb_temperature_cooling_set_point, t_start, t_end)
            heating_setpoint = self._slice_window(building.indoor_dry_bulb_temperature_heating_set_point, t_start, t_end)
            occupants = self._slice_window(building.occupant_count, t_start, t_end)
            discomfort_kwargs = {
                'indoor_dry_bulb_temperature': indoor,
                'dry_bulb_temperature_cooling_set_point': cooling_setpoint,
                'dry_bulb_temperature_heating_set_point': heating_setpoint,
                'band': building.comfort_band if comfort_band is None else comfort_band,
                'occupant_count': occupants,
            }
            unmet, cold, hot, \
                cold_minimum_delta, cold_maximum_delta, cold_average_delta, \
                hot_minimum_delta, hot_maximum_delta, hot_average_delta = CostFunction.discomfort(**discomfort_kwargs)

            expected_energy = (
                self._slice_window(building.cooling_demand, t_start, t_end)
                + self._slice_window(building.heating_demand, t_start, t_end)
                + self._slice_window(building.dhw_demand, t_start, t_end)
                + self._slice_window(building.non_shiftable_load, t_start, t_end)
            )
            served_energy = (
                self._slice_window(building.energy_from_cooling_device, t_start, t_end)
                + self._slice_window(building.energy_from_cooling_storage, t_start, t_end)
                + self._slice_window(building.energy_from_heating_device, t_start, t_end)
                + self._slice_window(building.energy_from_heating_storage, t_start, t_end)
                + self._slice_window(building.energy_from_dhw_device, t_start, t_end)
                + self._slice_window(building.energy_from_dhw_storage, t_start, t_end)
                + self._slice_window(building.energy_to_non_shiftable_load, t_start, t_end)
            )

            net_c_series = self._slice_window(
                getattr(building, f'net_electricity_consumption{building_control_condition.value}'),
                t_start,
                t_end,
            )
            net_b_series = self._slice_window(
                getattr(building, f'net_electricity_consumption{building_baseline_condition.value}'),
                t_start,
                t_end,
            )
            ec_c = self._positive_total(net_c_series)
            ec_b = self._positive_total(net_b_series)
            export_c = self._sum_finite(np.clip(-net_c_series, 0.0, None))
            export_b = self._sum_finite(np.clip(-net_b_series, 0.0, None))
            zne_c = self._net_total(net_c_series)
            zne_b = self._net_total(net_b_series)
            ce_c_series = self._slice_window(
                getattr(building, f'net_electricity_consumption_emission{building_control_condition.value}'),
                t_start,
                t_end,
            )
            ce_b_series = self._slice_window(
                getattr(building, f'net_electricity_consumption_emission{building_baseline_condition.value}'),
                t_start,
                t_end,
            )
            ce_c = self._positive_total(ce_c_series)
            ce_b = (
                self._positive_total(ce_b_series)
                if np.sum(self._slice_window(building.carbon_intensity.carbon_intensity, t_start, t_end)) != 0
                else 0.0
            )
            control_cost_series = self._slice_window(
                getattr(building, f'net_electricity_consumption_cost{building_control_condition.value}'),
                t_start,
                t_end,
            )
            baseline_cost_series = self._slice_window(
                getattr(building, f'net_electricity_consumption_cost{building_baseline_condition.value}'),
                t_start,
                t_end,
            )
            cost_c_legacy = self._positive_total(control_cost_series)
            cost_b_legacy = self._positive_total(baseline_cost_series)
            cost_c_raw = self._sum_finite(control_cost_series)
            cost_b_raw = self._sum_finite(baseline_cost_series)
            equity_benefit = self._equity_relative_benefit_percent(cost_c_raw, cost_b_raw)
            equity_relative_benefit_by_building[building.name] = equity_benefit

            if equity_benefit is not None:
                equity_valid_benefits[building.name] = float(equity_benefit)

            legacy_building_frame = pd.DataFrame([{
                'cost_function': 'electricity_consumption_total',
                'value': self._safe_div(ec_c, ec_b),
            }, {
                'cost_function': 'zero_net_energy',
                'value': self._safe_div(zne_c, zne_b),
            }, {
                'cost_function': 'carbon_emissions_total',
                'value': self._safe_div(ce_c, ce_b),
            }, {
                'cost_function': 'cost_total',
                'value': self._safe_div(cost_c_legacy, cost_b_legacy),
            }, {
                'cost_function': 'discomfort_proportion',
                'value': unmet[-1],
            }, {
                'cost_function': 'discomfort_cold_proportion',
                'value': cold[-1],
            }, {
                'cost_function': 'discomfort_hot_proportion',
                'value': hot[-1],
            }, {
                'cost_function': 'discomfort_cold_delta_minimum',
                'value': cold_minimum_delta[-1],
            }, {
                'cost_function': 'discomfort_cold_delta_maximum',
                'value': cold_maximum_delta[-1],
            }, {
                'cost_function': 'discomfort_cold_delta_average',
                'value': cold_average_delta[-1],
            }, {
                'cost_function': 'discomfort_hot_delta_minimum',
                'value': hot_minimum_delta[-1],
            }, {
                'cost_function': 'discomfort_hot_delta_maximum',
                'value': hot_maximum_delta[-1],
            }, {
                'cost_function': 'discomfort_hot_delta_average',
                'value': hot_average_delta[-1],
            }, {
                'cost_function': 'one_minus_thermal_resilience_proportion',
                'value': self._cost_last(
                    np.asarray(
                        CostFunction.one_minus_thermal_resilience(
                            power_outage=self._slice_window(building.power_outage_signal, t_start, t_end),
                            **discomfort_kwargs,
                        ),
                        dtype='float64',
                    ),
                    lambda x: x,
                ),
            }, {
                'cost_function': 'power_outage_normalized_unserved_energy_total',
                'value': self._cost_last(
                    np.asarray(
                        CostFunction.normalized_unserved_energy(
                            expected_energy,
                            served_energy,
                            power_outage=self._slice_window(building.power_outage_signal, t_start, t_end),
                        ),
                        dtype='float64',
                    ),
                    lambda x: x,
                ),
            }, {
                'cost_function': 'annual_normalized_unserved_energy_total',
                'value': self._cost_last(
                    np.asarray(CostFunction.normalized_unserved_energy(expected_energy, served_energy), dtype='float64'),
                    lambda x: x,
                ),
            }])
            legacy_building_frame['name'] = building.name
            legacy_building_frames.append(legacy_building_frame)

            extended_building_rows.extend([
                self._metric('electricity_consumption_control_total_kwh', ec_c, building.name, 'building'),
                self._metric('electricity_consumption_baseline_total_kwh', ec_b, building.name, 'building'),
                self._metric('electricity_consumption_delta_total_kwh', ec_c - ec_b, building.name, 'building'),
                self._metric('electricity_consumption_control_daily_average_kwh', self._daily_average(ec_c, building_days), building.name, 'building'),
                self._metric('electricity_consumption_baseline_daily_average_kwh', self._daily_average(ec_b, building_days), building.name, 'building'),
                self._metric('electricity_consumption_delta_daily_average_kwh', self._daily_average(ec_c - ec_b, building_days), building.name, 'building'),
                self._metric('electricity_export_control_total_kwh', export_c, building.name, 'building'),
                self._metric('electricity_export_baseline_total_kwh', export_b, building.name, 'building'),
                self._metric('electricity_export_delta_total_kwh', export_c - export_b, building.name, 'building'),
                self._metric('electricity_export_control_daily_average_kwh', self._daily_average(export_c, building_days), building.name, 'building'),
                self._metric('electricity_export_baseline_daily_average_kwh', self._daily_average(export_b, building_days), building.name, 'building'),
                self._metric('electricity_export_delta_daily_average_kwh', self._daily_average(export_c - export_b, building_days), building.name, 'building'),
                self._metric('zero_net_energy_control_total_kwh', zne_c, building.name, 'building'),
                self._metric('zero_net_energy_baseline_total_kwh', zne_b, building.name, 'building'),
                self._metric('zero_net_energy_delta_total_kwh', zne_c - zne_b, building.name, 'building'),
                self._metric('zero_net_energy_control_daily_average_kwh', self._daily_average(zne_c, building_days), building.name, 'building'),
                self._metric('zero_net_energy_baseline_daily_average_kwh', self._daily_average(zne_b, building_days), building.name, 'building'),
                self._metric('zero_net_energy_delta_daily_average_kwh', self._daily_average(zne_c - zne_b, building_days), building.name, 'building'),
                self._metric('carbon_emissions_control_total_kgco2', ce_c, building.name, 'building'),
                self._metric('carbon_emissions_baseline_total_kgco2', ce_b, building.name, 'building'),
                self._metric('carbon_emissions_delta_total_kgco2', ce_c - ce_b, building.name, 'building'),
                self._metric('carbon_emissions_control_daily_average_kgco2', self._daily_average(ce_c, building_days), building.name, 'building'),
                self._metric('carbon_emissions_baseline_daily_average_kgco2', self._daily_average(ce_b, building_days), building.name, 'building'),
                self._metric('carbon_emissions_delta_daily_average_kgco2', self._daily_average(ce_c - ce_b, building_days), building.name, 'building'),
                self._metric('cost_control_total_eur', cost_c_raw, building.name, 'building'),
                self._metric('cost_baseline_total_eur', cost_b_raw, building.name, 'building'),
                self._metric('cost_delta_total_eur', cost_c_raw - cost_b_raw, building.name, 'building'),
                self._metric('cost_control_daily_average_eur', self._daily_average(cost_c_raw, building_days), building.name, 'building'),
                self._metric('cost_baseline_daily_average_eur', self._daily_average(cost_b_raw, building_days), building.name, 'building'),
                self._metric('cost_delta_daily_average_eur', self._daily_average(cost_c_raw - cost_b_raw, building_days), building.name, 'building'),
                self._metric('equity_relative_benefit_percent', equity_benefit, building.name, 'building'),
            ])

            ev_metrics = self._compute_ev_metrics(building, t_start=t_start, t_final=t_end)
            extended_building_rows.extend([
                self._metric('ev_departure_events_count', ev_metrics['departures_total'], building.name, 'building'),
                self._metric('ev_departure_met_events_count', ev_metrics['departures_met'], building.name, 'building'),
                self._metric('ev_departure_min_acceptable_events_count', ev_metrics['departures_min_acceptable'], building.name, 'building'),
                self._metric('ev_departure_within_tolerance_events_count', ev_metrics['departures_within_tolerance'], building.name, 'building'),
                self._metric('ev_departure_target_feasible_events_count', ev_metrics['departures_target_feasible'], building.name, 'building'),
                self._metric('ev_departure_target_infeasible_events_count', ev_metrics['departures_target_infeasible'], building.name, 'building'),
                self._metric('ev_departure_min_acceptable_feasible_events_count', ev_metrics['departures_min_acceptable_feasible'], building.name, 'building'),
                self._metric('ev_departure_min_acceptable_infeasible_events_count', ev_metrics['departures_min_acceptable_infeasible'], building.name, 'building'),
                self._metric('ev_departure_within_tolerance_feasible_events_count', ev_metrics['departures_within_tolerance_feasible'], building.name, 'building'),
                self._metric('ev_departure_within_tolerance_infeasible_events_count', ev_metrics['departures_within_tolerance_infeasible'], building.name, 'building'),
                self._metric('ev_departure_success_rate', ev_metrics['ev_departure_success_rate'], building.name, 'building'),
                self._metric('ev_departure_min_acceptable_rate', ev_metrics['ev_departure_min_acceptable_rate'], building.name, 'building'),
                self._metric('ev_departure_within_tolerance_rate', ev_metrics['ev_departure_within_tolerance_rate'], building.name, 'building'),
                self._metric('ev_departure_success_feasible_rate', ev_metrics['ev_departure_success_feasible_rate'], building.name, 'building'),
                self._metric('ev_departure_min_acceptable_feasible_rate', ev_metrics['ev_departure_min_acceptable_feasible_rate'], building.name, 'building'),
                self._metric('ev_departure_within_tolerance_feasible_rate', ev_metrics['ev_departure_within_tolerance_feasible_rate'], building.name, 'building'),
                self._metric('ev_departure_soc_deficit_mean', ev_metrics['ev_departure_soc_deficit_mean'], building.name, 'building'),
                self._metric('ev_departure_shortfall_beyond_tolerance_mean', ev_metrics['ev_departure_shortfall_beyond_tolerance_mean'], building.name, 'building'),
                self._metric('ev_departure_soc_surplus_mean', ev_metrics['ev_departure_soc_surplus_mean'], building.name, 'building'),
                self._metric('ev_departure_soc_absolute_error_mean', ev_metrics['ev_departure_soc_absolute_error_mean'], building.name, 'building'),
                self._metric('ev_departure_tolerance_ratio', ev_metrics['ev_departure_tolerance_ratio'], building.name, 'building'),
                self._metric('ev_charge_total_kwh', ev_metrics['ev_charge_total_kwh'], building.name, 'building'),
                self._metric('ev_v2g_export_total_kwh', ev_metrics['ev_v2g_export_total_kwh'], building.name, 'building'),
                self._metric('ev_connected_soc_gain_total_kwh', ev_metrics['ev_connected_soc_gain_total_kwh'], building.name, 'building'),
                self._metric('ev_energy_accounting_shortfall_kwh', ev_metrics['ev_energy_accounting_shortfall_kwh'], building.name, 'building'),
            ])
            ev_departures_total += ev_metrics['departures_total']
            ev_departures_met += ev_metrics['departures_met']
            ev_departures_min_acceptable += ev_metrics['departures_min_acceptable']
            ev_departures_within_tolerance += ev_metrics['departures_within_tolerance']
            ev_departures_target_feasible += ev_metrics['departures_target_feasible']
            ev_departures_target_infeasible += ev_metrics['departures_target_infeasible']
            ev_departures_min_acceptable_feasible += ev_metrics['departures_min_acceptable_feasible']
            ev_departures_min_acceptable_infeasible += ev_metrics['departures_min_acceptable_infeasible']
            ev_departures_within_tolerance_feasible += ev_metrics['departures_within_tolerance_feasible']
            ev_departures_within_tolerance_infeasible += ev_metrics['departures_within_tolerance_infeasible']
            ev_departures_met_target_feasible += ev_metrics['_departures_met_target_feasible']
            ev_departures_min_acceptable_feasible_met += ev_metrics['_departures_min_acceptable_feasible_met']
            ev_departures_within_tolerance_feasible_met += ev_metrics['_departures_within_tolerance_feasible_met']
            ev_deficit_sum += ev_metrics['departure_deficit_sum']
            ev_shortfall_beyond_tolerance_sum += ev_metrics['departure_shortfall_beyond_tolerance_sum']
            ev_surplus_sum += ev_metrics['departure_surplus_sum']
            ev_absolute_error_sum += ev_metrics['departure_absolute_error_sum']
            ev_charge_total += ev_metrics['ev_charge_total_kwh']
            ev_v2g_total += ev_metrics['ev_v2g_export_total_kwh']
            ev_connected_soc_gain_total += ev_metrics['ev_connected_soc_gain_total_kwh']
            ev_energy_accounting_shortfall_total += ev_metrics['ev_energy_accounting_shortfall_kwh']

            bess_metrics = self._compute_bess_metrics(building, t_start=t_start, t_final=t_end)
            extended_building_rows.extend([
                self._metric('bess_charge_total_kwh', bess_metrics['bess_charge_total_kwh'], building.name, 'building'),
                self._metric('bess_discharge_total_kwh', bess_metrics['bess_discharge_total_kwh'], building.name, 'building'),
                self._metric('bess_throughput_total_kwh', bess_metrics['bess_throughput_total_kwh'], building.name, 'building'),
                self._metric('bess_equivalent_full_cycles', bess_metrics['bess_equivalent_full_cycles'], building.name, 'building'),
                self._metric('bess_capacity_fade_ratio', bess_metrics['bess_capacity_fade_ratio'], building.name, 'building'),
            ])
            bess_charge_total += bess_metrics['bess_charge_total_kwh']
            bess_discharge_total += bess_metrics['bess_discharge_total_kwh']
            bess_throughput_total += bess_metrics['bess_throughput_total_kwh']
            bess_capacity_total += bess_metrics['_bess_capacity_kwh']
            bess_capacity_loss_total += max(bess_metrics['_bess_capacity_kwh'] - bess_metrics['_bess_degraded_capacity_kwh'], 0.0)

            pv_metrics = self._compute_pv_metrics(building, t_start=t_start, t_final=t_end)
            extended_building_rows.extend([
                self._metric('pv_generation_total_kwh', pv_metrics['pv_generation_total_kwh'], building.name, 'building'),
                self._metric('pv_export_total_kwh', pv_metrics['pv_export_total_kwh'], building.name, 'building'),
                self._metric('pv_generation_daily_average_kwh', self._daily_average(pv_metrics['pv_generation_total_kwh'], building_days), building.name, 'building'),
                self._metric('pv_export_daily_average_kwh', self._daily_average(pv_metrics['pv_export_total_kwh'], building_days), building.name, 'building'),
                self._metric('pv_self_consumption_ratio', pv_metrics['pv_self_consumption_ratio'], building.name, 'building'),
            ])
            phase_metrics = self._compute_phase_metrics(building, t_start=t_start, t_final=t_end)
            extended_building_rows.extend([
                self._metric('electrical_service_violation_total_kwh', phase_metrics['electrical_service_violation_total_kwh'], building.name, 'building'),
                self._metric('electrical_service_violation_time_step_count', phase_metrics['electrical_service_violation_time_step_count'], building.name, 'building'),
                self._metric('electrical_service_requested_pressure_total_kwh', phase_metrics['electrical_service_requested_pressure_total_kwh'], building.name, 'building'),
                self._metric('electrical_service_requested_pressure_time_step_count', phase_metrics['electrical_service_requested_pressure_time_step_count'], building.name, 'building'),
                self._metric('phase_imbalance_ratio_average', phase_metrics['phase_imbalance_ratio_average'], building.name, 'building'),
            ])
            for phase_name, value in phase_metrics['phase_import_peak_kw'].items():
                extended_building_rows.append(self._metric(f'phase_import_peak_kw_{phase_name}', value, building.name, 'building'))
            for phase_name, value in phase_metrics['phase_export_peak_kw'].items():
                extended_building_rows.append(self._metric(f'phase_export_peak_kw_{phase_name}', value, building.name, 'building'))

            phase_violation_total += phase_metrics['electrical_service_violation_total_kwh']
            phase_violation_count += phase_metrics['electrical_service_violation_time_step_count']
            phase_requested_pressure_total += phase_metrics['electrical_service_requested_pressure_total_kwh']
            phase_requested_pressure_count += phase_metrics['electrical_service_requested_pressure_time_step_count']
            phase_imbalance_sum += phase_metrics['_imbalance_sum']
            phase_imbalance_count += phase_metrics['_imbalance_count']

        legacy_building = pd.concat(legacy_building_frames, ignore_index=True) if legacy_building_frames else pd.DataFrame(columns=['cost_function', 'value', 'name'])
        legacy_building['level'] = 'building'

        env_control_condition = (
            evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
            if control_condition is None else control_condition
        )
        env_baseline_condition = (
            evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
            if baseline_condition is None else baseline_condition
        )

        net_c_env_series = np.array(get_net_electricity_consumption(env, env_control_condition), dtype='float64')
        net_b_env_series = np.array(get_net_electricity_consumption(env, env_baseline_condition), dtype='float64')
        ramp_c = self._ramping_final(net_c_env_series)
        ramp_b = self._ramping_final(net_b_env_series)
        dlf_daily_c = self._one_minus_load_factor_final(net_c_env_series, window=daily_steps)
        dlf_daily_b = self._one_minus_load_factor_final(net_b_env_series, window=daily_steps)
        dlf_monthly_c = self._one_minus_load_factor_final(net_c_env_series, window=monthly_steps)
        dlf_monthly_b = self._one_minus_load_factor_final(net_b_env_series, window=monthly_steps)
        peak_daily_c = self._peak_final(net_c_env_series, window=daily_steps)
        peak_daily_b = self._peak_final(net_b_env_series, window=daily_steps)
        peak_all_c = self._peak_final(net_c_env_series, window=env.time_steps)
        peak_all_b = self._peak_final(net_b_env_series, window=env.time_steps)

        legacy_district_base = pd.DataFrame([{
            'cost_function': 'ramping_average',
            'value': self._safe_div(ramp_c, ramp_b),
        }, {
            'cost_function': 'daily_one_minus_load_factor_average',
            'value': self._safe_div(dlf_daily_c, dlf_daily_b),
        }, {
            'cost_function': 'monthly_one_minus_load_factor_average',
            'value': self._safe_div(dlf_monthly_c, dlf_monthly_b),
        }, {
            'cost_function': 'daily_peak_average',
            'value': self._safe_div(peak_daily_c, peak_daily_b),
        }, {
            'cost_function': 'all_time_peak_average',
            'value': self._safe_div(peak_all_c, peak_all_b),
        }])

        legacy_district = pd.concat([legacy_district_base, legacy_building], ignore_index=True, sort=False)
        legacy_district = legacy_district.groupby(['cost_function'])[['value']].mean().reset_index()
        legacy_district['name'] = 'District'
        legacy_district['level'] = 'district'

        legacy_cost_functions = pd.concat([legacy_district, legacy_building], ignore_index=True, sort=False)

        # Extended KPI district-level
        ec_c_env = self._positive_total(net_c_env_series)
        ec_b_env = self._positive_total(net_b_env_series)
        export_c_env = self._sum_finite(np.clip(-net_c_env_series, 0.0, None))
        export_b_env = self._sum_finite(np.clip(-net_b_env_series, 0.0, None))
        zne_c_env = self._net_total(net_c_env_series)
        zne_b_env = self._net_total(net_b_env_series)
        ce_c_env = self._positive_total(get_net_electricity_consumption_emission(env, env_control_condition))
        ce_b_env = self._positive_total(get_net_electricity_consumption_emission(env, env_baseline_condition))
        env_control_cost_series = get_net_electricity_consumption_cost(env, env_control_condition)
        env_baseline_cost_series = get_net_electricity_consumption_cost(env, env_baseline_condition)
        cost_c_env_raw = self._sum_finite(env_control_cost_series)
        cost_b_env_raw = self._sum_finite(env_baseline_cost_series)

        extended_district_rows = [
            self._metric('electricity_consumption_control_total_kwh', ec_c_env, 'District', 'district'),
            self._metric('electricity_consumption_baseline_total_kwh', ec_b_env, 'District', 'district'),
            self._metric('electricity_consumption_delta_total_kwh', ec_c_env - ec_b_env, 'District', 'district'),
            self._metric('electricity_consumption_control_daily_average_kwh', self._daily_average(ec_c_env, simulated_days), 'District', 'district'),
            self._metric('electricity_consumption_baseline_daily_average_kwh', self._daily_average(ec_b_env, simulated_days), 'District', 'district'),
            self._metric('electricity_consumption_delta_daily_average_kwh', self._daily_average(ec_c_env - ec_b_env, simulated_days), 'District', 'district'),
            self._metric('electricity_export_control_total_kwh', export_c_env, 'District', 'district'),
            self._metric('electricity_export_baseline_total_kwh', export_b_env, 'District', 'district'),
            self._metric('electricity_export_delta_total_kwh', export_c_env - export_b_env, 'District', 'district'),
            self._metric('electricity_export_control_daily_average_kwh', self._daily_average(export_c_env, simulated_days), 'District', 'district'),
            self._metric('electricity_export_baseline_daily_average_kwh', self._daily_average(export_b_env, simulated_days), 'District', 'district'),
            self._metric('electricity_export_delta_daily_average_kwh', self._daily_average(export_c_env - export_b_env, simulated_days), 'District', 'district'),
            self._metric('zero_net_energy_control_total_kwh', zne_c_env, 'District', 'district'),
            self._metric('zero_net_energy_baseline_total_kwh', zne_b_env, 'District', 'district'),
            self._metric('zero_net_energy_delta_total_kwh', zne_c_env - zne_b_env, 'District', 'district'),
            self._metric('zero_net_energy_control_daily_average_kwh', self._daily_average(zne_c_env, simulated_days), 'District', 'district'),
            self._metric('zero_net_energy_baseline_daily_average_kwh', self._daily_average(zne_b_env, simulated_days), 'District', 'district'),
            self._metric('zero_net_energy_delta_daily_average_kwh', self._daily_average(zne_c_env - zne_b_env, simulated_days), 'District', 'district'),
            self._metric('carbon_emissions_control_total_kgco2', ce_c_env, 'District', 'district'),
            self._metric('carbon_emissions_baseline_total_kgco2', ce_b_env, 'District', 'district'),
            self._metric('carbon_emissions_delta_total_kgco2', ce_c_env - ce_b_env, 'District', 'district'),
            self._metric('carbon_emissions_control_daily_average_kgco2', self._daily_average(ce_c_env, simulated_days), 'District', 'district'),
            self._metric('carbon_emissions_baseline_daily_average_kgco2', self._daily_average(ce_b_env, simulated_days), 'District', 'district'),
            self._metric('carbon_emissions_delta_daily_average_kgco2', self._daily_average(ce_c_env - ce_b_env, simulated_days), 'District', 'district'),
            self._metric('cost_control_total_eur', cost_c_env_raw, 'District', 'district'),
            self._metric('cost_baseline_total_eur', cost_b_env_raw, 'District', 'district'),
            self._metric('cost_delta_total_eur', cost_c_env_raw - cost_b_env_raw, 'District', 'district'),
            self._metric('cost_control_daily_average_eur', self._daily_average(cost_c_env_raw, simulated_days), 'District', 'district'),
            self._metric('cost_baseline_daily_average_eur', self._daily_average(cost_b_env_raw, simulated_days), 'District', 'district'),
            self._metric('cost_delta_daily_average_eur', self._daily_average(cost_c_env_raw - cost_b_env_raw, simulated_days), 'District', 'district'),
        ]

        ev_success_rate = None if ev_departures_total <= 0.0 else ev_departures_met / ev_departures_total
        ev_min_acceptable_rate = None if ev_departures_total <= 0.0 else ev_departures_min_acceptable / ev_departures_total
        ev_within_tolerance_rate = None if ev_departures_total <= 0.0 else ev_departures_within_tolerance / ev_departures_total
        ev_success_feasible_rate = (
            None if ev_departures_target_feasible <= 0.0
            else ev_departures_met_target_feasible / ev_departures_target_feasible
        )
        ev_min_acceptable_feasible_rate = (
            None if ev_departures_min_acceptable_feasible <= 0.0
            else ev_departures_min_acceptable_feasible_met / ev_departures_min_acceptable_feasible
        )
        ev_within_tolerance_feasible_rate = (
            None if ev_departures_within_tolerance_feasible <= 0.0
            else ev_departures_within_tolerance_feasible_met / ev_departures_within_tolerance_feasible
        )
        ev_deficit_mean = None if ev_departures_total <= 0.0 else ev_deficit_sum / ev_departures_total
        ev_shortfall_beyond_tolerance_mean = None if ev_departures_total <= 0.0 else ev_shortfall_beyond_tolerance_sum / ev_departures_total
        ev_surplus_mean = None if ev_departures_total <= 0.0 else ev_surplus_sum / ev_departures_total
        ev_absolute_error_mean = None if ev_departures_total <= 0.0 else ev_absolute_error_sum / ev_departures_total
        extended_district_rows.extend([
            self._metric('ev_departure_events_count', ev_departures_total, 'District', 'district'),
            self._metric('ev_departure_met_events_count', ev_departures_met, 'District', 'district'),
            self._metric('ev_departure_min_acceptable_events_count', ev_departures_min_acceptable, 'District', 'district'),
            self._metric('ev_departure_within_tolerance_events_count', ev_departures_within_tolerance, 'District', 'district'),
            self._metric('ev_departure_target_feasible_events_count', ev_departures_target_feasible, 'District', 'district'),
            self._metric('ev_departure_target_infeasible_events_count', ev_departures_target_infeasible, 'District', 'district'),
            self._metric('ev_departure_min_acceptable_feasible_events_count', ev_departures_min_acceptable_feasible, 'District', 'district'),
            self._metric('ev_departure_min_acceptable_infeasible_events_count', ev_departures_min_acceptable_infeasible, 'District', 'district'),
            self._metric('ev_departure_within_tolerance_feasible_events_count', ev_departures_within_tolerance_feasible, 'District', 'district'),
            self._metric('ev_departure_within_tolerance_infeasible_events_count', ev_departures_within_tolerance_infeasible, 'District', 'district'),
            self._metric('ev_departure_success_rate', ev_success_rate, 'District', 'district'),
            self._metric('ev_departure_min_acceptable_rate', ev_min_acceptable_rate, 'District', 'district'),
            self._metric('ev_departure_within_tolerance_rate', ev_within_tolerance_rate, 'District', 'district'),
            self._metric('ev_departure_success_feasible_rate', ev_success_feasible_rate, 'District', 'district'),
            self._metric('ev_departure_min_acceptable_feasible_rate', ev_min_acceptable_feasible_rate, 'District', 'district'),
            self._metric('ev_departure_within_tolerance_feasible_rate', ev_within_tolerance_feasible_rate, 'District', 'district'),
            self._metric('ev_departure_soc_deficit_mean', ev_deficit_mean, 'District', 'district'),
            self._metric('ev_departure_shortfall_beyond_tolerance_mean', ev_shortfall_beyond_tolerance_mean, 'District', 'district'),
            self._metric('ev_departure_soc_surplus_mean', ev_surplus_mean, 'District', 'district'),
            self._metric('ev_departure_soc_absolute_error_mean', ev_absolute_error_mean, 'District', 'district'),
            self._metric('ev_departure_tolerance_ratio', self._ev_departure_service_tolerance(), 'District', 'district'),
            self._metric('ev_charge_total_kwh', ev_charge_total, 'District', 'district'),
            self._metric('ev_v2g_export_total_kwh', ev_v2g_total, 'District', 'district'),
            self._metric('ev_connected_soc_gain_total_kwh', ev_connected_soc_gain_total, 'District', 'district'),
            self._metric('ev_energy_accounting_shortfall_kwh', ev_energy_accounting_shortfall_total, 'District', 'district'),
        ])

        district_bess_cycles = None if bess_capacity_total <= 0.0 else bess_throughput_total / (2.0 * bess_capacity_total)
        district_bess_fade = None if bess_capacity_total <= 0.0 else bess_capacity_loss_total / bess_capacity_total
        extended_district_rows.extend([
            self._metric('bess_charge_total_kwh', bess_charge_total, 'District', 'district'),
            self._metric('bess_discharge_total_kwh', bess_discharge_total, 'District', 'district'),
            self._metric('bess_throughput_total_kwh', bess_throughput_total, 'District', 'district'),
            self._metric('bess_equivalent_full_cycles', district_bess_cycles, 'District', 'district'),
            self._metric('bess_capacity_fade_ratio', district_bess_fade, 'District', 'district'),
        ])

        final_t = int(max(getattr(env, 'time_step', 0), 0))
        district_pv_metrics = self._compute_district_pv_metrics(kpi_buildings, building_windows, final_t=final_t)
        pv_generation_total = district_pv_metrics['pv_generation_total_kwh']
        pv_export_total = district_pv_metrics['pv_export_total_kwh']
        district_pv_ratio = district_pv_metrics['pv_self_consumption_ratio']
        extended_district_rows.extend([
            self._metric('pv_generation_total_kwh', pv_generation_total, 'District', 'district'),
            self._metric('pv_export_total_kwh', pv_export_total, 'District', 'district'),
            self._metric('pv_generation_daily_average_kwh', self._daily_average(pv_generation_total, simulated_days), 'District', 'district'),
            self._metric('pv_export_daily_average_kwh', self._daily_average(pv_export_total, simulated_days), 'District', 'district'),
            self._metric('pv_self_consumption_ratio', district_pv_ratio, 'District', 'district'),
        ])

        district_phase_imbalance = None if phase_imbalance_count <= 0.0 else phase_imbalance_sum / phase_imbalance_count
        extended_district_rows.extend([
            self._metric('electrical_service_violation_total_kwh', phase_violation_total, 'District', 'district'),
            self._metric('electrical_service_violation_time_step_count', phase_violation_count, 'District', 'district'),
            self._metric('electrical_service_requested_pressure_total_kwh', phase_requested_pressure_total, 'District', 'district'),
            self._metric('electrical_service_requested_pressure_time_step_count', phase_requested_pressure_count, 'District', 'district'),
            self._metric('phase_imbalance_ratio_average', district_phase_imbalance, 'District', 'district'),
        ])

        phase_union = ['L1', 'L2', 'L3']
        for phase_name in phase_union:
            phase_series = np.zeros(final_t + 1, dtype='float64')
            has_phase_data = False
            for building in kpi_buildings:
                history_map = getattr(building, '_charging_phase_power_history_kw', {}) or {}
                if phase_name not in history_map:
                    continue
                t_start, t_end = building_windows.get(building.name, (0, -1))
                if t_end < t_start:
                    continue
                values = np.array(history_map[phase_name], dtype='float64')
                start = max(int(t_start), 0)
                end = min(int(t_end), final_t, len(values) - 1)
                if end < start:
                    continue
                phase_series[start:end + 1] += values[start:end + 1]
                has_phase_data = True

            if not has_phase_data:
                continue

            extended_district_rows.append(
                self._metric(f'phase_import_peak_kw_{phase_name}', float(np.clip(phase_series, 0.0, None).max(initial=0.0)), 'District', 'district')
            )
            extended_district_rows.append(
                self._metric(f'phase_export_peak_kw_{phase_name}', float(np.clip(-phase_series, 0.0, None).max(initial=0.0)), 'District', 'district')
            )

        market_by_building, market_district = self._collect_market_totals(building_names)

        for building_name in building_names:
            t_start, t_end = building_windows.get(building_name, (0, -1))
            building_days = self._window_days(t_start, t_end, env.seconds_per_time_step)
            totals = market_by_building.get(building_name, {})
            local_import = self._to_scalar(totals.get('community_local_import_total_kwh'), 0.0)
            local_export = self._to_scalar(totals.get('community_local_export_total_kwh'), 0.0)
            grid_import = self._to_scalar(totals.get('community_grid_import_after_local_total_kwh'), 0.0)
            grid_export = self._to_scalar(totals.get('community_grid_export_after_local_total_kwh'), 0.0)
            import_share = None if (local_import + grid_import) <= 0.0 else local_import / (local_import + grid_import)
            export_share = None if (local_export + grid_export) <= 0.0 else local_export / (local_export + grid_export)

            extended_building_rows.extend([
                self._metric('community_local_import_total_kwh', local_import, building_name, 'building'),
                self._metric('community_local_export_total_kwh', local_export, building_name, 'building'),
                self._metric('community_grid_import_after_local_total_kwh', grid_import, building_name, 'building'),
                self._metric('community_grid_export_after_local_total_kwh', grid_export, building_name, 'building'),
                self._metric('community_local_import_daily_average_kwh', self._daily_average(local_import, building_days), building_name, 'building'),
                self._metric('community_local_export_daily_average_kwh', self._daily_average(local_export, building_days), building_name, 'building'),
                self._metric('community_grid_import_after_local_daily_average_kwh', self._daily_average(grid_import, building_days), building_name, 'building'),
                self._metric('community_grid_export_after_local_daily_average_kwh', self._daily_average(grid_export, building_days), building_name, 'building'),
                self._metric('community_settled_cost_total_eur', totals.get('community_settled_cost_total_eur', 0.0), building_name, 'building'),
                self._metric('community_counterfactual_cost_total_eur', totals.get('community_counterfactual_cost_total_eur', 0.0), building_name, 'building'),
                self._metric('community_market_savings_total_eur', totals.get('community_market_savings_total_eur', 0.0), building_name, 'building'),
                self._metric('community_settled_cost_daily_average_eur', self._daily_average(totals.get('community_settled_cost_total_eur', 0.0), building_days), building_name, 'building'),
                self._metric('community_counterfactual_cost_daily_average_eur', self._daily_average(totals.get('community_counterfactual_cost_total_eur', 0.0), building_days), building_name, 'building'),
                self._metric('community_market_savings_daily_average_eur', self._daily_average(totals.get('community_market_savings_total_eur', 0.0), building_days), building_name, 'building'),
                self._metric('community_local_share_of_demand', import_share, building_name, 'building'),
                self._metric('community_local_share_of_export', export_share, building_name, 'building'),
            ])

        district_local_import = market_district['community_local_import_total_kwh']
        district_local_export = market_district['community_local_export_total_kwh']
        district_grid_import = market_district['community_grid_import_after_local_total_kwh']
        district_grid_export = market_district['community_grid_export_after_local_total_kwh']

        extended_district_rows.extend([
            self._metric('community_local_import_total_kwh', district_local_import, 'District', 'district'),
            self._metric('community_local_export_total_kwh', district_local_export, 'District', 'district'),
            self._metric('community_grid_import_after_local_total_kwh', district_grid_import, 'District', 'district'),
            self._metric('community_grid_export_after_local_total_kwh', district_grid_export, 'District', 'district'),
            self._metric('community_local_import_daily_average_kwh', self._daily_average(district_local_import, simulated_days), 'District', 'district'),
            self._metric('community_local_export_daily_average_kwh', self._daily_average(district_local_export, simulated_days), 'District', 'district'),
            self._metric('community_grid_import_after_local_daily_average_kwh', self._daily_average(district_grid_import, simulated_days), 'District', 'district'),
            self._metric('community_grid_export_after_local_daily_average_kwh', self._daily_average(district_grid_export, simulated_days), 'District', 'district'),
            self._metric('community_settled_cost_total_eur', market_district['community_settled_cost_total_eur'], 'District', 'district'),
            self._metric('community_counterfactual_cost_total_eur', market_district['community_counterfactual_cost_total_eur'], 'District', 'district'),
            self._metric('community_market_savings_total_eur', market_district['community_market_savings_total_eur'], 'District', 'district'),
            self._metric('community_settled_cost_daily_average_eur', self._daily_average(market_district['community_settled_cost_total_eur'], simulated_days), 'District', 'district'),
            self._metric('community_counterfactual_cost_daily_average_eur', self._daily_average(market_district['community_counterfactual_cost_total_eur'], simulated_days), 'District', 'district'),
            self._metric('community_market_savings_daily_average_eur', self._daily_average(market_district['community_market_savings_total_eur'], simulated_days), 'District', 'district'),
            self._metric(
                'community_local_share_of_demand',
                None if (district_local_import + district_grid_import) <= 0.0 else district_local_import / (district_local_import + district_grid_import),
                'District',
                'district',
            ),
            self._metric(
                'community_local_share_of_export',
                None if (district_local_export + district_grid_export) <= 0.0 else district_local_export / (district_local_export + district_grid_export),
                'District',
                'district',
            ),
        ])

        equity_distribution = self._equity_distribution_metrics(np.array(list(equity_valid_benefits.values()), dtype='float64'))
        non_negative_benefits = {name: max(value, 0.0) for name, value in equity_valid_benefits.items()}
        has_complete_manual_groups = all(
            equity_group_by_building.get(name) in {'asset_rich', 'asset_poor'}
            for name in building_names
        )
        equity_bpr = self._equity_bpr(non_negative_benefits, equity_group_by_building) if has_complete_manual_groups else None

        extended_district_rows.extend([
            self._metric('equity_gini_benefit', equity_distribution['equity_gini_benefit'], 'District', 'district'),
            self._metric('equity_cr20_benefit', equity_distribution['equity_cr20_benefit'], 'District', 'district'),
            self._metric('equity_losers_percent', equity_distribution['equity_losers_percent'], 'District', 'district'),
            self._metric('equity_bpr_asset_poor_over_rich', equity_bpr, 'District', 'district'),
        ])

        district_deferrable = {
            'completed_cycles': 0.0,
            'missed_cycles': 0.0,
            'served_energy_kwh': 0.0,
            'unserved_energy_kwh': 0.0,
        }
        delay_weighted_sum = 0.0
        delay_weight = 0.0
        for building in kpi_buildings:
            building_summary = {
                'completed_cycles': 0.0,
                'missed_cycles': 0.0,
                'served_energy_kwh': 0.0,
                'unserved_energy_kwh': 0.0,
            }
            building_delay_weighted_sum = 0.0
            building_delay_weight = 0.0
            for appliance in self._deferrable_appliances_for_metrics(building):
                summary = appliance.service_summary()
                completed = self._to_scalar(summary.get('completed_cycles'), 0.0)
                missed = self._to_scalar(summary.get('missed_cycles'), 0.0)
                for key in building_summary:
                    building_summary[key] += self._to_scalar(summary.get(key), 0.0)
                if completed > 0.0:
                    delay = self._to_scalar(summary.get('average_start_delay_hours'), 0.0)
                    building_delay_weighted_sum += delay * completed
                    building_delay_weight += completed

            requested = building_summary['completed_cycles'] + building_summary['missed_cycles']
            service_level = None if requested <= 0.0 else building_summary['completed_cycles'] / requested
            avg_delay = 0.0 if building_delay_weight <= 0.0 else building_delay_weighted_sum / building_delay_weight
            extended_building_rows.extend([
                self._metric('deferrable_appliance_completed_cycles_count', building_summary['completed_cycles'], building.name, 'building'),
                self._metric('deferrable_appliance_missed_cycles_count', building_summary['missed_cycles'], building.name, 'building'),
                self._metric('deferrable_appliance_service_level_ratio', service_level, building.name, 'building'),
                self._metric('deferrable_appliance_served_energy_total_kwh', building_summary['served_energy_kwh'], building.name, 'building'),
                self._metric('deferrable_appliance_unserved_energy_total_kwh', building_summary['unserved_energy_kwh'], building.name, 'building'),
                self._metric('deferrable_appliance_average_start_delay_hours', avg_delay, building.name, 'building'),
            ])

            for key in district_deferrable:
                district_deferrable[key] += building_summary[key]
            delay_weighted_sum += building_delay_weighted_sum
            delay_weight += building_delay_weight

        district_requested = district_deferrable['completed_cycles'] + district_deferrable['missed_cycles']
        district_service_level = None if district_requested <= 0.0 else district_deferrable['completed_cycles'] / district_requested
        district_avg_delay = 0.0 if delay_weight <= 0.0 else delay_weighted_sum / delay_weight
        extended_district_rows.extend([
            self._metric('deferrable_appliance_completed_cycles_count', district_deferrable['completed_cycles'], 'District', 'district'),
            self._metric('deferrable_appliance_missed_cycles_count', district_deferrable['missed_cycles'], 'District', 'district'),
            self._metric('deferrable_appliance_service_level_ratio', district_service_level, 'District', 'district'),
            self._metric('deferrable_appliance_served_energy_total_kwh', district_deferrable['served_energy_kwh'], 'District', 'district'),
            self._metric('deferrable_appliance_unserved_energy_total_kwh', district_deferrable['unserved_energy_kwh'], 'District', 'district'),
            self._metric('deferrable_appliance_average_start_delay_hours', district_avg_delay, 'District', 'district'),
        ])

        # Escalators deliberately use aggregate passenger demand rather than a
        # queueing model. These KPIs expose the resulting energy/service tradeoff
        # alongside the existing electricity cost metrics.
        district_escalator = {
            'requested_passengers': 0.0,
            'served_passengers': 0.0,
            'unserved_passengers': 0.0,
            'electricity_consumption_kwh': 0.0,
            'state_changes_count': 0.0,
        }
        for building in kpi_buildings:
            building_escalator = dict.fromkeys(district_escalator, 0.0)
            for escalator in getattr(building, 'escalators', []) or []:
                summary = escalator.service_summary()
                for key in building_escalator:
                    building_escalator[key] += self._to_scalar(summary.get(key), 0.0)

            requested = building_escalator['requested_passengers']
            service_level = None if requested <= 0.0 else building_escalator['served_passengers'] / requested
            extended_building_rows.extend([
                self._metric('escalator_requested_passengers_total_count', requested, building.name, 'building'),
                self._metric('escalator_served_passengers_total_count', building_escalator['served_passengers'], building.name, 'building'),
                self._metric('escalator_unserved_passengers_total_count', building_escalator['unserved_passengers'], building.name, 'building'),
                self._metric('escalator_service_level_ratio', service_level, building.name, 'building'),
                self._metric('escalator_electricity_consumption_total_kwh', building_escalator['electricity_consumption_kwh'], building.name, 'building'),
                self._metric('escalator_state_changes_count', building_escalator['state_changes_count'], building.name, 'building'),
            ])
            for key in district_escalator:
                district_escalator[key] += building_escalator[key]

        district_requested = district_escalator['requested_passengers']
        district_service_level = None if district_requested <= 0.0 else district_escalator['served_passengers'] / district_requested
        extended_district_rows.extend([
            self._metric('escalator_requested_passengers_total_count', district_requested, 'District', 'district'),
            self._metric('escalator_served_passengers_total_count', district_escalator['served_passengers'], 'District', 'district'),
            self._metric('escalator_unserved_passengers_total_count', district_escalator['unserved_passengers'], 'District', 'district'),
            self._metric('escalator_service_level_ratio', district_service_level, 'District', 'district'),
            self._metric('escalator_electricity_consumption_total_kwh', district_escalator['electricity_consumption_kwh'], 'District', 'district'),
            self._metric('escalator_state_changes_count', district_escalator['state_changes_count'], 'District', 'district'),
        ])

        extended_building = pd.DataFrame(extended_building_rows)
        extended_district = pd.DataFrame(extended_district_rows)

        cost_functions = pd.concat([legacy_cost_functions, extended_district, extended_building], ignore_index=True, sort=False)

        return cost_functions

    @staticmethod
    def _v2_name(
        level: str,
        family: str,
        subfamily: str,
        metric: str,
        variant: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> str:
        tokens = [level, family, subfamily, metric]

        if variant not in (None, ''):
            tokens.append(str(variant))

        if unit not in (None, ''):
            tokens.append(str(unit))

        return '_'.join(tokens)

    def evaluate_legacy(
        self,
        control_condition=None,
        baseline_condition=None,
        comfort_band: float = None,
        *,
        evaluation_condition_cls,
        dynamics_building_cls,
    ) -> pd.DataFrame:
        all_metrics = self.evaluate(
            control_condition=control_condition,
            baseline_condition=baseline_condition,
            comfort_band=comfort_band,
            evaluation_condition_cls=evaluation_condition_cls,
            dynamics_building_cls=dynamics_building_cls,
        )

        legacy = all_metrics[all_metrics['cost_function'].isin(self.LEGACY_COST_FUNCTIONS)].copy()
        return legacy.reset_index(drop=True)

    def _add_business_as_usual_records(
        self,
        records: Dict[Tuple[str, str, str], Dict[str, object]],
        v2,
        simulated_days: float,
        *,
        evaluation_condition_cls,
        dynamics_building_cls,
    ):
        env = self.env
        result = env.run_business_as_usual_baseline()
        bau_df = result.kpis_v2
        if bau_df.empty:
            return

        bau_lookup = {
            (str(row['level']), str(row['name']), str(row['cost_function'])): row['value']
            for _, row in bau_df.iterrows()
        }

        def put(level: str, name: str, cost_function: str, value):
            records[(level, name, cost_function)] = {
                'cost_function': cost_function,
                'value': value,
                'name': name,
                'level': level,
            }

        def current(level: str, name: str, cost_function: str):
            return records.get((level, name, cost_function), {}).get('value')

        def bau(level: str, name: str, cost_function: str):
            return bau_lookup.get((level, name, cost_function))

        def finite_delta(a, b):
            a = self._to_scalar(a, np.nan)
            b = self._to_scalar(b, np.nan)
            return None if not (np.isfinite(a) and np.isfinite(b)) else float(a - b)

        def add_total_daily_ratio(
            level: str,
            name: str,
            *,
            family: str,
            total_subfamily: str,
            control_total_metric: str,
            bau_total_metric: str,
            delta_total_metric: str,
            daily_subfamily: str,
            control_daily_metric: str,
            bau_daily_metric: str,
            delta_daily_metric: str,
            ratio_metric: str,
            unit: str,
            control_variant: Optional[str] = None,
            bau_variant: Optional[str] = None,
            delta_variant: Optional[str] = None,
        ):
            control_total_key = v2(level, family, total_subfamily, control_total_metric, control_variant, unit)
            bau_total = bau(level, name, control_total_key)
            control_total = current(level, name, control_total_key)
            put(level, name, v2(level, family, total_subfamily, bau_total_metric, bau_variant, unit), bau_total)
            put(
                level,
                name,
                v2(level, family, total_subfamily, delta_total_metric, delta_variant, unit),
                finite_delta(control_total, bau_total),
            )
            put(
                level,
                name,
                v2(level, family, 'ratio_to_business_as_usual', ratio_metric, None, 'ratio'),
                self._safe_div(control_total, bau_total),
            )

            control_daily_key = v2(level, family, daily_subfamily, control_daily_metric, control_variant, unit)
            bau_daily = bau(level, name, control_daily_key)
            control_daily = current(level, name, control_daily_key)
            put(level, name, v2(level, family, daily_subfamily, bau_daily_metric, bau_variant, unit), bau_daily)
            put(
                level,
                name,
                v2(level, family, daily_subfamily, delta_daily_metric, delta_variant, unit),
                finite_delta(control_daily, bau_daily),
            )

        def add_plain_metric(
            level: str,
            name: str,
            *,
            family: str,
            subfamily: str,
            metric: str,
            unit: str,
            current_key: str,
            ratio_metric: str,
        ):
            bau_value = bau(level, name, current_key)
            control_value = current(level, name, current_key)
            put(level, name, v2(level, family, subfamily, metric, 'business_as_usual', unit), bau_value)
            put(
                level,
                name,
                v2(level, family, subfamily, metric, 'delta_to_business_as_usual', unit),
                finite_delta(control_value, bau_value),
            )
            put(
                level,
                name,
                v2(level, family, 'ratio_to_business_as_usual', ratio_metric, None, 'ratio'),
                self._safe_div(control_value, bau_value),
            )

        names_by_level = {}
        for level, name, _ in records.keys():
            names_by_level.setdefault(level, set()).add(name)

        for level, names in names_by_level.items():
            for name in names:
                add_total_daily_ratio(
                    level,
                    name,
                    family='cost',
                    total_subfamily='total',
                    control_total_metric='control',
                    bau_total_metric='business_as_usual',
                    delta_total_metric='delta_to_business_as_usual',
                    daily_subfamily='daily_average',
                    control_daily_metric='control',
                    bau_daily_metric='business_as_usual',
                    delta_daily_metric='delta_to_business_as_usual',
                    ratio_metric='total',
                    unit='eur',
                )
                add_total_daily_ratio(
                    level,
                    name,
                    family='emissions',
                    total_subfamily='total',
                    control_total_metric='control',
                    bau_total_metric='business_as_usual',
                    delta_total_metric='delta_to_business_as_usual',
                    daily_subfamily='daily_average',
                    control_daily_metric='control',
                    bau_daily_metric='business_as_usual',
                    delta_daily_metric='delta_to_business_as_usual',
                    ratio_metric='total',
                    unit='kgco2',
                )

                for metric, ratio_metric in [
                    ('import', 'import_total'),
                    ('export', 'export_total'),
                    ('net_exchange', 'net_exchange_total'),
                ]:
                    add_total_daily_ratio(
                        level,
                        name,
                        family='energy_grid',
                        total_subfamily='total',
                        control_total_metric=metric,
                        bau_total_metric=metric,
                        delta_total_metric=metric,
                        daily_subfamily='daily_average',
                        control_daily_metric=metric,
                        bau_daily_metric=metric,
                        delta_daily_metric=metric,
                        ratio_metric=ratio_metric,
                        unit='kwh',
                        control_variant='control',
                        bau_variant='business_as_usual',
                        delta_variant='delta_to_business_as_usual',
                    )

                for metric, unit in [
                    ('charge', 'kwh'),
                    ('v2g_export', 'kwh'),
                ]:
                    add_plain_metric(
                        level,
                        name,
                        family='ev',
                        subfamily='total',
                        metric=metric,
                        unit=unit,
                        current_key=v2(level, 'ev', 'total', metric, None, unit),
                        ratio_metric=f'{metric}_total',
                    )

                for metric, unit in [
                    ('charge', 'kwh'),
                    ('discharge', 'kwh'),
                    ('throughput', 'kwh'),
                    ('equivalent_full_cycles', 'count'),
                    ('capacity_fade', 'ratio'),
                ]:
                    subfamily = 'health' if metric in {'equivalent_full_cycles', 'capacity_fade'} else 'total'
                    add_plain_metric(
                        level,
                        name,
                        family='battery',
                        subfamily=subfamily,
                        metric=metric,
                        unit=unit,
                        current_key=v2(level, 'battery', subfamily, metric, None, unit),
                        ratio_metric=metric,
                    )

                for metric in ['generation', 'export']:
                    add_total_daily_ratio(
                        level,
                        name,
                        family='solar_self_consumption',
                        total_subfamily='total',
                        control_total_metric=metric,
                        bau_total_metric=metric,
                        delta_total_metric=metric,
                        daily_subfamily='daily_average',
                        control_daily_metric=metric,
                        bau_daily_metric=metric,
                        delta_daily_metric=metric,
                        ratio_metric=f'{metric}_total',
                        unit='kwh',
                        bau_variant='business_as_usual',
                        delta_variant='delta_to_business_as_usual',
                    )

                add_plain_metric(
                    level,
                    name,
                    family='solar_self_consumption',
                    subfamily='ratio',
                    metric='self_consumption',
                    unit='ratio',
                    current_key=v2(level, 'solar_self_consumption', 'ratio', 'self_consumption', None, 'ratio'),
                    ratio_metric='self_consumption',
                )

                for metric, unit in [
                    ('completed_cycles', 'count'),
                    ('missed_cycles', 'count'),
                    ('service_level', 'ratio'),
                    ('served_energy_total', 'kwh'),
                    ('unserved_energy_total', 'kwh'),
                    ('average_start_delay', 'hours'),
                ]:
                    add_plain_metric(
                        level,
                        name,
                        family='deferrable_appliance',
                        subfamily='service',
                        metric=metric,
                        unit=unit,
                        current_key=v2(level, 'deferrable_appliance', 'service', metric, None, unit),
                        ratio_metric=metric,
                    )

        self._add_business_as_usual_shape_records(records, result.env, v2, simulated_days)

    def _add_business_as_usual_shape_records(
        self,
        records: Dict[Tuple[str, str, str], Dict[str, object]],
        bau_env: "CityLearnEnv",
        v2,
        simulated_days: float,
    ):
        env = self.env
        daily_steps = self._window_steps(24.0 * 3600.0, env.seconds_per_time_step)
        monthly_steps = self._window_steps(730.0 * 3600.0, env.seconds_per_time_step)
        current_net = np.array(env.net_electricity_consumption, dtype='float64')
        bau_net = np.array(bau_env.net_electricity_consumption, dtype='float64')
        step_hours = max(float(env.seconds_per_time_step) / 3600.0, 1.0e-12)
        current_net_power = current_net / step_hours
        bau_net_power = bau_net / step_hours

        metrics = [
            ('ramping_average', self._ramping_final(current_net_power), self._ramping_final(bau_net_power), 'kw'),
            (
                'load_factor_penalty_daily_average',
                self._one_minus_load_factor_final(current_net_power, window=daily_steps),
                self._one_minus_load_factor_final(bau_net_power, window=daily_steps),
                'ratio',
            ),
            (
                'load_factor_penalty_monthly_average',
                self._one_minus_load_factor_final(current_net_power, window=monthly_steps),
                self._one_minus_load_factor_final(bau_net_power, window=monthly_steps),
                'ratio',
            ),
            (
                'peak_daily_average',
                self._peak_final(current_net_power, window=daily_steps),
                self._peak_final(bau_net_power, window=daily_steps),
                'kw',
            ),
            (
                'peak_all_time_average',
                self._peak_final(current_net_power, window=max(int(getattr(env, 'time_steps', 1)), 1)),
                self._peak_final(bau_net_power, window=max(int(getattr(bau_env, 'time_steps', 1)), 1)),
                'kw',
            ),
        ]

        for metric, control_value, bau_value, unit in metrics:
            records[('district', 'District', v2('district', 'energy_grid', 'shape_quality', metric, 'business_as_usual', unit))] = {
                'cost_function': v2('district', 'energy_grid', 'shape_quality', metric, 'business_as_usual', unit),
                'value': bau_value,
                'name': 'District',
                'level': 'district',
            }
            records[('district', 'District', v2('district', 'energy_grid', 'shape_quality', metric, 'delta_to_business_as_usual', unit))] = {
                'cost_function': v2('district', 'energy_grid', 'shape_quality', metric, 'delta_to_business_as_usual', unit),
                'value': control_value - bau_value,
                'name': 'District',
                'level': 'district',
            }
            records[('district', 'District', v2('district', 'energy_grid', 'shape_quality', f'{metric}_to_business_as_usual', None, 'ratio'))] = {
                'cost_function': v2('district', 'energy_grid', 'shape_quality', f'{metric}_to_business_as_usual', None, 'ratio'),
                'value': self._safe_div(control_value, bau_value),
                'name': 'District',
                'level': 'district',
            }

    def evaluate_v2(
        self,
        control_condition=None,
        baseline_condition=None,
        comfort_band: float = None,
        include_business_as_usual: bool = True,
        *,
        evaluation_condition_cls,
        dynamics_building_cls,
    ) -> pd.DataFrame:
        env = self.env
        all_metrics = self.evaluate(
            control_condition=control_condition,
            baseline_condition=baseline_condition,
            comfort_band=comfort_band,
            evaluation_condition_cls=evaluation_condition_cls,
            dynamics_building_cls=dynamics_building_cls,
        )
        legacy_df = all_metrics[all_metrics['cost_function'].isin(self.LEGACY_COST_FUNCTIONS)].copy()
        extended_df = all_metrics[~all_metrics['cost_function'].isin(self.LEGACY_COST_FUNCTIONS)].copy()
        simulated_days = self._simulated_days(env)

        records: Dict[Tuple[str, str, str], Dict[str, object]] = {}

        def put(level: str, name: str, cost_function: str, value):
            records[(level, name, cost_function)] = {
                'cost_function': cost_function,
                'value': value,
                'name': name,
                'level': level,
            }

        def v2(
            level: str,
            family: str,
            subfamily: str,
            metric: str,
            variant: Optional[str] = None,
            unit: Optional[str] = None,
        ) -> str:
            return self._v2_name(
                level,
                family,
                subfamily,
                metric,
                variant=variant,
                unit=unit,
            )

        def map_from(
            source_df: pd.DataFrame,
            old_name: str,
            family: str,
            subfamily: str,
            metric: str,
            variant: Optional[str] = None,
            unit: Optional[str] = None,
        ):
            subset = source_df[source_df['cost_function'] == old_name]
            for _, row in subset.iterrows():
                level = str(row['level'])
                put(
                    level,
                    str(row['name']),
                    v2(level, family, subfamily, metric, variant, unit),
                    row['value'],
                )

        # Ratios to baseline and comfort/resilience from legacy.
        legacy_map = [
            ('electricity_consumption_total', 'energy_grid', 'ratio_to_baseline', 'import_total', None, 'ratio'),
            ('zero_net_energy', 'energy_grid', 'ratio_to_baseline', 'net_exchange_total', None, 'ratio'),
            ('carbon_emissions_total', 'emissions', 'ratio_to_baseline', 'total', None, 'ratio'),
            ('cost_total', 'cost', 'ratio_to_baseline', 'total', None, 'ratio'),
            ('ramping_average', 'energy_grid', 'shape_quality', 'ramping_average_to_baseline', None, 'ratio'),
            ('daily_one_minus_load_factor_average', 'energy_grid', 'shape_quality', 'load_factor_penalty_daily_average_to_baseline', None, 'ratio'),
            ('monthly_one_minus_load_factor_average', 'energy_grid', 'shape_quality', 'load_factor_penalty_monthly_average_to_baseline', None, 'ratio'),
            ('daily_peak_average', 'energy_grid', 'shape_quality', 'peak_daily_average_to_baseline', None, 'ratio'),
            ('all_time_peak_average', 'energy_grid', 'shape_quality', 'peak_all_time_average_to_baseline', None, 'ratio'),
            ('discomfort_proportion', 'comfort_resilience', 'discomfort', 'overall', None, 'ratio'),
            ('discomfort_cold_proportion', 'comfort_resilience', 'discomfort', 'cold', None, 'ratio'),
            ('discomfort_hot_proportion', 'comfort_resilience', 'discomfort', 'hot', None, 'ratio'),
            ('discomfort_cold_delta_minimum', 'comfort_resilience', 'discomfort', 'cold_delta', 'min', 'c'),
            ('discomfort_cold_delta_maximum', 'comfort_resilience', 'discomfort', 'cold_delta', 'max', 'c'),
            ('discomfort_cold_delta_average', 'comfort_resilience', 'discomfort', 'cold_delta', 'average', 'c'),
            ('discomfort_hot_delta_minimum', 'comfort_resilience', 'discomfort', 'hot_delta', 'min', 'c'),
            ('discomfort_hot_delta_maximum', 'comfort_resilience', 'discomfort', 'hot_delta', 'max', 'c'),
            ('discomfort_hot_delta_average', 'comfort_resilience', 'discomfort', 'hot_delta', 'average', 'c'),
            ('one_minus_thermal_resilience_proportion', 'comfort_resilience', 'resilience', 'one_minus_thermal', None, 'ratio'),
            ('power_outage_normalized_unserved_energy_total', 'comfort_resilience', 'resilience', 'unserved_energy_outage_normalized', None, 'ratio'),
            ('annual_normalized_unserved_energy_total', 'comfort_resilience', 'resilience', 'unserved_energy_annual_normalized', None, 'ratio'),
        ]
        for old_name, family, subfamily, metric, variant, unit in legacy_map:
            map_from(legacy_df, old_name, family, subfamily, metric, variant, unit)

        extended_map = [
            ('cost_control_total_eur', 'cost', 'total', 'control', None, 'eur'),
            ('cost_baseline_total_eur', 'cost', 'total', 'baseline', None, 'eur'),
            ('cost_delta_total_eur', 'cost', 'total', 'delta', None, 'eur'),
            ('cost_control_daily_average_eur', 'cost', 'daily_average', 'control', None, 'eur'),
            ('cost_baseline_daily_average_eur', 'cost', 'daily_average', 'baseline', None, 'eur'),
            ('cost_delta_daily_average_eur', 'cost', 'daily_average', 'delta', None, 'eur'),
            ('electricity_consumption_control_total_kwh', 'energy_grid', 'total', 'import', 'control', 'kwh'),
            ('electricity_consumption_baseline_total_kwh', 'energy_grid', 'total', 'import', 'baseline', 'kwh'),
            ('electricity_consumption_delta_total_kwh', 'energy_grid', 'total', 'import', 'delta', 'kwh'),
            ('electricity_consumption_control_daily_average_kwh', 'energy_grid', 'daily_average', 'import', 'control', 'kwh'),
            ('electricity_consumption_baseline_daily_average_kwh', 'energy_grid', 'daily_average', 'import', 'baseline', 'kwh'),
            ('electricity_consumption_delta_daily_average_kwh', 'energy_grid', 'daily_average', 'import', 'delta', 'kwh'),
            ('electricity_export_control_total_kwh', 'energy_grid', 'total', 'export', 'control', 'kwh'),
            ('electricity_export_baseline_total_kwh', 'energy_grid', 'total', 'export', 'baseline', 'kwh'),
            ('electricity_export_delta_total_kwh', 'energy_grid', 'total', 'export', 'delta', 'kwh'),
            ('electricity_export_control_daily_average_kwh', 'energy_grid', 'daily_average', 'export', 'control', 'kwh'),
            ('electricity_export_baseline_daily_average_kwh', 'energy_grid', 'daily_average', 'export', 'baseline', 'kwh'),
            ('electricity_export_delta_daily_average_kwh', 'energy_grid', 'daily_average', 'export', 'delta', 'kwh'),
            ('zero_net_energy_control_total_kwh', 'energy_grid', 'total', 'net_exchange', 'control', 'kwh'),
            ('zero_net_energy_baseline_total_kwh', 'energy_grid', 'total', 'net_exchange', 'baseline', 'kwh'),
            ('zero_net_energy_delta_total_kwh', 'energy_grid', 'total', 'net_exchange', 'delta', 'kwh'),
            ('zero_net_energy_control_daily_average_kwh', 'energy_grid', 'daily_average', 'net_exchange', 'control', 'kwh'),
            ('zero_net_energy_baseline_daily_average_kwh', 'energy_grid', 'daily_average', 'net_exchange', 'baseline', 'kwh'),
            ('zero_net_energy_delta_daily_average_kwh', 'energy_grid', 'daily_average', 'net_exchange', 'delta', 'kwh'),
            ('carbon_emissions_control_total_kgco2', 'emissions', 'total', 'control', None, 'kgco2'),
            ('carbon_emissions_baseline_total_kgco2', 'emissions', 'total', 'baseline', None, 'kgco2'),
            ('carbon_emissions_delta_total_kgco2', 'emissions', 'total', 'delta', None, 'kgco2'),
            ('carbon_emissions_control_daily_average_kgco2', 'emissions', 'daily_average', 'control', None, 'kgco2'),
            ('carbon_emissions_baseline_daily_average_kgco2', 'emissions', 'daily_average', 'baseline', None, 'kgco2'),
            ('carbon_emissions_delta_daily_average_kgco2', 'emissions', 'daily_average', 'delta', None, 'kgco2'),
            ('pv_generation_total_kwh', 'solar_self_consumption', 'total', 'generation', None, 'kwh'),
            ('pv_export_total_kwh', 'solar_self_consumption', 'total', 'export', None, 'kwh'),
            ('pv_generation_daily_average_kwh', 'solar_self_consumption', 'daily_average', 'generation', None, 'kwh'),
            ('pv_export_daily_average_kwh', 'solar_self_consumption', 'daily_average', 'export', None, 'kwh'),
            ('pv_self_consumption_ratio', 'solar_self_consumption', 'ratio', 'self_consumption', None, 'ratio'),
            ('ev_departure_events_count', 'ev', 'events', 'departure', None, 'count'),
            ('ev_departure_met_events_count', 'ev', 'events', 'departure_met', None, 'count'),
            ('ev_departure_min_acceptable_events_count', 'ev', 'events', 'departure_min_acceptable', None, 'count'),
            ('ev_departure_within_tolerance_events_count', 'ev', 'events', 'departure_within_tolerance', None, 'count'),
            ('ev_departure_target_feasible_events_count', 'ev', 'events', 'departure_target_feasible', None, 'count'),
            ('ev_departure_target_infeasible_events_count', 'ev', 'events', 'departure_target_infeasible', None, 'count'),
            ('ev_departure_min_acceptable_feasible_events_count', 'ev', 'events', 'departure_min_acceptable_feasible', None, 'count'),
            ('ev_departure_min_acceptable_infeasible_events_count', 'ev', 'events', 'departure_min_acceptable_infeasible', None, 'count'),
            ('ev_departure_within_tolerance_feasible_events_count', 'ev', 'events', 'departure_within_tolerance_feasible', None, 'count'),
            ('ev_departure_within_tolerance_infeasible_events_count', 'ev', 'events', 'departure_within_tolerance_infeasible', None, 'count'),
            ('ev_departure_success_rate', 'ev', 'performance', 'departure_success', None, 'ratio'),
            ('ev_departure_min_acceptable_rate', 'ev', 'performance', 'departure_min_acceptable', None, 'ratio'),
            ('ev_departure_within_tolerance_rate', 'ev', 'performance', 'departure_within_tolerance', None, 'ratio'),
            ('ev_departure_success_feasible_rate', 'ev', 'performance', 'departure_success', 'feasible', 'ratio'),
            ('ev_departure_min_acceptable_feasible_rate', 'ev', 'performance', 'departure_min_acceptable', 'feasible', 'ratio'),
            ('ev_departure_within_tolerance_feasible_rate', 'ev', 'performance', 'departure_within_tolerance', 'feasible', 'ratio'),
            ('ev_departure_soc_deficit_mean', 'ev', 'performance', 'departure_soc_deficit_mean', None, 'ratio'),
            ('ev_departure_shortfall_beyond_tolerance_mean', 'ev', 'performance', 'departure_shortfall_beyond_tolerance_mean', None, 'ratio'),
            ('ev_departure_soc_surplus_mean', 'ev', 'performance', 'departure_soc_surplus_mean', None, 'ratio'),
            ('ev_departure_soc_absolute_error_mean', 'ev', 'performance', 'departure_soc_absolute_error_mean', None, 'ratio'),
            ('ev_departure_tolerance_ratio', 'ev', 'performance', 'departure_tolerance', None, 'ratio'),
            ('ev_charge_total_kwh', 'ev', 'total', 'charge', None, 'kwh'),
            ('ev_v2g_export_total_kwh', 'ev', 'total', 'v2g_export', None, 'kwh'),
            ('ev_connected_soc_gain_total_kwh', 'ev', 'energy_accounting', 'connected_soc_gain', None, 'kwh'),
            ('ev_energy_accounting_shortfall_kwh', 'ev', 'energy_accounting', 'shortfall', None, 'kwh'),
            ('bess_charge_total_kwh', 'battery', 'total', 'charge', None, 'kwh'),
            ('bess_discharge_total_kwh', 'battery', 'total', 'discharge', None, 'kwh'),
            ('bess_throughput_total_kwh', 'battery', 'total', 'throughput', None, 'kwh'),
            ('bess_equivalent_full_cycles', 'battery', 'health', 'equivalent_full_cycles', None, 'count'),
            ('bess_capacity_fade_ratio', 'battery', 'health', 'capacity_fade', None, 'ratio'),
            ('electrical_service_violation_total_kwh', 'electrical_service_phase', 'violations', 'energy_total', None, 'kwh'),
            ('electrical_service_violation_time_step_count', 'electrical_service_phase', 'violations', 'event', None, 'count'),
            ('electrical_service_requested_pressure_total_kwh', 'electrical_service_phase', 'requested_pressure', 'energy_total', None, 'kwh'),
            ('electrical_service_requested_pressure_time_step_count', 'electrical_service_phase', 'requested_pressure', 'event', None, 'count'),
            ('phase_imbalance_ratio_average', 'electrical_service_phase', 'imbalance', 'phase_average', None, 'ratio'),
            ('community_local_import_total_kwh', 'energy_grid', 'community_market', 'local_import', 'total', 'kwh'),
            ('community_local_export_total_kwh', 'energy_grid', 'community_market', 'local_export', 'total', 'kwh'),
            ('community_grid_import_after_local_total_kwh', 'energy_grid', 'community_market', 'grid_import_after_local', 'total', 'kwh'),
            ('community_grid_export_after_local_total_kwh', 'energy_grid', 'community_market', 'grid_export_after_local', 'total', 'kwh'),
            ('community_local_import_daily_average_kwh', 'energy_grid', 'community_market', 'local_import', 'daily_average', 'kwh'),
            ('community_local_export_daily_average_kwh', 'energy_grid', 'community_market', 'local_export', 'daily_average', 'kwh'),
            ('community_grid_import_after_local_daily_average_kwh', 'energy_grid', 'community_market', 'grid_import_after_local', 'daily_average', 'kwh'),
            ('community_grid_export_after_local_daily_average_kwh', 'energy_grid', 'community_market', 'grid_export_after_local', 'daily_average', 'kwh'),
            ('community_settled_cost_total_eur', 'cost', 'community_market', 'settled', 'total', 'eur'),
            ('community_counterfactual_cost_total_eur', 'cost', 'community_market', 'counterfactual', 'total', 'eur'),
            ('community_market_savings_total_eur', 'cost', 'community_market', 'savings', 'total', 'eur'),
            ('community_settled_cost_daily_average_eur', 'cost', 'community_market', 'settled', 'daily_average', 'eur'),
            ('community_counterfactual_cost_daily_average_eur', 'cost', 'community_market', 'counterfactual', 'daily_average', 'eur'),
            ('community_market_savings_daily_average_eur', 'cost', 'community_market', 'savings', 'daily_average', 'eur'),
            ('community_local_share_of_demand', 'solar_self_consumption', 'community_market', 'local_share_of_demand', None, 'ratio'),
            ('community_local_share_of_export', 'solar_self_consumption', 'community_market', 'local_share_of_export', None, 'ratio'),
            ('equity_relative_benefit_percent', 'equity', 'benefit', 'relative', None, 'percent'),
            ('equity_gini_benefit', 'equity', 'distribution', 'gini_benefit', None, 'ratio'),
            ('equity_cr20_benefit', 'equity', 'distribution', 'top20_benefit', None, 'ratio'),
            ('equity_losers_percent', 'equity', 'distribution', 'losers', None, 'percent'),
            ('equity_bpr_asset_poor_over_rich', 'equity', 'distribution', 'bpr_asset_poor_over_rich', None, 'ratio'),
            ('deferrable_appliance_completed_cycles_count', 'deferrable_appliance', 'service', 'completed_cycles', None, 'count'),
            ('deferrable_appliance_missed_cycles_count', 'deferrable_appliance', 'service', 'missed_cycles', None, 'count'),
            ('deferrable_appliance_service_level_ratio', 'deferrable_appliance', 'service', 'service_level', None, 'ratio'),
            ('deferrable_appliance_served_energy_total_kwh', 'deferrable_appliance', 'service', 'served_energy_total', None, 'kwh'),
            ('deferrable_appliance_unserved_energy_total_kwh', 'deferrable_appliance', 'service', 'unserved_energy_total', None, 'kwh'),
            ('deferrable_appliance_average_start_delay_hours', 'deferrable_appliance', 'service', 'average_start_delay', None, 'hours'),
            ('escalator_requested_passengers_total_count', 'escalator', 'service', 'requested_passengers_total', None, 'count'),
            ('escalator_served_passengers_total_count', 'escalator', 'service', 'served_passengers_total', None, 'count'),
            ('escalator_unserved_passengers_total_count', 'escalator', 'service', 'unserved_passengers_total', None, 'count'),
            ('escalator_service_level_ratio', 'escalator', 'service', 'service_level', None, 'ratio'),
            ('escalator_electricity_consumption_total_kwh', 'escalator', 'energy', 'electricity_consumption_total', None, 'kwh'),
            ('escalator_state_changes_count', 'escalator', 'operation', 'state_changes', None, 'count'),
        ]
        for old_name, family, subfamily, metric, variant, unit in extended_map:
            map_from(extended_df, old_name, family, subfamily, metric, variant, unit)

        # Phase peaks have dynamic suffixes (L1/L2/L3) and are conditionally present.
        for _, row in extended_df.iterrows():
            old_name = str(row['cost_function'])
            level = str(row['level'])
            name = str(row['name'])

            if old_name.startswith('phase_import_peak_kw_'):
                phase = old_name.split('phase_import_peak_kw_', 1)[1].lower()
                put(
                    level,
                    name,
                    v2(level, 'electrical_service_phase', 'phase_peaks', f'import_peak_{phase}', None, 'kw'),
                    row['value'],
                )
            elif old_name.startswith('phase_export_peak_kw_'):
                phase = old_name.split('phase_export_peak_kw_', 1)[1].lower()
                put(
                    level,
                    name,
                    v2(level, 'electrical_service_phase', 'phase_peaks', f'export_peak_{phase}', None, 'kw'),
                    row['value'],
                )

        kpi_buildings = self._get_kpi_buildings()
        building_windows = {building.name: self._building_active_window(building) for building in kpi_buildings}
        building_names = [building.name for building in kpi_buildings]

        # Optional community KPIs are district-only.
        if getattr(env, 'community_market_enabled', False):
            district_rows = extended_df[(extended_df['level'] == 'district') & (extended_df['name'] == 'District')]
            if getattr(env, 'community_market_kpi_local_traded_enabled', True):
                local_total = district_rows[district_rows['cost_function'] == 'community_local_import_total_kwh']['value']
                local_daily = district_rows[district_rows['cost_function'] == 'community_local_import_daily_average_kwh']['value']
                if len(local_total) > 0:
                    put(
                        'district',
                        'District',
                        v2('district', 'energy_grid', 'community_market', 'local_traded', 'total', 'kwh'),
                        local_total.iloc[0],
                    )
                if len(local_daily) > 0:
                    put(
                        'district',
                        'District',
                        v2('district', 'energy_grid', 'community_market', 'local_traded', 'daily_average', 'kwh'),
                        local_daily.iloc[0],
                    )

            if getattr(env, 'community_market_kpi_self_consumption_enabled', True):
                local_total = district_rows[district_rows['cost_function'] == 'community_local_import_total_kwh']['value']
                if len(local_total) > 0:
                    local_total_value = self._to_scalar(local_total.iloc[0], 0.0)
                    grid_import_after_local = district_rows[
                        district_rows['cost_function'] == 'community_grid_import_after_local_total_kwh'
                    ]['value']
                    grid_import_after_local_value = (
                        self._to_scalar(grid_import_after_local.iloc[0], 0.0)
                        if len(grid_import_after_local) > 0 else 0.0
                    )
                    total_member_demand = local_total_value + grid_import_after_local_value
                    share = None if total_member_demand <= float(ZERO_DIVISION_PLACEHOLDER) else local_total_value / total_member_demand
                    put(
                        'district',
                        'District',
                        v2('district', 'solar_self_consumption', 'community_market', 'import_share', None, 'ratio'),
                        share,
                    )

        demand_response_service = getattr(env, '_demand_response_service', None)
        if getattr(demand_response_service, 'enabled', False):
            dr_by_building, dr_district = demand_response_service.summarize(building_names)

            def dr_name(level: str, suffix: str) -> str:
                return f'{level}_demand_response_{suffix}'

            dr_metrics = [
                ('events_count', 'demand_response_events_count'),
                ('active_time_step_count', 'demand_response_active_time_step_count'),
                ('requested_total_kwh', 'demand_response_requested_total_kwh'),
                ('delivered_total_kwh', 'demand_response_delivered_total_kwh'),
                ('shortfall_total_kwh', 'demand_response_shortfall_total_kwh'),
                ('compliance_ratio', 'demand_response_compliance_ratio'),
                ('revenue_total_eur', 'demand_response_revenue_total_eur'),
                ('penalty_total_eur', 'demand_response_penalty_total_eur'),
                ('net_revenue_total_eur', 'demand_response_net_revenue_total_eur'),
                ('invalid_baseline_time_step_count', 'demand_response_invalid_baseline_time_step_count'),
            ]

            for suffix, key in dr_metrics:
                put('district', 'District', dr_name('district', suffix), dr_district.get(key))

            for building_name in building_names:
                totals = dr_by_building.get(building_name, {})
                for suffix, key in dr_metrics:
                    put('building', building_name, dr_name('building', suffix), totals.get(key))

        robustness_service = getattr(env, '_robustness_service', None)
        if getattr(robustness_service, 'enabled', False):
            robustness_by_building, robustness_district = robustness_service.summarize(building_names)

            robustness_metrics = [
                'robustness_events_count',
                'robustness_active_time_step_count',
                'robustness_observation_corruption_count',
                'robustness_forecast_corruption_count',
                'robustness_action_corruption_count',
                'robustness_asset_unavailable_time_step_count',
                'robustness_missing_observation_count',
                'robustness_action_dropout_count',
            ]

            for key in robustness_metrics:
                put('district', 'District', f'district_{key}', robustness_district.get(key))

            for building_name in building_names:
                totals = robustness_by_building.get(building_name, {})
                for key in robustness_metrics:
                    put('building', building_name, f'building_{key}', totals.get(key))

        # Export ratio_to_baseline is derived from totals with safe division.
        for building in kpi_buildings:
            control_cond, baseline_cond = self._default_building_conditions(
                building,
                control_condition,
                baseline_condition,
                evaluation_condition_cls=evaluation_condition_cls,
                dynamics_building_cls=dynamics_building_cls,
            )
            t_start, t_end = building_windows.get(building.name, (0, -1))
            net_c = self._slice_window(getattr(building, f'net_electricity_consumption{control_cond.value}'), t_start, t_end)
            net_b = self._slice_window(getattr(building, f'net_electricity_consumption{baseline_cond.value}'), t_start, t_end)
            export_c = self._sum_finite(np.clip(-net_c, 0.0, None))
            export_b = self._sum_finite(np.clip(-net_b, 0.0, None))
            put(
                'building',
                building.name,
                v2('building', 'energy_grid', 'ratio_to_baseline', 'export_total', None, 'ratio'),
                self._safe_div(export_c, export_b),
            )

        env_control_condition = (
            evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
            if control_condition is None else control_condition
        )
        env_baseline_condition = (
            evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
            if baseline_condition is None else baseline_condition
        )
        env_net_c = np.array(getattr(env, f'net_electricity_consumption{env_control_condition.value}'), dtype='float64')
        env_net_b = np.array(getattr(env, f'net_electricity_consumption{env_baseline_condition.value}'), dtype='float64')
        env_export_c = self._sum_finite(np.clip(-env_net_c, 0.0, None))
        env_export_b = self._sum_finite(np.clip(-env_net_b, 0.0, None))
        put(
            'district',
            'District',
            v2('district', 'energy_grid', 'ratio_to_baseline', 'export_total', None, 'ratio'),
            self._safe_div(env_export_c, env_export_b),
        )

        # Cost metrics: market-enabled scenarios settle both control and baseline with same rules.
        control_condition_by_building = {}
        baseline_condition_by_building = {}
        for building in kpi_buildings:
            c_cond, b_cond = self._default_building_conditions(
                building,
                control_condition,
                baseline_condition,
                evaluation_condition_cls=evaluation_condition_cls,
                dynamics_building_cls=dynamics_building_cls,
            )
            control_condition_by_building[building.name] = c_cond
            baseline_condition_by_building[building.name] = b_cond

        control_cost_totals: Dict[str, float] = {}
        baseline_cost_totals: Dict[str, float] = {}
        district_control_total = 0.0
        district_baseline_total = 0.0

        if getattr(env, 'community_market_enabled', False):
            control_cost_totals, district_control_total = self._condition_settled_cost_totals(
                condition_by_building=control_condition_by_building,
            )
            baseline_cost_totals, district_baseline_total = self._condition_settled_cost_totals(
                condition_by_building=baseline_condition_by_building,
            )
        else:
            # Read mapped values from records when market is disabled.
            for building_name in building_names:
                key_control = ('building', building_name, v2('building', 'cost', 'total', 'control', None, 'eur'))
                key_baseline = ('building', building_name, v2('building', 'cost', 'total', 'baseline', None, 'eur'))
                control_cost_totals[building_name] = self._to_scalar(records.get(key_control, {}).get('value'), 0.0)
                baseline_cost_totals[building_name] = self._to_scalar(records.get(key_baseline, {}).get('value'), 0.0)

            district_control_total = self._to_scalar(
                records.get(('district', 'District', v2('district', 'cost', 'total', 'control', None, 'eur')), {}).get('value'),
                0.0,
            )
            district_baseline_total = self._to_scalar(
                records.get(('district', 'District', v2('district', 'cost', 'total', 'baseline', None, 'eur')), {}).get('value'),
                0.0,
            )

        for building_name in building_names:
            t_start, t_end = building_windows.get(building_name, (0, -1))
            building_days = self._window_days(t_start, t_end, env.seconds_per_time_step)
            control_total = self._to_scalar(control_cost_totals.get(building_name), 0.0)
            baseline_total = self._to_scalar(baseline_cost_totals.get(building_name), 0.0)
            delta_total = control_total - baseline_total
            put('building', building_name, v2('building', 'cost', 'total', 'control', None, 'eur'), control_total)
            put('building', building_name, v2('building', 'cost', 'total', 'baseline', None, 'eur'), baseline_total)
            put('building', building_name, v2('building', 'cost', 'total', 'delta', None, 'eur'), delta_total)
            put('building', building_name, v2('building', 'cost', 'daily_average', 'control', None, 'eur'), self._daily_average(control_total, building_days))
            put('building', building_name, v2('building', 'cost', 'daily_average', 'baseline', None, 'eur'), self._daily_average(baseline_total, building_days))
            put('building', building_name, v2('building', 'cost', 'daily_average', 'delta', None, 'eur'), self._daily_average(delta_total, building_days))
            put('building', building_name, v2('building', 'cost', 'ratio_to_baseline', 'total', None, 'ratio'), self._safe_div(control_total, baseline_total))

        district_delta_total = district_control_total - district_baseline_total
        put('district', 'District', v2('district', 'cost', 'total', 'control', None, 'eur'), district_control_total)
        put('district', 'District', v2('district', 'cost', 'total', 'baseline', None, 'eur'), district_baseline_total)
        put('district', 'District', v2('district', 'cost', 'total', 'delta', None, 'eur'), district_delta_total)
        put('district', 'District', v2('district', 'cost', 'daily_average', 'control', None, 'eur'), self._daily_average(district_control_total, simulated_days))
        put('district', 'District', v2('district', 'cost', 'daily_average', 'baseline', None, 'eur'), self._daily_average(district_baseline_total, simulated_days))
        put('district', 'District', v2('district', 'cost', 'daily_average', 'delta', None, 'eur'), self._daily_average(district_delta_total, simulated_days))
        put('district', 'District', v2('district', 'cost', 'ratio_to_baseline', 'total', None, 'ratio'), self._safe_div(district_control_total, district_baseline_total))

        # Equity metrics are recomputed from cost totals for consistency.
        equity_valid_benefits: Dict[str, float] = {}
        groups = {building.name: getattr(building, 'equity_group', None) for building in kpi_buildings}
        for building_name in building_names:
            benefit = self._equity_relative_benefit_percent(
                control_cost_totals.get(building_name, 0.0),
                baseline_cost_totals.get(building_name, 0.0),
            )
            put(
                'building',
                building_name,
                v2('building', 'equity', 'benefit', 'relative', None, 'percent'),
                benefit,
            )
            if benefit is not None:
                equity_valid_benefits[building_name] = float(benefit)

        equity_distribution = self._equity_distribution_metrics(np.array(list(equity_valid_benefits.values()), dtype='float64'))
        non_negative_benefits = {name: max(value, 0.0) for name, value in equity_valid_benefits.items()}
        has_complete_manual_groups = all(groups.get(name) in {'asset_rich', 'asset_poor'} for name in building_names)
        equity_bpr = self._equity_bpr(non_negative_benefits, groups) if has_complete_manual_groups else None
        put('district', 'District', v2('district', 'equity', 'distribution', 'gini_benefit', None, 'ratio'), equity_distribution['equity_gini_benefit'])
        put('district', 'District', v2('district', 'equity', 'distribution', 'top20_benefit', None, 'ratio'), equity_distribution['equity_cr20_benefit'])
        put('district', 'District', v2('district', 'equity', 'distribution', 'losers', None, 'percent'), equity_distribution['equity_losers_percent'])
        put('district', 'District', v2('district', 'equity', 'distribution', 'bpr_asset_poor_over_rich', None, 'ratio'), equity_bpr)

        if include_business_as_usual and not bool(getattr(env, '_business_as_usual_sidecar', False)):
            self._add_business_as_usual_records(
                records,
                v2,
                simulated_days,
                evaluation_condition_cls=evaluation_condition_cls,
                dynamics_building_cls=dynamics_building_cls,
            )

        output = pd.DataFrame(list(records.values()))
        if output.empty:
            return pd.DataFrame(columns=['cost_function', 'value', 'name', 'level'])

        if getattr(env, 'topology_mode', 'static') == 'dynamic':
            lifecycle = getattr(env, 'topology_member_lifecycle', {}) or {}
            existing_buildings = set(output[output['level'] == 'building']['name'].astype(str).tolist())
            lifecycle_rows = []

            for member_id in lifecycle.keys():
                if member_id not in existing_buildings:
                    lifecycle_rows.append(
                        {
                            'cost_function': 'topology_lifecycle_presence',
                            'value': np.nan,
                            'name': member_id,
                            'level': 'building',
                        }
                    )

            if lifecycle_rows:
                output = pd.concat([output, pd.DataFrame(lifecycle_rows)], ignore_index=True)

            output['topology_born_at'] = np.nan
            output['topology_removed_at'] = np.nan
            output['topology_active'] = np.nan
            for member_id, state in lifecycle.items():
                mask = (output['level'] == 'building') & (output['name'] == member_id)
                output.loc[mask, 'topology_born_at'] = state.get('born_at')
                output.loc[mask, 'topology_removed_at'] = state.get('removed_at')
                output.loc[mask, 'topology_active'] = 1.0 if state.get('active', False) else 0.0

        output = output.sort_values(['level', 'name', 'cost_function']).reset_index(drop=True)
        return output
