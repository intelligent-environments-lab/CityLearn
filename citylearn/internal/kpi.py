from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from citylearn.cost_function import CostFunction
from citylearn.data import EnergySimulation, ZERO_DIVISION_PLACEHOLDER

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class CityLearnKPIService:
    """Internal KPI/evaluation service for `CityLearnEnv`."""

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

    def _compute_ev_metrics(self, building) -> Dict[str, float]:
        t_final = int(max(building.time_step, 0))
        departures_total = 0
        departures_met = 0
        departure_deficit_sum = 0.0
        charge_total_kwh = 0.0
        v2g_export_total_kwh = 0.0

        for charger in building.electric_vehicle_chargers or []:
            consumption = np.array(charger.electricity_consumption[0:t_final + 1], dtype='float64')
            charge_total_kwh += float(np.clip(consumption, 0.0, None).sum())
            v2g_export_total_kwh += float(np.clip(-consumption, 0.0, None).sum())

            sim = charger.charger_simulation
            states = np.array(sim.electric_vehicle_charger_state, dtype='float64')
            required_soc = np.array(sim.electric_vehicle_required_soc_departure, dtype='float64')
            history_limit = min(t_final, len(states) - 2, len(required_soc) - 1, len(charger.past_connected_evs) - 1)

            if history_limit < 0:
                continue

            for t in range(history_limit + 1):
                current_state = states[t]
                next_state = states[t + 1]

                if current_state != 1 or next_state == 1:
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

                departures_total += 1
                deficit = max(target_soc - actual_soc, 0.0)
                departure_deficit_sum += deficit
                if deficit <= 1e-6:
                    departures_met += 1

        success_rate = None if departures_total == 0 else departures_met / departures_total
        deficit_mean = None if departures_total == 0 else departure_deficit_sum / departures_total

        return {
            'departures_total': float(departures_total),
            'departures_met': float(departures_met),
            'departure_deficit_sum': float(departure_deficit_sum),
            'ev_departure_success_rate': success_rate,
            'ev_departure_soc_deficit_mean': deficit_mean,
            'ev_charge_total_kwh': float(charge_total_kwh),
            'ev_v2g_export_total_kwh': float(v2g_export_total_kwh),
        }

    def _compute_bess_metrics(self, building) -> Dict[str, float]:
        t_final = int(max(building.time_step, 0))
        storage = building.electrical_storage
        storage_series = np.array(building.electrical_storage_electricity_consumption[0:t_final + 1], dtype='float64')
        charge_total = float(np.clip(storage_series, 0.0, None).sum())
        discharge_total = float(np.clip(-storage_series, 0.0, None).sum())
        throughput_total = charge_total + discharge_total

        capacity = self._to_scalar(getattr(storage, 'capacity', 0.0), 0.0)
        degraded_capacity = self._to_scalar(getattr(storage, 'degraded_capacity', capacity), capacity)
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

    def _compute_pv_metrics(self, building) -> Dict[str, float]:
        t_final = int(max(building.time_step, 0))
        solar = np.array(building.solar_generation[0:t_final + 1], dtype='float64')
        net = np.array(building.net_electricity_consumption[0:t_final + 1], dtype='float64')

        generation = np.clip(-solar, 0.0, None)
        export = np.clip(-net, 0.0, None)
        pv_generation_total = float(generation.sum())
        pv_export_total = float(np.minimum(generation, export).sum())
        self_consumption_ratio = None if pv_generation_total <= 0.0 else (pv_generation_total - pv_export_total) / pv_generation_total

        return {
            'pv_generation_total_kwh': pv_generation_total,
            'pv_export_total_kwh': pv_export_total,
            'pv_self_consumption_ratio': self_consumption_ratio,
        }

    def _compute_phase_metrics(self, building) -> Dict[str, object]:
        if not getattr(building, '_electrical_service_enabled', False):
            return {
                'electrical_service_violation_total_kwh': 0.0,
                'electrical_service_violation_time_step_count': 0.0,
                'phase_imbalance_ratio_average': None,
                'phase_import_peak_kw': {},
                'phase_export_peak_kw': {},
                '_imbalance_sum': 0.0,
                '_imbalance_count': 0.0,
            }

        t_final = int(max(building.time_step, 0))
        violation_history = np.array(getattr(building, '_charging_constraint_violation_history', [0.0]), dtype='float64')[0:t_final + 1]
        violation_total = float(np.clip(violation_history, 0.0, None).sum())
        violation_count = float(np.count_nonzero(violation_history > 1e-9))

        phase_history = getattr(building, '_charging_phase_power_history_kw', {}) or {}
        phase_import_peak = {}
        phase_export_peak = {}

        for phase_name, values in phase_history.items():
            series = np.array(values[0:t_final + 1], dtype='float64')
            phase_import_peak[phase_name] = float(np.clip(series, 0.0, None).max(initial=0.0))
            phase_export_peak[phase_name] = float(np.clip(-series, 0.0, None).max(initial=0.0))

        imbalance_sum = 0.0
        imbalance_count = 0.0
        imbalance_average = None

        if getattr(building, '_electrical_service_mode', 'single_phase') == 'three_phase':
            names = [n for n in ['L1', 'L2', 'L3'] if n in phase_history]
            if len(names) == 3:
                stacked = np.stack([np.array(phase_history[n][0:t_final + 1], dtype='float64') for n in names], axis=1)
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
        ev_deficit_sum = 0.0
        ev_charge_total = 0.0
        ev_v2g_total = 0.0

        bess_charge_total = 0.0
        bess_discharge_total = 0.0
        bess_throughput_total = 0.0
        bess_capacity_total = 0.0
        bess_capacity_loss_total = 0.0

        pv_generation_total = 0.0
        pv_export_total = 0.0

        phase_violation_total = 0.0
        phase_violation_count = 0.0
        phase_imbalance_sum = 0.0
        phase_imbalance_count = 0.0

        building_names = [building.name for building in env.buildings]
        equity_group_by_building = {building.name: getattr(building, 'equity_group', None) for building in env.buildings}
        equity_relative_benefit_by_building: Dict[str, Optional[float]] = {}
        equity_valid_benefits: Dict[str, float] = {}

        for building in env.buildings:
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

            discomfort_kwargs = {
                'indoor_dry_bulb_temperature': building.indoor_dry_bulb_temperature,
                'dry_bulb_temperature_cooling_set_point': building.indoor_dry_bulb_temperature_cooling_set_point,
                'dry_bulb_temperature_heating_set_point': building.indoor_dry_bulb_temperature_heating_set_point,
                'band': building.comfort_band if comfort_band is None else comfort_band,
                'occupant_count': building.occupant_count,
            }
            unmet, cold, hot, \
                cold_minimum_delta, cold_maximum_delta, cold_average_delta, \
                hot_minimum_delta, hot_maximum_delta, hot_average_delta = CostFunction.discomfort(**discomfort_kwargs)
            expected_energy = building.cooling_demand + building.heating_demand + building.dhw_demand + building.non_shiftable_load
            served_energy = building.energy_from_cooling_device + building.energy_from_cooling_storage \
                + building.energy_from_heating_device + building.energy_from_heating_storage \
                + building.energy_from_dhw_device + building.energy_from_dhw_storage \
                + building.energy_to_non_shiftable_load
            ec_c = CostFunction.electricity_consumption(get_net_electricity_consumption(building, building_control_condition))[-1]
            ec_b = CostFunction.electricity_consumption(get_net_electricity_consumption(building, building_baseline_condition))[-1]
            zne_c = CostFunction.zero_net_energy(get_net_electricity_consumption(building, building_control_condition))[-1]
            zne_b = CostFunction.zero_net_energy(get_net_electricity_consumption(building, building_baseline_condition))[-1]
            ce_c = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(building, building_control_condition))[-1]
            ce_b = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(building, building_baseline_condition))[-1] if sum(building.carbon_intensity.carbon_intensity) != 0 else 0
            control_cost_series = get_net_electricity_consumption_cost(building, building_control_condition)
            baseline_cost_series = get_net_electricity_consumption_cost(building, building_baseline_condition)
            cost_c_legacy = CostFunction.cost(control_cost_series)[-1]
            cost_b_legacy = CostFunction.cost(baseline_cost_series)[-1]
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
                'value': CostFunction.one_minus_thermal_resilience(power_outage=building.power_outage_signal, **discomfort_kwargs)[-1],
            }, {
                'cost_function': 'power_outage_normalized_unserved_energy_total',
                'value': CostFunction.normalized_unserved_energy(expected_energy, served_energy, power_outage=building.power_outage_signal)[-1],
            }, {
                'cost_function': 'annual_normalized_unserved_energy_total',
                'value': CostFunction.normalized_unserved_energy(expected_energy, served_energy)[-1],
            }])
            legacy_building_frame['name'] = building.name
            legacy_building_frames.append(legacy_building_frame)

            extended_building_rows.extend([
                self._metric('electricity_consumption_control_total_kwh', ec_c, building.name, 'building'),
                self._metric('electricity_consumption_baseline_total_kwh', ec_b, building.name, 'building'),
                self._metric('electricity_consumption_delta_total_kwh', ec_c - ec_b, building.name, 'building'),
                self._metric('electricity_consumption_control_daily_average_kwh', self._daily_average(ec_c, simulated_days), building.name, 'building'),
                self._metric('electricity_consumption_baseline_daily_average_kwh', self._daily_average(ec_b, simulated_days), building.name, 'building'),
                self._metric('electricity_consumption_delta_daily_average_kwh', self._daily_average(ec_c - ec_b, simulated_days), building.name, 'building'),
                self._metric('zero_net_energy_control_total_kwh', zne_c, building.name, 'building'),
                self._metric('zero_net_energy_baseline_total_kwh', zne_b, building.name, 'building'),
                self._metric('zero_net_energy_delta_total_kwh', zne_c - zne_b, building.name, 'building'),
                self._metric('zero_net_energy_control_daily_average_kwh', self._daily_average(zne_c, simulated_days), building.name, 'building'),
                self._metric('zero_net_energy_baseline_daily_average_kwh', self._daily_average(zne_b, simulated_days), building.name, 'building'),
                self._metric('zero_net_energy_delta_daily_average_kwh', self._daily_average(zne_c - zne_b, simulated_days), building.name, 'building'),
                self._metric('carbon_emissions_control_total_kgco2', ce_c, building.name, 'building'),
                self._metric('carbon_emissions_baseline_total_kgco2', ce_b, building.name, 'building'),
                self._metric('carbon_emissions_delta_total_kgco2', ce_c - ce_b, building.name, 'building'),
                self._metric('carbon_emissions_control_daily_average_kgco2', self._daily_average(ce_c, simulated_days), building.name, 'building'),
                self._metric('carbon_emissions_baseline_daily_average_kgco2', self._daily_average(ce_b, simulated_days), building.name, 'building'),
                self._metric('carbon_emissions_delta_daily_average_kgco2', self._daily_average(ce_c - ce_b, simulated_days), building.name, 'building'),
                self._metric('cost_control_total_eur', cost_c_raw, building.name, 'building'),
                self._metric('cost_baseline_total_eur', cost_b_raw, building.name, 'building'),
                self._metric('cost_delta_total_eur', cost_c_raw - cost_b_raw, building.name, 'building'),
                self._metric('cost_control_daily_average_eur', self._daily_average(cost_c_raw, simulated_days), building.name, 'building'),
                self._metric('cost_baseline_daily_average_eur', self._daily_average(cost_b_raw, simulated_days), building.name, 'building'),
                self._metric('cost_delta_daily_average_eur', self._daily_average(cost_c_raw - cost_b_raw, simulated_days), building.name, 'building'),
                self._metric('equity_relative_benefit_percent', equity_benefit, building.name, 'building'),
            ])

            ev_metrics = self._compute_ev_metrics(building)
            extended_building_rows.extend([
                self._metric('ev_departure_success_rate', ev_metrics['ev_departure_success_rate'], building.name, 'building'),
                self._metric('ev_departure_soc_deficit_mean', ev_metrics['ev_departure_soc_deficit_mean'], building.name, 'building'),
                self._metric('ev_charge_total_kwh', ev_metrics['ev_charge_total_kwh'], building.name, 'building'),
                self._metric('ev_v2g_export_total_kwh', ev_metrics['ev_v2g_export_total_kwh'], building.name, 'building'),
            ])
            ev_departures_total += ev_metrics['departures_total']
            ev_departures_met += ev_metrics['departures_met']
            ev_deficit_sum += ev_metrics['departure_deficit_sum']
            ev_charge_total += ev_metrics['ev_charge_total_kwh']
            ev_v2g_total += ev_metrics['ev_v2g_export_total_kwh']

            bess_metrics = self._compute_bess_metrics(building)
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

            pv_metrics = self._compute_pv_metrics(building)
            extended_building_rows.extend([
                self._metric('pv_generation_total_kwh', pv_metrics['pv_generation_total_kwh'], building.name, 'building'),
                self._metric('pv_export_total_kwh', pv_metrics['pv_export_total_kwh'], building.name, 'building'),
                self._metric('pv_generation_daily_average_kwh', self._daily_average(pv_metrics['pv_generation_total_kwh'], simulated_days), building.name, 'building'),
                self._metric('pv_export_daily_average_kwh', self._daily_average(pv_metrics['pv_export_total_kwh'], simulated_days), building.name, 'building'),
                self._metric('pv_self_consumption_ratio', pv_metrics['pv_self_consumption_ratio'], building.name, 'building'),
            ])
            pv_generation_total += pv_metrics['pv_generation_total_kwh']
            pv_export_total += pv_metrics['pv_export_total_kwh']

            phase_metrics = self._compute_phase_metrics(building)
            extended_building_rows.extend([
                self._metric('electrical_service_violation_total_kwh', phase_metrics['electrical_service_violation_total_kwh'], building.name, 'building'),
                self._metric('electrical_service_violation_time_step_count', phase_metrics['electrical_service_violation_time_step_count'], building.name, 'building'),
                self._metric('phase_imbalance_ratio_average', phase_metrics['phase_imbalance_ratio_average'], building.name, 'building'),
            ])
            for phase_name, value in phase_metrics['phase_import_peak_kw'].items():
                extended_building_rows.append(self._metric(f'phase_import_peak_kw_{phase_name}', value, building.name, 'building'))
            for phase_name, value in phase_metrics['phase_export_peak_kw'].items():
                extended_building_rows.append(self._metric(f'phase_export_peak_kw_{phase_name}', value, building.name, 'building'))

            phase_violation_total += phase_metrics['electrical_service_violation_total_kwh']
            phase_violation_count += phase_metrics['electrical_service_violation_time_step_count']
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

        ramp_c = CostFunction.ramping(get_net_electricity_consumption(env, env_control_condition))[-1]
        ramp_b = CostFunction.ramping(get_net_electricity_consumption(env, env_baseline_condition))[-1]
        dlf_daily_c = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, env_control_condition), window=daily_steps)[-1]
        dlf_daily_b = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, env_baseline_condition), window=daily_steps)[-1]
        dlf_monthly_c = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, env_control_condition), window=monthly_steps)[-1]
        dlf_monthly_b = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, env_baseline_condition), window=monthly_steps)[-1]
        peak_daily_c = CostFunction.peak(get_net_electricity_consumption(env, env_control_condition), window=daily_steps)[-1]
        peak_daily_b = CostFunction.peak(get_net_electricity_consumption(env, env_baseline_condition), window=daily_steps)[-1]
        peak_all_c = CostFunction.peak(get_net_electricity_consumption(env, env_control_condition), window=env.time_steps)[-1]
        peak_all_b = CostFunction.peak(get_net_electricity_consumption(env, env_baseline_condition), window=env.time_steps)[-1]

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
        ec_c_env = CostFunction.electricity_consumption(get_net_electricity_consumption(env, env_control_condition))[-1]
        ec_b_env = CostFunction.electricity_consumption(get_net_electricity_consumption(env, env_baseline_condition))[-1]
        zne_c_env = CostFunction.zero_net_energy(get_net_electricity_consumption(env, env_control_condition))[-1]
        zne_b_env = CostFunction.zero_net_energy(get_net_electricity_consumption(env, env_baseline_condition))[-1]
        ce_c_env = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(env, env_control_condition))[-1]
        ce_b_env = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(env, env_baseline_condition))[-1]
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
        ev_deficit_mean = None if ev_departures_total <= 0.0 else ev_deficit_sum / ev_departures_total
        extended_district_rows.extend([
            self._metric('ev_departure_success_rate', ev_success_rate, 'District', 'district'),
            self._metric('ev_departure_soc_deficit_mean', ev_deficit_mean, 'District', 'district'),
            self._metric('ev_charge_total_kwh', ev_charge_total, 'District', 'district'),
            self._metric('ev_v2g_export_total_kwh', ev_v2g_total, 'District', 'district'),
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

        district_pv_ratio = None if pv_generation_total <= 0.0 else (pv_generation_total - pv_export_total) / pv_generation_total
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
            self._metric('phase_imbalance_ratio_average', district_phase_imbalance, 'District', 'district'),
        ])

        phase_union = ['L1', 'L2', 'L3']
        for phase_name in phase_union:
            phase_series = None
            for building in env.buildings:
                history_map = getattr(building, '_charging_phase_power_history_kw', {}) or {}
                if phase_name not in history_map:
                    continue
                t_final = int(max(building.time_step, 0))
                values = np.array(history_map[phase_name][0:t_final + 1], dtype='float64')
                if phase_series is None:
                    phase_series = np.zeros_like(values)
                size = min(len(phase_series), len(values))
                phase_series[:size] += values[:size]

            if phase_series is None:
                continue

            extended_district_rows.append(
                self._metric(f'phase_import_peak_kw_{phase_name}', float(np.clip(phase_series, 0.0, None).max(initial=0.0)), 'District', 'district')
            )
            extended_district_rows.append(
                self._metric(f'phase_export_peak_kw_{phase_name}', float(np.clip(-phase_series, 0.0, None).max(initial=0.0)), 'District', 'district')
            )

        market_by_building, market_district = self._collect_market_totals(building_names)

        for building_name in building_names:
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
                self._metric('community_local_import_daily_average_kwh', self._daily_average(local_import, simulated_days), building_name, 'building'),
                self._metric('community_local_export_daily_average_kwh', self._daily_average(local_export, simulated_days), building_name, 'building'),
                self._metric('community_grid_import_after_local_daily_average_kwh', self._daily_average(grid_import, simulated_days), building_name, 'building'),
                self._metric('community_grid_export_after_local_daily_average_kwh', self._daily_average(grid_export, simulated_days), building_name, 'building'),
                self._metric('community_settled_cost_total_eur', totals.get('community_settled_cost_total_eur', 0.0), building_name, 'building'),
                self._metric('community_counterfactual_cost_total_eur', totals.get('community_counterfactual_cost_total_eur', 0.0), building_name, 'building'),
                self._metric('community_market_savings_total_eur', totals.get('community_market_savings_total_eur', 0.0), building_name, 'building'),
                self._metric('community_settled_cost_daily_average_eur', self._daily_average(totals.get('community_settled_cost_total_eur', 0.0), simulated_days), building_name, 'building'),
                self._metric('community_counterfactual_cost_daily_average_eur', self._daily_average(totals.get('community_counterfactual_cost_total_eur', 0.0), simulated_days), building_name, 'building'),
                self._metric('community_market_savings_daily_average_eur', self._daily_average(totals.get('community_market_savings_total_eur', 0.0), simulated_days), building_name, 'building'),
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

        extended_building = pd.DataFrame(extended_building_rows)
        extended_district = pd.DataFrame(extended_district_rows)

        cost_functions = pd.concat([legacy_cost_functions, extended_district, extended_building], ignore_index=True, sort=False)

        return cost_functions
