from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from citylearn.cost_function import CostFunction
from citylearn.data import EnergySimulation

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class CityLearnKPIService:
    """Internal KPI/evaluation service for `CityLearnEnv`."""

    def __init__(self, env: "CityLearnEnv"):
        self.env = env

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

        def _safe_div(control_value: float, baseline_value: float):
            try:
                c = control_value
                b = baseline_value

                def _coerce(x):
                    try:
                        v = float(x)
                        return v if np.isfinite(v) else 0.0
                    except Exception:
                        return 0.0

                c = _coerce(c)
                b = _coerce(b)
                if b == 0.0:
                    return 1.0 if c == 0.0 else None
                return c / b
            except Exception:
                return None

        comfort_band = EnergySimulation.DEFUALT_COMFORT_BAND if comfort_band is None else comfort_band
        building_level = []

        for building in env.buildings:
            if isinstance(building, dynamics_building_cls):
                control_condition = evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV if control_condition is None else control_condition
                baseline_condition = evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV if baseline_condition is None else baseline_condition
            else:
                control_condition = evaluation_condition_cls.WITH_STORAGE_AND_PV if control_condition is None else control_condition
                baseline_condition = evaluation_condition_cls.WITHOUT_STORAGE_BUT_WITH_PV if baseline_condition is None else baseline_condition

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
            ec_c = CostFunction.electricity_consumption(get_net_electricity_consumption(building, control_condition))[-1]
            ec_b = CostFunction.electricity_consumption(get_net_electricity_consumption(building, baseline_condition))[-1]
            zne_c = CostFunction.zero_net_energy(get_net_electricity_consumption(building, control_condition))[-1]
            zne_b = CostFunction.zero_net_energy(get_net_electricity_consumption(building, baseline_condition))[-1]
            ce_c = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(building, control_condition))[-1]
            ce_b = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(building, baseline_condition))[-1] if sum(building.carbon_intensity.carbon_intensity) != 0 else 0
            cost_c = CostFunction.cost(get_net_electricity_consumption_cost(building, control_condition))[-1]
            cost_b = CostFunction.cost(get_net_electricity_consumption_cost(building, baseline_condition))[-1] if sum(building.pricing.electricity_pricing) != 0 else 0

            building_level_ = pd.DataFrame([{
                'cost_function': 'electricity_consumption_total',
                'value': _safe_div(ec_c, ec_b),
            }, {
                'cost_function': 'zero_net_energy',
                'value': _safe_div(zne_c, zne_b),
            }, {
                'cost_function': 'carbon_emissions_total',
                'value': _safe_div(ce_c, ce_b),
            }, {
                'cost_function': 'cost_total',
                'value': _safe_div(cost_c, cost_b),
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
            building_level_['name'] = building.name
            building_level.append(building_level_)

        building_level = pd.concat(building_level, ignore_index=True)
        building_level['level'] = 'building'

        control_condition = evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV if control_condition is None else control_condition
        baseline_condition = evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV if baseline_condition is None else baseline_condition

        ramp_c = CostFunction.ramping(get_net_electricity_consumption(env, control_condition))[-1]
        ramp_b = CostFunction.ramping(get_net_electricity_consumption(env, baseline_condition))[-1]
        dlf24_c = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, control_condition), window=24)[-1]
        dlf24_b = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, baseline_condition), window=24)[-1]
        dlf730_c = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, control_condition), window=730)[-1]
        dlf730_b = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, baseline_condition), window=730)[-1]
        peak24_c = CostFunction.peak(get_net_electricity_consumption(env, control_condition), window=24)[-1]
        peak24_b = CostFunction.peak(get_net_electricity_consumption(env, baseline_condition), window=24)[-1]
        peak_all_c = CostFunction.peak(get_net_electricity_consumption(env, control_condition), window=env.time_steps)[-1]
        peak_all_b = CostFunction.peak(get_net_electricity_consumption(env, baseline_condition), window=env.time_steps)[-1]

        district_level = pd.DataFrame([{
            'cost_function': 'ramping_average',
            'value': _safe_div(ramp_c, ramp_b),
        }, {
            'cost_function': 'daily_one_minus_load_factor_average',
            'value': _safe_div(dlf24_c, dlf24_b),
        }, {
            'cost_function': 'monthly_one_minus_load_factor_average',
            'value': _safe_div(dlf730_c, dlf730_b),
        }, {
            'cost_function': 'daily_peak_average',
            'value': _safe_div(peak24_c, peak24_b),
        }, {
            'cost_function': 'all_time_peak_average',
            'value': _safe_div(peak_all_c, peak_all_b),
        }])

        district_level = pd.concat([district_level, building_level], ignore_index=True, sort=False)
        district_level = district_level.groupby(['cost_function'])[['value']].mean().reset_index()
        district_level['name'] = 'District'
        district_level['level'] = 'district'
        cost_functions = pd.concat([district_level, building_level], ignore_index=True, sort=False)

        return cost_functions
