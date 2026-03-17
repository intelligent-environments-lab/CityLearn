from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Union

import numpy as np

from citylearn.base import Environment
from citylearn.data import ChargerSimulation

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class CityLearnRuntimeService:
    """Internal runtime orchestration for `CityLearnEnv`."""

    def __init__(self, env: "CityLearnEnv"):
        self.env = env

    @staticmethod
    def _ev_unconnected_drift_std(seconds_per_time_step: float) -> float:
        """Return per-step drift std scaled by physical step duration."""

        seconds = max(float(seconds_per_time_step), 1.0)
        step_hours = seconds / 3600.0
        return 0.2 * np.sqrt(step_hours)

    def step(self, actions: List[List[float]]):
        """Apply actions, update env variables/reward, then advance time."""

        env = self.env

        if env.terminated or env.truncated:
            raise RuntimeError('Episode has already terminated/truncated. Call reset() before calling step() again.')

        env._observations_cache = None
        env._observations_cache_time_step = -1
        actions = self.parse_actions(actions)

        for building, building_actions in zip(env.buildings, actions):
            building.apply_actions(**building_actions)

        self.update_variables()

        if env.debug_timing:
            import time
            building_observations_retrieval_start = time.perf_counter()

        reward_observations = [
            b.observations(include_all=True, normalize=False, periodic_normalization=False) for b in env.buildings
        ]
        if env.debug_timing:
            building_observations_retrieval_end = time.perf_counter()

        reward = env.reward_function.calculate(observations=reward_observations)
        env.rewards.append(reward)

        partial_render_time = self.next_time_step()
        end_export_time = 0.0
        env._maybe_log_periodic_metrics()

        if env.terminated:
            rewards = np.array(env.rewards[1:], dtype='float32')
            env.episode_rewards.append({
                'min': rewards.min(axis=0).tolist(),
                'max': rewards.max(axis=0).tolist(),
                'sum': rewards.sum(axis=0).tolist(),
                'mean': rewards.mean(axis=0).tolist(),
            })
            if env.render_mode == 'end' and env.render_enabled:
                final_index = min(env.time_steps - 1, env.time_step - 1) if env.time_step > 0 else 0
                if env.debug_timing:
                    import time
                    export_start = time.perf_counter()
                env._export_episode_render_data(final_index)
                if env.debug_timing:
                    end_export_time = time.perf_counter() - export_start

            if env.export_kpis_on_episode_end and not env._final_kpis_exported:
                env.export_final_kpis()

        next_observations = env.observations
        info = dict(env.get_info())
        if env.debug_timing:
            info['building_observations_retrieval_time'] = building_observations_retrieval_end - building_observations_retrieval_start
            info['partial_render_time'] = partial_render_time
            info['end_export_time'] = end_export_time

        return next_observations, reward, env.terminated, env.truncated, info

    def parse_actions(self, actions: List[List[float]]) -> List[Mapping[str, float]]:
        """Return mapping of action name to action value for each building."""

        env = self.env

        building_actions = []
        cache = getattr(env, '_active_actions_cache', None)
        cached_expected = getattr(env, '_expected_central_action_count', None)
        current_actions = [list(b.active_actions) for b in env.buildings]
        current_expected = sum(len(v) for v in current_actions)

        if cache is None or cached_expected != current_expected or cache != current_actions:
            env._refresh_action_cache()

        def _is_scalar(value: Any) -> bool:
            return bool(np.isscalar(value))

        def _to_vector(value: Any, *, context: str) -> List[float]:
            if isinstance(value, np.ndarray):
                array = np.asarray(value)
                if array.ndim == 1:
                    return array.tolist()
                if array.ndim == 2 and array.shape[0] == 1:
                    return array[0].tolist()
                raise AssertionError(f'{context} must be a 1D action vector.')

            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return []
                if all(_is_scalar(v) for v in value):
                    return list(value)
                if len(value) == 1:
                    inner = value[0]
                    if isinstance(inner, (list, tuple, np.ndarray)):
                        return _to_vector(inner, context=context)
                raise AssertionError(f'{context} must be a 1D action vector.')

            raise AssertionError(f'{context} must be a 1D action vector.')

        if env.central_agent:
            actions = _to_vector(actions, context='central_agent actions')
            number_of_actions = len(actions)
            expected_number_of_actions = env._expected_central_action_count
            assert number_of_actions == expected_number_of_actions, \
                f'Expected {expected_number_of_actions} actions but {number_of_actions} were parsed to env.step.'

            for building in env.buildings:
                size = building.action_space.shape[0]
                building_actions.append(actions[0:size])
                actions = actions[size:]

        else:
            if isinstance(actions, np.ndarray):
                array = np.asarray(actions)
                if array.ndim == 2:
                    building_actions = [row.tolist() for row in array]
                else:
                    raise AssertionError(
                        'Expected one action vector per building when central_agent=False.'
                    )
            elif isinstance(actions, (list, tuple)):
                building_actions = []
                for idx, action_vector in enumerate(actions):
                    if isinstance(action_vector, (list, tuple, np.ndarray)):
                        building_actions.append(_to_vector(action_vector, context=f'building action vector at index {idx}'))
                    else:
                        raise AssertionError(
                            'Expected one action vector per building when central_agent=False.'
                        )
            else:
                raise AssertionError('Expected one action vector per building when central_agent=False.')

            number_of_building_actions = len(building_actions)
            expected_building_actions = len(env.buildings)
            assert number_of_building_actions == expected_building_actions, \
                f'Expected {expected_building_actions} building action vectors but {number_of_building_actions} were provided.'

        for building, building_action in zip(env.buildings, building_actions):
            number_of_actions = len(building_action)
            expected_number_of_actions = building.action_space.shape[0]
            assert number_of_actions == expected_number_of_actions, \
                f'Expected {expected_number_of_actions} for {building.name} but {number_of_actions} actions were provided.'

        active_actions = env._active_actions_cache
        parsed_actions = []

        for i, _building in enumerate(env.buildings):
            action_dict = {}
            electric_vehicle_actions = {}
            washing_machine_actions = {}

            for action_name, action in zip(active_actions[i], building_actions[i]):
                if 'electric_vehicle_storage' in action_name:
                    charger_id = action_name.replace('electric_vehicle_storage_', '')
                    electric_vehicle_actions[charger_id] = action
                elif 'washing_machine' in action_name:
                    washing_machine_actions[action_name] = action
                else:
                    action_dict[f'{action_name}_action'] = action

            if electric_vehicle_actions:
                action_dict['electric_vehicle_storage_actions'] = electric_vehicle_actions

            if washing_machine_actions:
                action_dict['washing_machine_actions'] = washing_machine_actions

            parsed_actions.append(action_dict)

        return parsed_actions

    def next_time_step(self):
        r"""Advance all buildings to next `time_step`."""

        env = self.env
        current_step = int(env.time_step)
        last_action_step = max(env.time_steps - 2, 0)
        reached_terminal_transition = current_step >= last_action_step

        partial_render_time = 0.0
        if getattr(env, 'render_enabled', False):
            if env.render_mode == 'during':
                if env.debug_timing:
                    import time
                    render_start = time.perf_counter()
                    env.render()
                    partial_render_time = time.perf_counter() - render_start
                else:
                    env.render()

        if not reached_terminal_transition:
            for building in env.buildings:
                building.next_time_step()

            for electric_vehicle in env.electric_vehicles:
                electric_vehicle.next_time_step()

        Environment.next_time_step(env)

        if not reached_terminal_transition:
            self.simulate_unconnected_ev_soc()
            self.associate_chargers_to_electric_vehicles()

        return partial_render_time

    def associate_chargers_to_electric_vehicles(self):
        r"""Associate charger to its corresponding EV based on charger simulation state."""

        env = self.env

        def _resolve_arrival_soc(
            simulation: ChargerSimulation,
            step: int,
            prev_state: float,
            prev_id: Union[str, None],
            ev_identifier: str,
        ) -> Union[float, None]:
            current_soc = getattr(simulation, 'electric_vehicle_current_soc', None)
            if current_soc is not None and 0 <= step < len(current_soc):
                current_value = current_soc[step]
                if isinstance(current_value, (float, np.floating)) and not np.isnan(current_value) and 0.0 <= current_value <= 1.0:
                    return float(current_value)

            candidate_index = None

            if prev_state in (2, 3) and step > 0:
                if isinstance(prev_id, str) and prev_id.strip() not in {'', 'nan'} and prev_id != ev_identifier:
                    raise ValueError(
                        f"Charger dataset EV mismatch: expected '{ev_identifier}' but found '{prev_id}' at time step {step - 1}."
                    )
                candidate_index = step - 1

            elif 0 <= step < len(simulation.electric_vehicle_estimated_soc_arrival):
                candidate_index = step

            soc_value = None

            if candidate_index is not None and 0 <= candidate_index < len(simulation.electric_vehicle_estimated_soc_arrival):
                candidate = simulation.electric_vehicle_estimated_soc_arrival[candidate_index]
                if isinstance(candidate, (float, np.floating)) and not np.isnan(candidate) and 0.0 <= candidate <= 1.0:
                    soc_value = float(candidate)

            if soc_value is None and 0 <= step < len(simulation.electric_vehicle_required_soc_departure):
                fallback = simulation.electric_vehicle_required_soc_departure[step]
                if isinstance(fallback, (float, np.floating)) and not np.isnan(fallback) and 0.0 <= fallback <= 1.0:
                    soc_value = float(fallback)

            return soc_value

        for building in env.buildings:
            if building.electric_vehicle_chargers is None:
                continue

            for charger in building.electric_vehicle_chargers:
                sim = charger.charger_simulation
                state = sim.electric_vehicle_charger_state[env.time_step]

                if np.isnan(state) or state not in [1, 2]:
                    continue

                ev_id = sim.electric_vehicle_id[env.time_step]
                prev_state = np.nan
                prev_ev_id = None
                if env.time_step > 0:
                    idx = env.time_step - 1
                    if idx < len(sim.electric_vehicle_charger_state):
                        prev_state = sim.electric_vehicle_charger_state[idx]
                    if idx < len(sim.electric_vehicle_id):
                        prev_ev_id = sim.electric_vehicle_id[idx]

                if isinstance(ev_id, str) and ev_id.strip() not in ['', 'nan']:
                    for ev in env.electric_vehicles:
                        if ev.name == ev_id:
                            if state == 1:
                                charger.plug_car(ev)
                                is_new_connection = (
                                    prev_state != 1
                                    or not isinstance(prev_ev_id, str)
                                    or prev_ev_id != ev_id
                                )
                                if is_new_connection:
                                    soc_value = _resolve_arrival_soc(sim, env.time_step, prev_state, prev_ev_id, ev_id)
                                    if soc_value is not None:
                                        ev.battery.force_set_soc(soc_value)
                            elif state == 2:
                                charger.associate_incoming_car(ev)

    def simulate_unconnected_ev_soc(self):
        """Simulate SOC changes for EVs that are not under charger control at t+1."""

        env = self.env
        random_state = getattr(env, '_ev_drift_random_state', None)

        if random_state is None:
            episode_index = int(getattr(getattr(env, 'episode_tracker', None), 'episode', 0))
            random_state = np.random.RandomState(int(env.random_seed) + episode_index)
            env._ev_drift_random_state = random_state

        t = env.time_step
        if t + 1 >= env.episode_tracker.episode_time_steps:
            return

        for ev in env.electric_vehicles:
            ev_id = ev.name
            found_in_charger = False

            for building in env.buildings:
                for charger in building.electric_vehicle_chargers or []:
                    sim: ChargerSimulation = charger.charger_simulation

                    curr_id = sim.electric_vehicle_id[t] if t < len(sim.electric_vehicle_id) else ''
                    next_id = sim.electric_vehicle_id[t + 1] if t + 1 < len(sim.electric_vehicle_id) else ''
                    curr_state = sim.electric_vehicle_charger_state[t] if t < len(sim.electric_vehicle_charger_state) else np.nan
                    next_state = sim.electric_vehicle_charger_state[t + 1] if t + 1 < len(sim.electric_vehicle_charger_state) else np.nan

                    currently_connected = isinstance(curr_id, str) and curr_id == ev_id and curr_state == 1
                    if currently_connected:
                        found_in_charger = True
                        break

                    is_connecting = (
                        isinstance(next_id, str)
                        and next_id == ev_id
                        and next_state == 1
                        and curr_state != 1
                    )
                    is_incoming = isinstance(curr_id, str) and curr_id == ev_id and curr_state == 2

                    if is_connecting:
                        found_in_charger = True
                        if is_incoming:
                            if t < len(sim.electric_vehicle_estimated_soc_arrival):
                                soc = sim.electric_vehicle_estimated_soc_arrival[t]
                            else:
                                soc = np.nan
                        else:
                            if t + 1 < len(sim.electric_vehicle_estimated_soc_arrival):
                                soc = sim.electric_vehicle_estimated_soc_arrival[t + 1]
                            else:
                                soc = np.nan

                        if 0 <= soc <= 1:
                            ev.battery.force_set_soc(soc)
                        break

                if found_in_charger:
                    break

            if not found_in_charger:
                if t > 0:
                    last_soc = ev.battery.soc[t - 1]
                    drift_std = self._ev_unconnected_drift_std(env.seconds_per_time_step)
                    variability = np.clip(random_state.normal(1.0, drift_std), 0.6, 1.4)
                    new_soc = np.clip(last_soc * variability, 0.0, 1.0)
                    ev.battery.force_set_soc(new_soc)

    def update_variables(self):
        """Update district aggregate series from current building states."""

        env = self.env

        for building in env.buildings:
            building.update_variables()

        if getattr(env, 'community_market_enabled', False):
            self._apply_community_market_settlement()

        def _set_or_append(lst, value):
            if len(lst) == env.time_step:
                lst.append(value)
            elif len(lst) == env.time_step + 1:
                lst[env.time_step] = value
            else:
                del lst[env.time_step + 1:]
                if len(lst) < env.time_step:
                    lst.extend([0.0] * (env.time_step - len(lst)))
                lst.append(value)

        total = sum(building.net_electricity_consumption[env.time_step] for building in env.buildings)
        _set_or_append(env.net_electricity_consumption, total)

        total_cost = sum(building.net_electricity_consumption_cost[env.time_step] for building in env.buildings)
        _set_or_append(env.net_electricity_consumption_cost, total_cost)

        total_emission = sum(building.net_electricity_consumption_emission[env.time_step] for building in env.buildings)
        _set_or_append(env.net_electricity_consumption_emission, total_emission)

    @staticmethod
    def _to_scalar(value, default: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(scalar):
            return float(default)

        return scalar

    def _resolve_step_value(self, value, time_step: int, default: float = 0.0) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return float(default)
            index = min(max(time_step, 0), len(value) - 1)
            return self._to_scalar(value[index], default)

        return self._to_scalar(value, default)

    @staticmethod
    def _allocate_equal_share_import(imports: np.ndarray, traded_kwh: float) -> np.ndarray:
        """Allocate local traded energy equally among importers with demand caps."""

        allocations = np.zeros_like(imports, dtype='float64')
        remaining = max(float(traded_kwh), 0.0)
        eps = 1e-9

        while remaining > eps:
            needs = imports - allocations
            active = needs > eps
            active_count = int(np.count_nonzero(active))

            if active_count == 0:
                break

            share = remaining / active_count
            granted = np.minimum(share, needs[active])
            granted_total = float(granted.sum())

            if granted_total <= eps:
                break

            allocations[active] += granted
            remaining -= granted_total

        return allocations

    def _apply_community_market_settlement(self):
        """Apply optional intracommunity settlement and override building costs for current step."""

        env = self.env
        t = env.time_step
        if len(env.buildings) == 0:
            return

        ratio = self._to_scalar(getattr(env, 'community_market_sell_ratio', 0.8), 0.8)
        ratio = min(max(ratio, 0.0), 1.0)

        net_values = np.array([self._to_scalar(building.net_electricity_consumption[t], 0.0) for building in env.buildings], dtype='float64')
        imports = np.clip(net_values, 0.0, None)
        exports = np.clip(-net_values, 0.0, None)

        total_import = float(imports.sum())
        total_export = float(exports.sum())
        traded_kwh = min(total_import, total_export)

        if total_import > 0.0 and traded_kwh > 0.0:
            local_import = self._allocate_equal_share_import(imports, traded_kwh)
        else:
            local_import = np.zeros_like(imports, dtype='float64')

        if total_export > 0.0:
            local_export = exports * (traded_kwh / total_export)
        else:
            local_export = np.zeros_like(exports, dtype='float64')

        grid_export_price_cfg = getattr(env, 'community_market_grid_export_price', 0.0)
        market_settlement = []

        for idx, building in enumerate(env.buildings):
            grid_import_price = self._to_scalar(building.pricing.electricity_pricing[t], 0.0)
            local_price = ratio * grid_import_price
            grid_export_price = self._resolve_step_value(grid_export_price_cfg, t, 0.0)
            counterfactual_legacy_cost = self._to_scalar(building.net_electricity_consumption_cost[t], 0.0)

            grid_import_remaining = max(imports[idx] - local_import[idx], 0.0)
            grid_export_remaining = max(exports[idx] - local_export[idx], 0.0)

            cost = (
                grid_import_remaining * grid_import_price
                + local_import[idx] * local_price
                - local_export[idx] * local_price
                - grid_export_remaining * grid_export_price
            )
            savings = counterfactual_legacy_cost - cost

            building.set_net_electricity_consumption_cost(cost, time_step=t)
            market_settlement.append(
                {
                    'building': building.name,
                    'local_import_kwh': float(local_import[idx]),
                    'local_export_kwh': float(local_export[idx]),
                    'grid_import_kwh': float(grid_import_remaining),
                    'grid_export_kwh': float(grid_export_remaining),
                    'local_price': float(local_price),
                    'grid_import_price': float(grid_import_price),
                    'grid_export_price': float(grid_export_price),
                    'counterfactual_cost_eur': float(counterfactual_legacy_cost),
                    'settled_cost_eur': float(cost),
                    'market_savings_eur': float(savings),
                }
            )

        env._last_community_market_settlement = market_settlement

        history = getattr(env, '_community_market_settlement_history', None)
        if history is not None:
            if len(history) == t:
                history.append(market_settlement)
            elif len(history) == t + 1:
                history[t] = market_settlement
            else:
                del history[t + 1:]
                if len(history) < t:
                    history.extend([[] for _ in range(t - len(history))])
                history.append(market_settlement)
