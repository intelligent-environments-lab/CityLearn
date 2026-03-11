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

            if env.render_enabled and not env._final_kpis_exported:
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

        actions = list(actions)
        building_actions = []

        if env.central_agent:
            actions = actions[0]
            number_of_actions = len(actions)
            expected_number_of_actions = env._expected_central_action_count
            assert number_of_actions == expected_number_of_actions, \
                f'Expected {expected_number_of_actions} actions but {number_of_actions} were parsed to env.step.'

            for building in env.buildings:
                size = building.action_space.shape[0]
                building_actions.append(actions[0:size])
                actions = actions[size:]

        else:
            building_actions = [list(a) for a in actions]
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

        for building in env.buildings:
            building.next_time_step()

        for electric_vehicle in env.electric_vehicles:
            electric_vehicle.next_time_step()

        Environment.next_time_step(env)

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
                    variability = np.clip(np.random.normal(1.0, 0.2), 0.6, 1.4)
                    new_soc = np.clip(last_soc * variability, 0.0, 1.0)
                    ev.battery.force_set_soc(new_soc)

    def update_variables(self):
        """Update district aggregate series from current building states."""

        env = self.env

        for building in env.buildings:
            building.update_variables()

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
