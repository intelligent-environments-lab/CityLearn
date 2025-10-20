from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


@pytest.fixture
def env():
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=96,
        random_seed=0,
    )
    env.reset()
    try:
        yield env
    finally:
        env.close()


def _zero_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def _ev_action_index(env: CityLearnEnv, building, charger_id: str) -> int:
    offset = 0
    for b in env.buildings:
        if b is building:
            break
        offset += b.action_space.shape[0]
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Building not found while resolving action index.")

    action_key = f"electric_vehicle_storage_{charger_id}"
    local_index = building.active_actions.index(action_key)
    return offset + local_index


def _get_charger(building, charger_id: str):
    for charger in building.electric_vehicle_chargers or []:
        if charger.charger_id == charger_id:
            return charger

    raise AssertionError(f"Charger {charger_id} not found in building {building.name}")


def test_soc_increases_when_connected_ev_is_charged(env: CityLearnEnv):
    for _ in range(env.episode_time_steps - 2):
        t = env.time_step

        for building in env.buildings:
            if not building.electric_vehicle_chargers:
                continue

            for charger in building.electric_vehicle_chargers:
                ev = charger.connected_electric_vehicle
                if ev is None:
                    continue

                sim = charger.charger_simulation
                if t + 1 >= len(sim.electric_vehicle_charger_state):
                    continue

                connected_now = (
                    isinstance(sim.electric_vehicle_id[t], str)
                    and sim.electric_vehicle_id[t] == ev.name
                    and sim.electric_vehicle_charger_state[t] == 1
                )
                connected_next = sim.electric_vehicle_charger_state[t + 1] == 1

                if connected_now and connected_next:
                    initial_soc = float(ev.battery.soc[t])
                    actions = _zero_actions(env)
                    actions[0][_ev_action_index(env, building, charger.charger_id)] = 1.0
                    env.step(actions)

                    charger_after = _get_charger(building, charger.charger_id)
                    ev_after = charger_after.connected_electric_vehicle
                    assert ev_after is not None

                    new_index = max(env.time_step - 1, 0)
                    new_soc = float(ev_after.battery.soc[new_index])
                    assert new_soc > initial_soc + 1e-3
                    return

        env.step(_zero_actions(env))

    pytest.fail("No EV remained connected long enough for SOC charging test.")


def test_soc_matches_dataset_on_arrival(env: CityLearnEnv):
    for _ in range(env.episode_time_steps - 2):
        t = env.time_step

        for building in env.buildings:
            if not building.electric_vehicle_chargers:
                continue

            for charger in building.electric_vehicle_chargers:
                sim = charger.charger_simulation
                if t + 1 >= len(sim.electric_vehicle_charger_state):
                    continue

                curr_state = sim.electric_vehicle_charger_state[t]
                next_state = sim.electric_vehicle_charger_state[t + 1]
                next_id = sim.electric_vehicle_id[t + 1] if t + 1 < len(sim.electric_vehicle_id) else ""

                if not isinstance(next_id, str) or not next_id.strip():
                    continue

                if curr_state == 1 or next_state != 1:
                    continue

                if curr_state == 2:
                    if t >= len(sim.electric_vehicle_estimated_soc_arrival):
                        continue
                    expected_soc = float(sim.electric_vehicle_estimated_soc_arrival[t])
                else:
                    if t + 1 >= len(sim.electric_vehicle_estimated_soc_arrival):
                        continue
                    expected_soc = float(sim.electric_vehicle_estimated_soc_arrival[t + 1])

                if not (0.0 <= expected_soc <= 1.0):
                    continue

                env.step(_zero_actions(env))

                charger_after = _get_charger(building, charger.charger_id)
                ev_after = charger_after.connected_electric_vehicle
                if ev_after is None or ev_after.name != next_id:
                    continue

                new_soc = float(ev_after.battery.soc[env.time_step])
                assert pytest.approx(expected_soc, rel=0, abs=1e-5) == new_soc
                return

        env.step(_zero_actions(env))

    pytest.fail("No EV arrival transition observed during simulation.")
