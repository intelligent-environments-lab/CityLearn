from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _assert_length_consistency(env: CityLearnEnv):
    t = env.time_step
    for building in env.buildings:
        assert len(building.net_electricity_consumption) == t + 1
        assert len(building.cooling_electricity_consumption) == t + 1
        assert len(building.heating_electricity_consumption) == t + 1
        assert len(building.dhw_electricity_consumption) == t + 1
        assert len(building.non_shiftable_load_electricity_consumption) == t + 1
        assert len(building.electrical_storage_electricity_consumption) == t + 1
        assert len(building.chargers_electricity_consumption) == t + 1
        assert len(building.washing_machines_electricity_consumption) == t + 1


def test_series_integrity_reset_and_step():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        assert env.time_step == 0
        assert len(env.net_electricity_consumption) == 1
        assert len(env.net_electricity_consumption_cost) == 1
        assert len(env.net_electricity_consumption_emission) == 1
        _assert_length_consistency(env)

        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]

        env.step(zeros)
        assert env.time_step == 1
        assert len(env.net_electricity_consumption) == 1
        assert len(env.net_electricity_consumption_cost) == 1
        assert len(env.net_electricity_consumption_emission) == 1
        _assert_length_consistency(env)

        env.step(zeros)
        assert env.time_step == 2
        assert len(env.net_electricity_consumption) == 2
        assert len(env.net_electricity_consumption_cost) == 2
        assert len(env.net_electricity_consumption_emission) == 2
        _assert_length_consistency(env)
    finally:
        env.close()


def test_bess_first_step_not_double_counted():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        action = np.zeros(env.action_space[0].shape[0], dtype="float32")

        offset = 0
        target_building = None
        target_action_index = None

        for building in env.buildings:
            action_count = len(building.active_actions)
            if "electrical_storage" in building.active_actions:
                local_index = building.active_actions.index("electrical_storage")
                target_action_index = offset + local_index
                target_building = building
                break
            offset += action_count

        assert target_building is not None
        assert target_action_index is not None
        action[target_action_index] = 0.5

        env.step([action])

        t = 0
        expected = target_building.electrical_storage.energy_balance[t]
        actual = target_building.electrical_storage_electricity_consumption[t]

        assert abs(expected) > 1e-9
        assert actual == pytest.approx(expected, abs=1e-6)
    finally:
        env.close()


def test_non_shiftable_first_step_not_double_counted():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        action = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
        env.step(action)

        for building in env.buildings:
            expected = building.energy_to_non_shiftable_load[0]
            actual = building.non_shiftable_load_electricity_consumption[0]
            assert actual == pytest.approx(expected, abs=1e-6)
    finally:
        env.close()


def test_non_shiftable_t0_update_variables_idempotent():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        before = {
            building.name: float(building.non_shiftable_load_electricity_consumption[0])
            for building in env.buildings
        }

        env.update_variables()

        after = {
            building.name: float(building.non_shiftable_load_electricity_consumption[0])
            for building in env.buildings
        }
        assert after == pytest.approx(before, abs=1e-6)
    finally:
        env.close()


def test_terminal_series_exclude_uncommitted_tail_slot():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]

        while not env.terminated:
            _, _, terminated, truncated, _ = env.step(zeros)
            if terminated or truncated:
                break

        assert env.time_step == env.time_steps - 1

        for building in env.buildings:
            assert building.time_step == env.time_step - 1
            assert len(building.net_electricity_consumption) == env.time_step
            assert len(building.net_electricity_consumption_cost) == env.time_step
            assert len(building.net_electricity_consumption_emission) == env.time_step
            assert len(building.electrical_storage_electricity_consumption) == env.time_step
    finally:
        env.close()
