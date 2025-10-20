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
