from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _zero_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def test_agent_observations_use_lagged_endogenous_values():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        observations, _ = env.reset()
        names = env.observation_names[0]
        net_ix = names.index("net_electricity_consumption")
        soc_ix = names.index("electrical_storage_soc")

        assert observations[0][net_ix] == pytest.approx(float(env.buildings[0].net_electricity_consumption[0]))
        assert observations[0][soc_ix] == pytest.approx(float(env.buildings[0].electrical_storage.soc[0]))

        next_observations, _, _, _, _ = env.step(_zero_actions(env))
        assert env.time_step == 1
        assert next_observations[0][net_ix] == pytest.approx(float(env.buildings[0].net_electricity_consumption[0]))
        assert next_observations[0][soc_ix] == pytest.approx(float(env.buildings[0].electrical_storage.soc[0]))
    finally:
        env.close()


def test_include_all_observations_remain_on_current_transition_time():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        env.step(_zero_actions(env))
        building = env.buildings[0]
        t = env.time_step
        lagged_t = max(t - 1, 0)

        all_obs = building.observations(include_all=True, normalize=False, periodic_normalization=False)
        agent_obs = building.observations(include_all=False, normalize=False, periodic_normalization=False)

        assert all_obs["net_electricity_consumption"] == pytest.approx(float(building.net_electricity_consumption[t]))
        assert agent_obs["net_electricity_consumption"] == pytest.approx(float(building.net_electricity_consumption[lagged_t]))
        assert all_obs["electrical_storage_soc"] == pytest.approx(float(building.electrical_storage.soc[t]))
        assert agent_obs["electrical_storage_soc"] == pytest.approx(float(building.electrical_storage.soc[lagged_t]))
    finally:
        env.close()


def test_step_after_terminal_raises_runtime_error():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=2, random_seed=0)

    try:
        env.reset()
        _, _, terminated, truncated, _ = env.step(_zero_actions(env))
        assert terminated
        assert not truncated

        with pytest.raises(RuntimeError, match="reset"):
            env.step(_zero_actions(env))
    finally:
        env.close()


def test_step_raises_for_invalid_central_action_count():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        expected = env.action_space[0].shape[0]
        invalid = np.zeros(expected + 1, dtype="float32")

        with pytest.raises(AssertionError, match="Expected"):
            env.step([invalid])
    finally:
        env.close()


def test_step_accepts_central_numpy_action_vector():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        action = np.zeros(env.action_space[0].shape[0], dtype="float32")
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_step_accepts_central_single_row_numpy_action_vector():
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        action = np.zeros((1, env.action_space[0].shape[0]), dtype="float32")
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_step_raises_for_missing_decentralized_action_vector():
    env = CityLearnEnv(str(SCHEMA), central_agent=False, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        actions = _zero_actions(env)

        if len(actions) < 2:
            pytest.skip("Dataset does not expose multiple buildings for decentralized action validation.")

        with pytest.raises(AssertionError, match="building action vectors"):
            env.step(actions[:-1])
    finally:
        env.close()


def test_step_accepts_valid_decentralized_action_vectors():
    env = CityLearnEnv(str(SCHEMA), central_agent=False, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        actions = _zero_actions(env)
        _, _, terminated, truncated, _ = env.step(actions)
        assert not terminated
        assert not truncated
    finally:
        env.close()
