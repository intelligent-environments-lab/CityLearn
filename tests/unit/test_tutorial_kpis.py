"""Tutorial KPI regression tests."""

from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv, EvaluationCondition
from citylearn.agents.base import Agent as RandomAgent


DATASET = Path(__file__).resolve().parents[2] / 'data/datasets/citylearn_challenge_2022_phase_all/schema.json'
KPI_KEYS = (
    'daily_peak_average',
    'daily_one_minus_load_factor_average',
    'ramping_average',
)


def _assert_baseline_kpis(df):
    district = df[df['name'] == 'District'].set_index('cost_function')['value']

    for key in KPI_KEYS:
        value = float(district.get(key, 1.0))
        assert np.isfinite(value)
        assert abs(value - 1.0) < 1e-5, f'{key} deviated: {value}'


def test_tutorial_baseline_kpis():
    env = CityLearnEnv(str(DATASET), central_agent=True, episode_time_steps=240)

    try:
        obs, _ = env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]

        while not env.terminated:
            obs, _, _, _, _ = env.step(zeros)

        df = env.evaluate(
            control_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
            baseline_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
        )
        _assert_baseline_kpis(df)
    finally:
        env.close()


def test_tutorial_random_agent_kpis():
    env = CityLearnEnv(str(DATASET), central_agent=True, episode_time_steps=240)
    agent = RandomAgent(env)

    try:
        obs, _ = env.reset()

        while not env.terminated:
            actions = agent.predict(obs)
            obs, _, _, _, _ = env.step(actions)

        df = env.evaluate(
            control_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
            baseline_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
        )
        _assert_baseline_kpis(df)
    finally:
        env.close()
