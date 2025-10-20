from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent
from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_charging_constraints_demo/schema.json"


def _zero_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def test_charging_constraint_demo_rollout(tmp_path):
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        render_mode="during",
        render_directory=tmp_path / "SimulationData",
        episode_time_steps=48,
        random_seed=0,
    )

    try:
        controller = Agent(env)
        controller.learn(episodes=1, logging_level=1)
        kpis = controller.env.evaluate()
        assert not kpis.empty
    finally:
        env.close()


def test_charging_constraint_demo_zero_actions():
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=12,
        random_seed=0,
    )

    try:
        env.reset()
        while not env.terminated:
            env.step(_zero_actions(env))

        df = env.evaluate()
        finite = df["value"].dropna()
        assert not finite.empty
        assert np.isfinite(finite).all()
    finally:
        env.close()
