from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent
from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _make_zero_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def test_basic_ev_rbc_completes_episode(tmp_path):
    render_dir = tmp_path / "render_evs"
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        render_mode="during",
        render_directory=render_dir,
        episode_time_steps=24,
        random_seed=0,
    )

    try:
        controller = Agent(env)
        controller.learn(episodes=1, logging_level=1)

        kpis = controller.env.evaluate()
        assert not kpis.empty

        # Ensure rendering pipeline produced output when enabled.
        assert render_dir.exists()
    finally:
        env.close()


def test_zero_action_rollout_keeps_kpis_finite():
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=12,
        random_seed=0,
    )

    try:
        env.reset()
        while not env.terminated:
            env.step(_make_zero_actions(env))

        kpis = env.evaluate()
        finite = kpis["value"].dropna()
        assert not finite.empty
        assert np.isfinite(finite).all()
    finally:
        env.close()


def test_rbc_sets_ev_actions_positive():
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=4,
        random_seed=0,
    )

    try:
        controller = Agent(env)
        observations, _ = env.reset()
        actions = controller.predict(observations, deterministic=True)

        ev_indices = [
            idx
            for idx, name in enumerate(env.action_names[0])
            if name.startswith("electric_vehicle_storage_")
        ]
        assert ev_indices, "Expected EV storage actions in action space."

        for idx in ev_indices:
            assert actions[0][idx] > 0.0
    finally:
        env.close()
