import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _run_fixed_episode(episode_time_steps: int = 12) -> pd.DataFrame:
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=episode_time_steps,
        random_seed=0,
    )

    try:
        env.reset()
        action_names = env.action_names[0]
        action = np.zeros(env.action_space[0].shape[0], dtype="float32")

        ev_indices = [i for i, name in enumerate(action_names) if name.startswith("electric_vehicle_storage_")]
        battery_indices = [i for i, name in enumerate(action_names) if "electrical_storage" in name]

        while not env.terminated:
            rollout_action = action.copy()
            if ev_indices:
                rollout_action[ev_indices] = 0.6
            if battery_indices:
                rollout_action[battery_indices] = 0.4

            _, _, terminated, truncated, _ = env.step([rollout_action])
            if terminated or truncated:
                break

        df = env.evaluate().sort_values(["level", "name", "cost_function"]).reset_index(drop=True)
        return df
    finally:
        env.close()


def test_loading_contract_shapes_and_counts_match_schema():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    expected_buildings = [name for name, cfg in schema["buildings"].items() if cfg.get("include", False)]
    expected_evs = [name for name, cfg in schema.get("electric_vehicles_def", {}).items() if cfg.get("include", False)]

    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=False,
        episode_time_steps=6,
        random_seed=0,
    )

    try:
        env.reset()

        actual_buildings = [b.name for b in env.buildings]
        actual_evs = [ev.name for ev in env.electric_vehicles]

        assert actual_buildings == expected_buildings
        assert actual_evs == expected_evs

        assert len(env.observation_space) == len(env.buildings)
        assert len(env.action_space) == len(env.buildings)
        assert len(env.observation_names) == len(env.buildings)
        assert len(env.action_names) == len(env.buildings)

        for i, _ in enumerate(env.buildings):
            assert env.observation_space[i].shape[0] == len(env.observation_names[i])
            assert env.action_space[i].shape[0] == len(env.action_names[i])
    finally:
        env.close()


def test_kpi_results_are_stable_for_fixed_seed_and_actions():
    first = _run_fixed_episode(episode_time_steps=12)
    second = _run_fixed_episode(episode_time_steps=12)

    pd.testing.assert_frame_equal(first, second, check_exact=False, atol=1e-10, rtol=1e-10)
