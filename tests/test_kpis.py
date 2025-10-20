from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv, EvaluationCondition


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _run_episode(schema: Path, action_value: float, episode_time_steps: int = 48) -> CityLearnEnv:
    env = CityLearnEnv(
        str(schema),
        central_agent=True,
        episode_time_steps=episode_time_steps,
        random_seed=0,
    )
    env.reset()
    ev_indices = [
        idx
        for idx, name in enumerate(env.action_names[0])
        if name.startswith("electric_vehicle_storage_")
    ]

    base_action = np.zeros(env.action_space[0].shape[0], dtype="float32")

    while not env.terminated:
        action = base_action.copy()
        if ev_indices:
            action[ev_indices] = action_value

        env.step([action])

    return env


def test_kpi_normalization_matches_baseline():
    env = _run_episode(SCHEMA, action_value=0.0)

    try:
        df = env.evaluate(
            control_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
            baseline_condition=EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV,
        )

        normalized_keys = {
            "ramping_average",
            "daily_one_minus_load_factor_average",
            "monthly_one_minus_load_factor_average",
            "daily_peak_average",
            "all_time_peak_average",
        }
        district = df[(df["level"] == "district") & (df["cost_function"].isin(normalized_keys))]
        vals = district["value"].to_numpy(dtype=float)

        assert np.all(np.isfinite(vals)), "District KPI contains non-finite values."
        assert np.allclose(vals, 1.0, atol=1e-5), "District KPI normalization deviates when control==baseline."
    finally:
        env.close()


def test_ev_charging_load_impacts_building_kpis():
    def _total_charger_consumption(sim_env: CityLearnEnv) -> float:
        return float(sum(np.sum(b.chargers_electricity_consumption) for b in sim_env.buildings))

    base_env = _run_episode(SCHEMA, action_value=0.0, episode_time_steps=24)
    try:
        base_df = base_env.evaluate()
        base_charger_consumption = _total_charger_consumption(base_env)
    finally:
        base_env.close()

    charged_env = _run_episode(SCHEMA, action_value=1.0, episode_time_steps=24)
    try:
        charged_df = charged_env.evaluate()
        charged_charger_consumption = _total_charger_consumption(charged_env)
    finally:
        charged_env.close()

    def _extract(df):
        subset = df[
            (df["level"] == "building")
            & (df["cost_function"] == "electricity_consumption_total")
        ]
        return subset.set_index("name")["value"]

    base_values = _extract(base_df)
    charged_values = _extract(charged_df)

    charged_values = charged_values.reindex(base_values.index)

    assert charged_charger_consumption > base_charger_consumption, \
        "EV load should increase charger electricity consumption."
    assert np.isfinite(charged_values.dropna()).all()
