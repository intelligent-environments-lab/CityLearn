import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv, EvaluationCondition
from citylearn.cost_function import CostFunction


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
THREE_PHASE_SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_three_phase_electrical_service_demo/schema.json"
MINUTE_DATASET_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "minute_ev_demo"


def _run_episode(schema: Path, seconds_per_time_step: int, episode_steps: int = 24) -> CityLearnEnv:
    env = CityLearnEnv(
        str(schema),
        central_agent=True,
        episode_time_steps=episode_steps,
        seconds_per_time_step=seconds_per_time_step,
        random_seed=0,
    )
    env.reset()

    action_names = env.action_names[0]
    base_action = np.zeros(env.action_space[0].shape[0], dtype="float32")
    ev_indices = [i for i, name in enumerate(action_names) if name.startswith("electric_vehicle_storage_")]
    bess_indices = [i for i, name in enumerate(action_names) if name == "electrical_storage"]

    while not env.terminated:
        action = base_action.copy()
        if ev_indices:
            action[ev_indices] = 0.7
        if bess_indices:
            action[bess_indices] = 0.5
        env.step([action])

    return env


def _build_two_building_market_schema(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "market_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    weather = pd.read_csv(MINUTE_DATASET_DIR / "weather.csv").iloc[:2].copy()
    weather.to_csv(dataset_dir / "weather.csv", index=False)

    carbon = pd.read_csv(MINUTE_DATASET_DIR / "carbon_intensity.csv").iloc[:2].copy()
    carbon.to_csv(dataset_dir / "carbon_intensity.csv", index=False)

    pricing = pd.DataFrame(
        {
            "electricity_pricing": [0.5, 0.5],
            "electricity_pricing_predicted_1": [0.5, 0.5],
            "electricity_pricing_predicted_2": [0.5, 0.5],
            "electricity_pricing_predicted_3": [0.5, 0.5],
        }
    )
    pricing.to_csv(dataset_dir / "pricing.csv", index=False)

    building_a = pd.DataFrame(
        {
            "month": [1, 1],
            "hour": [0, 1],
            "minutes": [0, 0],
            "day_type": [1, 1],
            "daylight_savings_status": [0, 0],
            "indoor_dry_bulb_temperature": [21.0, 21.0],
            "average_unmet_cooling_setpoint_difference": [0.0, 0.0],
            "indoor_relative_humidity": [45.0, 45.0],
            "non_shiftable_load": [2.0, 2.0],
            "dhw_demand": [0.0, 0.0],
            "cooling_demand": [0.0, 0.0],
            "heating_demand": [0.0, 0.0],
            "solar_generation": [0.0, 0.0],
        }
    )
    building_b = building_a.copy()
    building_b["non_shiftable_load"] = [0.0, 0.0]
    building_b["solar_generation"] = [2000.0, 2000.0]

    building_a.to_csv(dataset_dir / "Building_A.csv", index=False)
    building_b.to_csv(dataset_dir / "Building_B.csv", index=False)

    schema = {
        "random_seed": 0,
        "root_directory": None,
        "central_agent": True,
        "simulation_start_time_step": 0,
        "simulation_end_time_step": 1,
        "episode_time_steps": 2,
        "rolling_episode_split": False,
        "random_episode_split": False,
        "seconds_per_time_step": 3600,
        "observations": {
            "month": {"active": True, "shared_in_central_agent": True},
            "hour": {"active": True, "shared_in_central_agent": True},
            "minutes": {"active": True, "shared_in_central_agent": True},
            "day_type": {"active": True, "shared_in_central_agent": True},
            "outdoor_dry_bulb_temperature": {"active": True, "shared_in_central_agent": True},
            "non_shiftable_load": {"active": True, "shared_in_central_agent": False},
            "solar_generation": {"active": True, "shared_in_central_agent": False},
            "net_electricity_consumption": {"active": True, "shared_in_central_agent": False},
            "electricity_pricing": {"active": True, "shared_in_central_agent": True},
        },
        "actions": {
            "electrical_storage": {"active": False},
        },
        "reward_function": {
            "type": "citylearn.reward_function.RewardFunction",
            "attributes": {},
        },
        "community_market": {
            "enabled": True,
            "intra_community_sell_ratio": 0.8,
            "grid_export_price": 0.0,
        },
        "buildings": {
            "Building_A": {
                "include": True,
                "energy_simulation": "Building_A.csv",
                "weather": "weather.csv",
                "carbon_intensity": "carbon_intensity.csv",
                "pricing": "pricing.csv",
                "inactive_observations": [],
                "inactive_actions": [],
                "pv": {
                    "type": "citylearn.energy_model.PV",
                    "autosize": False,
                    "attributes": {"nominal_power": 0.0},
                },
            },
            "Building_B": {
                "include": True,
                "energy_simulation": "Building_B.csv",
                "weather": "weather.csv",
                "carbon_intensity": "carbon_intensity.csv",
                "pricing": "pricing.csv",
                "inactive_observations": [],
                "inactive_actions": [],
                "pv": {
                    "type": "citylearn.energy_model.PV",
                    "autosize": False,
                    "attributes": {"nominal_power": 1.0},
                },
            },
        },
    }

    schema_path = dataset_dir / "schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema_path


def _build_schema_with_manual_equity_groups(
    tmp_path: Path,
    source_schema: Path,
    *,
    missing_first_group: bool,
) -> Path:
    with open(source_schema, "r", encoding="utf-8") as f:
        schema = json.load(f)
    schema["root_directory"] = str(source_schema.parent)

    building_names = [name for name, config in schema.get("buildings", {}).items() if config.get("include", False)]

    for i, name in enumerate(building_names):
        if missing_first_group and i == 0:
            schema["buildings"][name].pop("equity_group", None)
            continue

        schema["buildings"][name]["equity_group"] = "asset_rich" if i % 2 == 0 else "asset_poor"

    schema_path = tmp_path / "schema_with_equity_groups.json"

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema_path


def _build_zero_sum_pricing_schema(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "zero_sum_pricing_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    weather = pd.read_csv(MINUTE_DATASET_DIR / "weather.csv").iloc[:3].copy()
    weather.to_csv(dataset_dir / "weather.csv", index=False)

    carbon = pd.read_csv(MINUTE_DATASET_DIR / "carbon_intensity.csv").iloc[:3].copy()
    carbon.to_csv(dataset_dir / "carbon_intensity.csv", index=False)

    pricing = pd.DataFrame(
        {
            "electricity_pricing": [1.0, -1.0, 0.0],
            "electricity_pricing_predicted_1": [1.0, -1.0, 0.0],
            "electricity_pricing_predicted_2": [1.0, -1.0, 0.0],
            "electricity_pricing_predicted_3": [1.0, -1.0, 0.0],
        }
    )
    pricing.to_csv(dataset_dir / "pricing.csv", index=False)

    building = pd.DataFrame(
        {
            "month": [1, 1, 1],
            "hour": [0, 1, 2],
            "minutes": [0, 0, 0],
            "day_type": [1, 1, 1],
            "daylight_savings_status": [0, 0, 0],
            "indoor_dry_bulb_temperature": [21.0, 21.0, 21.0],
            "average_unmet_cooling_setpoint_difference": [0.0, 0.0, 0.0],
            "indoor_relative_humidity": [45.0, 45.0, 45.0],
            "non_shiftable_load": [1.0, 2.0, 3.0],
            "dhw_demand": [0.0, 0.0, 0.0],
            "cooling_demand": [0.0, 0.0, 0.0],
            "heating_demand": [0.0, 0.0, 0.0],
            "solar_generation": [0.0, 0.0, 0.0],
        }
    )
    building.to_csv(dataset_dir / "Building_1.csv", index=False)

    schema = {
        "random_seed": 0,
        "root_directory": None,
        "central_agent": True,
        "simulation_start_time_step": 0,
        "simulation_end_time_step": 2,
        "episode_time_steps": 3,
        "rolling_episode_split": False,
        "random_episode_split": False,
        "seconds_per_time_step": 3600,
        "observations": {
            "month": {"active": True, "shared_in_central_agent": True},
            "hour": {"active": True, "shared_in_central_agent": True},
            "minutes": {"active": True, "shared_in_central_agent": True},
            "day_type": {"active": True, "shared_in_central_agent": True},
            "outdoor_dry_bulb_temperature": {"active": True, "shared_in_central_agent": True},
            "non_shiftable_load": {"active": True, "shared_in_central_agent": False},
            "net_electricity_consumption": {"active": True, "shared_in_central_agent": False},
            "electricity_pricing": {"active": True, "shared_in_central_agent": True},
        },
        "actions": {"electrical_storage": {"active": False}},
        "reward_function": {"type": "citylearn.reward_function.RewardFunction", "attributes": {}},
        "buildings": {
            "Building_1": {
                "include": True,
                "energy_simulation": "Building_1.csv",
                "weather": "weather.csv",
                "carbon_intensity": "carbon_intensity.csv",
                "pricing": "pricing.csv",
                "inactive_observations": [],
                "inactive_actions": [],
                "pv": {
                    "type": "citylearn.energy_model.PV",
                    "autosize": False,
                    "attributes": {"nominal_power": 0.0},
                },
            }
        },
    }

    schema_path = dataset_dir / "schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema_path


@pytest.mark.parametrize("seconds_per_time_step", [5, 10, 60, 300, 900])
def test_daily_and_monthly_kpis_use_time_aware_windows(seconds_per_time_step: int):
    env = _run_episode(SCHEMA, seconds_per_time_step=seconds_per_time_step, episode_steps=24)

    try:
        control = EvaluationCondition.WITH_STORAGE_AND_PV
        baseline = EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PV
        df = env.evaluate(control_condition=control, baseline_condition=baseline)

        daily_steps = max(1, int(round((24 * 3600) / seconds_per_time_step)))
        monthly_steps = max(1, int(round((730 * 3600) / seconds_per_time_step)))

        dlf_daily_c = CostFunction.one_minus_load_factor(env.net_electricity_consumption, window=daily_steps)[-1]
        dlf_daily_b = CostFunction.one_minus_load_factor(
            getattr(env, f"net_electricity_consumption{baseline.value}"),
            window=daily_steps,
        )[-1]
        dlf_monthly_c = CostFunction.one_minus_load_factor(env.net_electricity_consumption, window=monthly_steps)[-1]
        dlf_monthly_b = CostFunction.one_minus_load_factor(
            getattr(env, f"net_electricity_consumption{baseline.value}"),
            window=monthly_steps,
        )[-1]
        peak_daily_c = CostFunction.peak(env.net_electricity_consumption, window=daily_steps)[-1]
        peak_daily_b = CostFunction.peak(
            getattr(env, f"net_electricity_consumption{baseline.value}"),
            window=daily_steps,
        )[-1]

        def safe_div(c, b):
            if b == 0.0:
                return 1.0 if c == 0.0 else np.nan
            return c / b

        expected_daily = safe_div(float(dlf_daily_c), float(dlf_daily_b))
        expected_monthly = safe_div(float(dlf_monthly_c), float(dlf_monthly_b))
        expected_peak_daily = safe_div(float(peak_daily_c), float(peak_daily_b))

        district = df[df["name"] == "District"].set_index("cost_function")["value"]
        assert float(district["daily_one_minus_load_factor_average"]) == pytest.approx(expected_daily)
        assert float(district["monthly_one_minus_load_factor_average"]) == pytest.approx(expected_monthly)
        assert float(district["daily_peak_average"]) == pytest.approx(expected_peak_daily)
    finally:
        env.close()


def test_cost_baseline_total_eur_not_forced_to_zero_when_price_sum_is_zero(tmp_path: Path):
    schema_path = _build_zero_sum_pricing_schema(tmp_path)
    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=3, random_seed=0)

    try:
        env.reset()
        zeros = np.zeros(len(env.action_names[0]), dtype="float32")
        while not env.terminated:
            env.step([zeros])

        baseline_condition = EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PV
        baseline_series = getattr(
            env.buildings[0],
            f"net_electricity_consumption_cost{baseline_condition.value}",
        )
        baseline_array = np.asarray(baseline_series, dtype="float64")
        expected_baseline_total = float(baseline_array[np.isfinite(baseline_array)].sum())

        df = env.evaluate(
            control_condition=EvaluationCondition.WITH_STORAGE_AND_PV,
            baseline_condition=baseline_condition,
        )
        building_value = float(
            df[(df["name"] == "Building_1") & (df["cost_function"] == "cost_baseline_total_eur")]["value"].iloc[0]
        )
        district_value = float(
            df[(df["name"] == "District") & (df["cost_function"] == "cost_baseline_total_eur")]["value"].iloc[0]
        )

        assert expected_baseline_total != 0.0
        assert building_value == pytest.approx(expected_baseline_total, abs=1e-9)
        assert district_value == pytest.approx(expected_baseline_total, abs=1e-9)
    finally:
        env.close()


def test_kpi_v2_adds_domain_and_market_metrics(tmp_path: Path):
    schema_path = _build_two_building_market_schema(tmp_path)
    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=2, random_seed=0)

    try:
        env.reset()
        env.step([np.zeros(len(env.action_names[0]), dtype="float32")])
        df = env.evaluate()

        expected = {
            "electricity_consumption_control_total_kwh",
            "cost_control_total_eur",
            "cost_control_daily_average_eur",
            "ev_departure_success_rate",
            "bess_throughput_total_kwh",
            "pv_generation_total_kwh",
            "pv_export_daily_average_kwh",
            "community_local_import_total_kwh",
            "community_grid_export_after_local_daily_average_kwh",
            "community_settled_cost_total_eur",
            "community_market_savings_daily_average_eur",
            "community_market_savings_total_eur",
        }
        assert expected.issubset(set(df["cost_function"].unique()))

        district = df[df["name"] == "District"].set_index("cost_function")["value"]
        settled = float(district["community_settled_cost_total_eur"])
        counterfactual = float(district["community_counterfactual_cost_total_eur"])
        savings = float(district["community_market_savings_total_eur"])

        assert savings == pytest.approx(counterfactual - settled, abs=1e-9)
    finally:
        env.close()


def test_daily_average_kpis_match_total_over_simulated_days(tmp_path: Path):
    schema_path = _build_two_building_market_schema(tmp_path)
    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=2, random_seed=0)

    try:
        env.reset()
        env.step([np.zeros(len(env.action_names[0]), dtype="float32")])
        df = env.evaluate()
        district = df[df["name"] == "District"].set_index("cost_function")["value"]

        simulated_days = max(int(env.time_step), 1) * float(env.seconds_per_time_step) / (24.0 * 3600.0)

        pairs = [
            ("cost_delta_total_eur", "cost_delta_daily_average_eur"),
            ("electricity_consumption_delta_total_kwh", "electricity_consumption_delta_daily_average_kwh"),
            ("pv_export_total_kwh", "pv_export_daily_average_kwh"),
            ("community_grid_export_after_local_total_kwh", "community_grid_export_after_local_daily_average_kwh"),
            ("community_market_savings_total_eur", "community_market_savings_daily_average_eur"),
        ]

        for total_key, daily_key in pairs:
            total_value = float(district[total_key])
            daily_value = float(district[daily_key])
            assert daily_value == pytest.approx(total_value / simulated_days, abs=1e-9)
    finally:
        env.close()


def test_phase_kpis_are_present_only_when_electrical_service_is_enabled():
    env_phase = _run_episode(THREE_PHASE_SCHEMA, seconds_per_time_step=60, episode_steps=8)
    env_legacy = _run_episode(SCHEMA, seconds_per_time_step=60, episode_steps=8)

    try:
        phase_df = env_phase.evaluate()
        legacy_df = env_legacy.evaluate()

        phase_keys = set(phase_df["cost_function"].unique())
        legacy_keys = set(legacy_df["cost_function"].unique())

        assert "phase_import_peak_kw_L1" in phase_keys
        assert "phase_import_peak_kw_L2" in phase_keys
        assert "phase_import_peak_kw_L3" in phase_keys
        assert "electrical_service_violation_total_kwh" in phase_keys

        assert "phase_import_peak_kw_L2" not in legacy_keys
        assert "phase_import_peak_kw_L3" not in legacy_keys
    finally:
        env_phase.close()
        env_legacy.close()


def test_equity_kpis_are_exported_and_bpr_is_none_when_groups_are_incomplete(tmp_path: Path):
    schema_path = _build_schema_with_manual_equity_groups(tmp_path, SCHEMA, missing_first_group=True)
    env = _run_episode(schema_path, seconds_per_time_step=60, episode_steps=12)

    try:
        df = env.evaluate()
        expected = {
            "equity_relative_benefit_percent",
            "equity_gini_benefit",
            "equity_cr20_benefit",
            "equity_losers_percent",
            "equity_bpr_asset_poor_over_rich",
        }
        assert expected.issubset(set(df["cost_function"].unique()))

        building_rows = df[
            (df["level"] == "building")
            & (df["cost_function"] == "equity_relative_benefit_percent")
        ]
        assert len(building_rows) == len(env.buildings)

        district = df[df["name"] == "District"].set_index("cost_function")["value"]
        assert pd.isna(district["equity_bpr_asset_poor_over_rich"])
    finally:
        env.close()


def test_equity_group_is_loaded_from_schema(tmp_path: Path):
    schema_path = _build_schema_with_manual_equity_groups(tmp_path, SCHEMA, missing_first_group=False)
    env = CityLearnEnv(
        str(schema_path),
        central_agent=True,
        episode_time_steps=2,
        random_seed=0,
    )

    try:
        env.reset()
        groups = [getattr(building, "equity_group", None) for building in env.buildings]
        assert all(group in {"asset_rich", "asset_poor"} for group in groups)
    finally:
        env.close()


def test_extended_cost_and_equity_use_raw_cost_series(tmp_path: Path):
    schema_path = _build_two_building_market_schema(tmp_path)
    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=2, random_seed=0)

    try:
        env.reset()
        env.step([np.zeros(len(env.action_names[0]), dtype="float32")])

        control = SimpleNamespace(value="_test_control")
        baseline = SimpleNamespace(value="_test_baseline")

        control_cost = np.array([-2.0, 1.0], dtype="float64")
        baseline_cost = np.array([1.0, 1.0], dtype="float64")
        control_net = np.array([0.5, 0.5], dtype="float64")
        baseline_net = np.array([1.0, 1.0], dtype="float64")
        zeros = np.array([0.0, 0.0], dtype="float64")

        for building in env.buildings:
            setattr(building, "net_electricity_consumption_test_control", control_net.copy())
            setattr(building, "net_electricity_consumption_test_baseline", baseline_net.copy())
            setattr(building, "net_electricity_consumption_emission_test_control", zeros.copy())
            setattr(building, "net_electricity_consumption_emission_test_baseline", zeros.copy())
            setattr(building, "net_electricity_consumption_cost_test_control", control_cost.copy())
            setattr(building, "net_electricity_consumption_cost_test_baseline", baseline_cost.copy())

        env_count = len(env.buildings)
        setattr(env, "net_electricity_consumption_test_control", control_net * env_count)
        setattr(env, "net_electricity_consumption_test_baseline", baseline_net * env_count)
        setattr(env, "net_electricity_consumption_emission_test_control", zeros.copy())
        setattr(env, "net_electricity_consumption_emission_test_baseline", zeros.copy())
        setattr(env, "net_electricity_consumption_cost_test_control", control_cost * env_count)
        setattr(env, "net_electricity_consumption_cost_test_baseline", baseline_cost * env_count)

        df = env.evaluate(control_condition=control, baseline_condition=baseline)
        building_name = env.buildings[0].name
        building_df = df[df["name"] == building_name].set_index("cost_function")["value"]
        district_df = df[df["name"] == "District"].set_index("cost_function")["value"]

        assert float(building_df["cost_control_total_eur"]) == pytest.approx(-1.0)
        assert float(building_df["cost_baseline_total_eur"]) == pytest.approx(2.0)
        assert float(building_df["cost_delta_total_eur"]) == pytest.approx(-3.0)
        assert float(building_df["equity_relative_benefit_percent"]) == pytest.approx(150.0)

        # Legacy normalized cost still uses clipped CostFunction.cost semantics.
        assert float(building_df["cost_total"]) == pytest.approx(0.5)

        assert float(district_df["cost_control_total_eur"]) == pytest.approx(-2.0)
        assert float(district_df["cost_baseline_total_eur"]) == pytest.approx(4.0)
        assert float(district_df["cost_delta_total_eur"]) == pytest.approx(-6.0)
    finally:
        env.close()
