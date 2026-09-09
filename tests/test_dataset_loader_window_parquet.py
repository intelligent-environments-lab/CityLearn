import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pyarrow")

from citylearn.citylearn import CityLearnEnv


SOURCE_DATASET = Path(__file__).resolve().parent / "data/minute_ev_demo"
DEFERRABLE_OBSERVATIONS = [
    "deferrable_appliance_pending",
    "deferrable_appliance_running",
    "deferrable_appliance_can_start",
    "deferrable_appliance_deadline_missed",
    "deferrable_appliance_earliest_start_time_step",
    "deferrable_appliance_latest_start_time_step",
    "deferrable_appliance_deadline_time_step",
    "deferrable_appliance_hours_until_latest_start",
    "deferrable_appliance_hours_until_deadline",
    "deferrable_appliance_slack_steps",
    "deferrable_appliance_slack_ratio",
    "deferrable_appliance_urgency_ratio",
    "deferrable_appliance_cycle_duration_steps",
    "deferrable_appliance_cycle_energy_kwh",
    "deferrable_appliance_remaining_energy_kwh",
    "deferrable_appliance_current_step_energy_kwh",
    "deferrable_appliance_priority",
    "deferrable_appliance_must_run",
    "deferrable_appliance_cycle_average_power_kw",
    "deferrable_appliance_cycle_peak_power_kw",
    "deferrable_appliance_cycle_load_factor_ratio",
    "deferrable_appliance_cycle_peak_step_offset_ratio",
    "deferrable_appliance_remaining_duration_steps",
    "deferrable_appliance_remaining_average_power_kw",
    "deferrable_appliance_current_step_power_kw",
]


def _copy_minute_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "minute_ev_demo"
    shutil.copytree(SOURCE_DATASET, dataset_dir)

    charger_path = dataset_dir / "charger_1_1.csv"
    charger_frame = pd.read_csv(charger_path)
    charger_frame["electric_vehicle_current_soc"] = [
        0.50, 0.525, 0.55, 0.575, np.nan, np.nan, np.nan, np.nan
    ]
    charger_frame.to_csv(charger_path, index=False)

    pd.DataFrame(
        [
            {
                "profile_id": "quick",
                "duration_steps": 2,
                "total_energy_kwh": 0.3,
                "load_profile": "[0.1,0.2]",
            },
            {
                "profile_id": "short",
                "duration_steps": 1,
                "total_energy_kwh": 0.3,
                "load_profile": "[0.3]",
            },
        ]
    ).to_csv(dataset_dir / "deferrable_appliance_1_cycle_profiles.csv", index=False)
    pd.DataFrame(
        [
            {
                "cycle_id": "cycle_1",
                "profile_id": "quick",
                "earliest_start_time_step": 2,
                "latest_start_time_step": 3,
                "deadline_time_step": 4,
                "priority": 0.7,
                "must_run": True,
            },
            {
                "cycle_id": "cycle_2",
                "profile_id": "short",
                "earliest_start_time_step": 5,
                "latest_start_time_step": 5,
                "deadline_time_step": 5,
                "priority": 1.0,
                "must_run": True,
            },
        ]
    ).to_csv(dataset_dir / "deferrable_appliance_1_flexibility_schedule.csv", index=False)

    schema_path = dataset_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["root_directory"] = str(dataset_dir)
    for observation in DEFERRABLE_OBSERVATIONS:
        schema["observations"][observation] = {
            "active": True,
            "shared_in_central_agent": True,
        }
    schema["actions"]["deferrable_appliance"] = {"active": True}
    schema["buildings"]["Building_1"]["deferrable_appliances"] = {
        "deferrable_appliance_1": {
            "type": "citylearn.energy_model.DeferrableAppliance",
            "autosize": False,
            "cycle_profiles_file": "deferrable_appliance_1_cycle_profiles.csv",
            "flexibility_schedule_file": "deferrable_appliance_1_flexibility_schedule.csv",
        }
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema_path


def _convert_schema_sources_to_parquet(schema_path: Path) -> Path:
    dataset_dir = schema_path.parent
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    converted = copy.deepcopy(schema)

    def convert(filename: str) -> str:
        source = dataset_dir / filename
        target = source.with_suffix(".parquet")
        pd.read_csv(source).to_parquet(target, index=False)
        return target.name

    for building in converted["buildings"].values():
        building["energy_simulation"] = convert(building["energy_simulation"])
        building["weather"] = convert(building["weather"])
        building["carbon_intensity"] = convert(building["carbon_intensity"])
        building["pricing"] = convert(building["pricing"])

        for charger in (building.get("chargers") or {}).values():
            charger["charger_simulation"] = convert(charger["charger_simulation"])

        for appliance in (building.get("deferrable_appliances") or {}).values():
            appliance["cycle_profiles_file"] = convert(appliance["cycle_profiles_file"])
            appliance["flexibility_schedule_file"] = convert(appliance["flexibility_schedule_file"])

    parquet_schema_path = dataset_dir / "schema_parquet.json"
    parquet_schema_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    return parquet_schema_path


def _zero_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def _rollout_trace(schema_path: Path):
    env = CityLearnEnv(
        str(schema_path),
        central_agent=True,
        simulation_start_time_step=2,
        simulation_end_time_step=6,
        episode_time_steps=3,
        rolling_episode_split=False,
        random_episode_split=False,
        random_seed=0,
    )

    try:
        observations, _ = env.reset(seed=0)
        building = env.buildings[0]

        raw_energy_rows = len(building.energy_simulation.__dict__["_month"])
        raw_weather_rows = len(building.weather.__dict__["_outdoor_dry_bulb_temperature"])
        raw_pricing_rows = len(building.pricing.__dict__["_electricity_pricing"])
        raw_carbon_rows = len(building.carbon_intensity.__dict__["_carbon_intensity"])
        raw_charger_rows = len(
            building.electric_vehicle_chargers[0].charger_simulation.__dict__["_electric_vehicle_charger_state"]
        )
        raw_profile_rows = len(building.deferrable_appliances[0].deferrable_appliance_simulation.cycle_profiles)
        raw_schedule_rows = len(building.deferrable_appliances[0].deferrable_appliance_simulation.flexibility_schedule)

        trace = {
            "reset_observations": np.asarray(observations[0], dtype="float64"),
            "raw_rows": (
                raw_energy_rows,
                raw_weather_rows,
                raw_pricing_rows,
                raw_carbon_rows,
                raw_charger_rows,
                raw_profile_rows,
                raw_schedule_rows,
            ),
            "episode_starts": [env.episode_tracker.episode_start_time_step],
            "reset_ev_soc": float(
                building.electric_vehicle_chargers[0]
                .connected_electric_vehicle.battery.soc[env.time_step]
            ),
            "net": [],
            "cost": [],
            "emission": [],
            "charger_state": [],
            "deferrable_earliest_start": [],
        }

        for _ in range(2):
            observations, *_ = env.step(_zero_actions(env))
            t = env.time_step - 1
            trace["net"].append(float(building.net_electricity_consumption[t]))
            trace["cost"].append(float(building.net_electricity_consumption_cost[t]))
            trace["emission"].append(float(building.net_electricity_consumption_emission[t]))
            trace["charger_state"].append(
                float(building.electric_vehicle_chargers[0].charger_simulation.electric_vehicle_charger_state[t])
            )
            trace["deferrable_earliest_start"].append(
                int(building.deferrable_appliances[0].observations()["earliest_start_time_step"])
            )

        env.reset(seed=0)
        trace["episode_starts"].append(env.episode_tracker.episode_start_time_step)
        return trace
    finally:
        env.close()


def test_windowed_loader_reads_only_simulation_window_and_parquet_matches_csv(tmp_path: Path):
    csv_schema_path = _copy_minute_dataset(tmp_path)
    parquet_schema_path = _convert_schema_sources_to_parquet(csv_schema_path)

    csv_trace = _rollout_trace(csv_schema_path)
    parquet_trace = _rollout_trace(parquet_schema_path)

    assert csv_trace["raw_rows"] == (5, 5, 5, 5, 5, 2, 2)
    assert parquet_trace["raw_rows"] == csv_trace["raw_rows"]
    assert parquet_trace["episode_starts"] == csv_trace["episode_starts"]
    assert csv_trace["reset_ev_soc"] == pytest.approx(0.55)
    assert parquet_trace["reset_ev_soc"] == pytest.approx(0.55)
    np.testing.assert_allclose(parquet_trace["reset_observations"], csv_trace["reset_observations"])
    np.testing.assert_allclose(parquet_trace["net"], csv_trace["net"])
    np.testing.assert_allclose(parquet_trace["cost"], csv_trace["cost"])
    np.testing.assert_allclose(parquet_trace["emission"], csv_trace["emission"])
    assert parquet_trace["charger_state"] == pytest.approx(csv_trace["charger_state"])
    assert parquet_trace["deferrable_earliest_start"] == csv_trace["deferrable_earliest_start"]


def test_shared_weather_pricing_and_carbon_sources_are_cached_between_buildings(tmp_path: Path):
    schema_path = _copy_minute_dataset(tmp_path)
    dataset_dir = schema_path.parent
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    schema["buildings"]["Building_2"] = copy.deepcopy(schema["buildings"]["Building_1"])
    schema["buildings"]["Building_2"]["energy_simulation"] = "Building_2.csv"
    schema["buildings"]["Building_2"].pop("chargers", None)
    schema["buildings"]["Building_2"].pop("deferrable_appliances", None)
    pd.read_csv(dataset_dir / "Building_1.csv").to_csv(dataset_dir / "Building_2.csv", index=False)
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=3, random_seed=0)

    try:
        env.reset(seed=0)
        assert env.buildings[0].weather is env.buildings[1].weather
        assert env.buildings[0].pricing is env.buildings[1].pricing
        assert env.buildings[0].carbon_intensity is env.buildings[1].carbon_intensity
        assert env.buildings[0].energy_simulation is not env.buildings[1].energy_simulation
    finally:
        env.close()
