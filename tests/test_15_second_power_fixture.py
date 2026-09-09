import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SECONDS_PER_STEP = 15


def _power_kw_to_energy_kwh(power_kw: np.ndarray) -> np.ndarray:
    return np.asarray(power_kw, dtype="float64") * SECONDS_PER_STEP / 3600.0


def _write_15_second_power_schema(tmp_path: Path):
    steps = 6
    dataset_dir = tmp_path / "real_power_15s"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    elapsed_seconds = np.arange(steps, dtype=int) * SECONDS_PER_STEP
    hours = ((elapsed_seconds // 3600) % 24).astype(int)
    minutes = ((elapsed_seconds % 3600) // 60).astype(int)
    seconds = (elapsed_seconds % 60).astype(int)

    load_kw = np.array([55.0, 60.0, 52.0, 58.0, 57.0, 54.0], dtype="float64")
    pv_kw = np.array([3.0, 4.0, 1.0, 0.0, 5.0, 2.0], dtype="float64")
    prices = np.array([0.10, 0.10, 0.20, 0.20, 0.30, 0.30], dtype="float64")
    carbon = np.array([0.40, 0.40, 0.35, 0.35, 0.30, 0.30], dtype="float64")

    load_kwh = _power_kw_to_energy_kwh(load_kw)
    pv_kwh = _power_kw_to_energy_kwh(pv_kw)

    building = pd.DataFrame(
        {
            "month": np.ones(steps, dtype=int),
            "hour": hours,
            "minutes": minutes,
            "seconds": seconds,
            "day_type": np.ones(steps, dtype=int),
            "daylight_savings_status": np.zeros(steps, dtype=int),
            "indoor_dry_bulb_temperature": np.full(steps, 21.0),
            "average_unmet_cooling_setpoint_difference": np.zeros(steps),
            "indoor_relative_humidity": np.full(steps, 45.0),
            "non_shiftable_load": load_kwh,
            "dhw_demand": np.zeros(steps),
            "cooling_demand": np.zeros(steps),
            "heating_demand": np.zeros(steps),
            "solar_generation": pv_kwh,
        }
    )
    building.to_csv(dataset_dir / "Building_1.csv", index=False)

    weather = pd.DataFrame(
        {
            "outdoor_dry_bulb_temperature": np.full(steps, 10.0),
            "outdoor_relative_humidity": np.full(steps, 50.0),
            "diffuse_solar_irradiance": np.zeros(steps),
            "direct_solar_irradiance": np.zeros(steps),
            "outdoor_dry_bulb_temperature_predicted_1": np.full(steps, 10.0),
            "outdoor_dry_bulb_temperature_predicted_2": np.full(steps, 10.0),
            "outdoor_dry_bulb_temperature_predicted_3": np.full(steps, 10.0),
            "outdoor_relative_humidity_predicted_1": np.full(steps, 50.0),
            "outdoor_relative_humidity_predicted_2": np.full(steps, 50.0),
            "outdoor_relative_humidity_predicted_3": np.full(steps, 50.0),
            "diffuse_solar_irradiance_predicted_1": np.zeros(steps),
            "diffuse_solar_irradiance_predicted_2": np.zeros(steps),
            "diffuse_solar_irradiance_predicted_3": np.zeros(steps),
            "direct_solar_irradiance_predicted_1": np.zeros(steps),
            "direct_solar_irradiance_predicted_2": np.zeros(steps),
            "direct_solar_irradiance_predicted_3": np.zeros(steps),
        }
    )
    weather.to_csv(dataset_dir / "weather.csv", index=False)
    pd.DataFrame({"carbon_intensity": carbon}).to_csv(dataset_dir / "carbon_intensity.csv", index=False)
    pd.DataFrame(
        {
            "electricity_pricing": prices,
            "electricity_pricing_predicted_1": prices,
            "electricity_pricing_predicted_2": prices,
            "electricity_pricing_predicted_3": prices,
        }
    ).to_csv(dataset_dir / "pricing.csv", index=False)

    schema = {
        "random_seed": 0,
        "root_directory": str(dataset_dir),
        "central_agent": True,
        "simulation_start_time_step": 0,
        "simulation_end_time_step": steps - 1,
        "episode_time_steps": steps,
        "rolling_episode_split": False,
        "random_episode_split": False,
        "seconds_per_time_step": SECONDS_PER_STEP,
        "observations": {
            "month": {"active": True, "shared_in_central_agent": True},
            "hour": {"active": True, "shared_in_central_agent": True},
            "minutes": {"active": True, "shared_in_central_agent": True},
            "seconds": {"active": True, "shared_in_central_agent": True},
            "day_type": {"active": True, "shared_in_central_agent": True},
            "non_shiftable_load": {"active": True, "shared_in_central_agent": False},
            "solar_generation": {"active": True, "shared_in_central_agent": False},
            "net_electricity_consumption": {"active": True, "shared_in_central_agent": False},
            "electricity_pricing": {"active": True, "shared_in_central_agent": True},
            "carbon_intensity": {"active": True, "shared_in_central_agent": True},
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
                    "attributes": {"nominal_power": 120.0, "generation_mode": "absolute"},
                },
            }
        },
    }

    schema_path = dataset_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema_path, {
        "load_kwh": load_kwh,
        "pv_kwh": pv_kwh,
        "prices": prices,
        "carbon": carbon,
        "steps": steps,
    }


def test_15_second_power_fixture_uses_kwh_per_step_without_conversion_warning(tmp_path: Path, capsys):
    schema_path, expected = _write_15_second_power_schema(tmp_path)
    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=expected["steps"], random_seed=0)

    try:
        captured = capsys.readouterr()
        assert "[CityLearn][unit-conversion]" not in captured.out

        env.reset()
        building = env.buildings[0]
        assert building.energy_simulation.dataset_seconds_per_time_step == pytest.approx(SECONDS_PER_STEP)
        assert building.time_step_ratio == pytest.approx(1.0)
        assert building.pv.generation_mode == "absolute"

        net_kwh = expected["load_kwh"] - expected["pv_kwh"]
        costs = net_kwh * expected["prices"]
        emissions = net_kwh * expected["carbon"]

        for t in range(expected["steps"] - 1):
            env.step([])

            assert building.non_shiftable_load_electricity_consumption[t] == pytest.approx(expected["load_kwh"][t])
            assert building.solar_generation[t] == pytest.approx(-expected["pv_kwh"][t])
            assert building.net_electricity_consumption[t] == pytest.approx(net_kwh[t])
            assert building.net_electricity_consumption_cost[t] == pytest.approx(costs[t])
            assert building.net_electricity_consumption_emission[t] == pytest.approx(emissions[t])
            assert env.net_electricity_consumption[t] == pytest.approx(net_kwh[t])
            assert env.net_electricity_consumption_cost[t] == pytest.approx(costs[t])
            assert env.net_electricity_consumption_emission[t] == pytest.approx(emissions[t])
    finally:
        env.close()
