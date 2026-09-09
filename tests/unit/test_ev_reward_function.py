import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import Electric_Vehicles_Reward_Function


MINUTE_DATASET_DIR = Path(__file__).resolve().parents[1] / "data/minute_ev_demo"


def _reward(weights):
    return Electric_Vehicles_Reward_Function({"central_agent": True}, weights=weights)


def _ev_observation(
    *,
    connected: bool = True,
    battery_soc: float = 0.2,
    required_soc: float = 0.8,
    battery_capacity: float = 10.0,
    hours_until_departure: float = 1.0,
    max_charging_power: float = 1.0,
    last_charged_kwh: float = 0.0,
):
    charger_data = {
        "connected": connected,
        "last_charged_kwh": last_charged_kwh,
    }
    if connected:
        charger_data.update(
            {
                "previous_battery_soc": battery_soc,
                "battery_soc": battery_soc,
                "battery_capacity": battery_capacity,
                "min_capacity": 0.0,
                "required_soc": required_soc,
                "hours_until_departure": hours_until_departure,
                "max_charging_power": max_charging_power,
                "max_discharging_power": 0.0,
            }
        )

    return {
        "net_electricity_consumption": 0.0,
        "electric_vehicles_chargers_dict": {"charger_1": charger_data},
    }


@pytest.mark.parametrize(
    "battery_soc,required_soc,max_charging_power,expected",
    [
        (0.2, 0.8, 1.0, -10.0),
        (1.0, 0.2, 1.0, 0.0),
        (0.5, 0.75, 2.5, 0.0),
        (0.5, 0.75, 2.49, -10.0),
    ],
)
def test_ev_reward_soc_impossible_penalizes_unreachable_deficits_only(
    battery_soc,
    required_soc,
    max_charging_power,
    expected,
):
    reward = _reward(
        {
            "no_car_charging": 0.0,
            "battery_limits": 0.0,
            "soc_impossible": -10.0,
            "soc_under": 0.0,
            "close_soc": 0.0,
            "self_ev_consumption": 0.0,
            "extra_self_production": 0.0,
        }
    )

    value = reward.calculate_ev_penalty(
        _ev_observation(
            battery_soc=battery_soc,
            required_soc=required_soc,
            battery_capacity=10.0,
            hours_until_departure=1.0,
            max_charging_power=max_charging_power,
        ),
        current_reward=0.0,
    )

    assert value == pytest.approx(expected)


@pytest.mark.parametrize("last_charged_kwh,expected", [(0.1, 0.0), (0.11, -5.0), (-0.11, -5.0)])
def test_ev_reward_no_car_charging_penalty_accumulates_for_disconnected_charger(last_charged_kwh, expected):
    reward = _reward(
        {
            "no_car_charging": -5.0,
            "battery_limits": 0.0,
            "soc_impossible": 0.0,
            "soc_under": 0.0,
            "close_soc": 0.0,
            "self_ev_consumption": 0.0,
            "extra_self_production": 0.0,
        }
    )

    value = reward.calculate_ev_penalty(
        _ev_observation(connected=False, last_charged_kwh=last_charged_kwh),
        current_reward=0.0,
    )

    assert value == pytest.approx(expected)


def test_env_step_ev_reward_penalizes_action_when_no_car_is_connected(tmp_path: Path):
    dataset_dir = tmp_path / "minute_ev_reward"
    shutil.copytree(MINUTE_DATASET_DIR, dataset_dir)
    schema_path = dataset_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["root_directory"] = str(dataset_dir)
    schema["reward_function"] = {
        "type": "citylearn.reward_function.Electric_Vehicles_Reward_Function",
        "attributes": {
            "weights": {
                "no_car_charging": -5.0,
                "battery_limits": 0.0,
                "soc_impossible": 0.0,
                "soc_under": 0.0,
                "close_soc": 0.0,
                "self_ev_consumption": 0.0,
                "extra_self_production": 0.0,
            }
        },
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=8, random_seed=0)

    try:
        env.reset()
        for _ in range(4):
            env.step([np.zeros(len(env.action_names[0]), dtype="float32")])

        building = env.buildings[0]
        charger = building.electric_vehicle_chargers[0]
        assert charger.connected_electric_vehicle is None

        actions = np.zeros(len(env.action_names[0]), dtype="float32")
        actions[env.action_names[0].index("electric_vehicle_storage_charger_1_1")] = 1.0
        _, reward, *_ = env.step([actions])

        assert reward[0] < 0.0
        assert charger.electricity_consumption[env.time_step - 1] == pytest.approx(0.0)
        assert charger.past_charging_action_values_kwh[env.time_step - 1] > 0.1
    finally:
        env.close()
