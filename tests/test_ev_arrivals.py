import numpy as np
import pandas as pd
import pytest

from glob import glob

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA_PATH = "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _zero_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def _expected_arrival_soc(csv_path: str, transition_index: int, from_state: int) -> float:
    df = pd.read_csv(csv_path)

    cap = df.loc[transition_index, "electric_vehicle_battery_capacity_khw"]
    if pd.isna(cap) and transition_index + 1 < len(df):
        cap = df.loc[transition_index + 1, "electric_vehicle_battery_capacity_khw"]
    cap = float(cap)
    assert cap > 0, "Battery capacity must be positive in dataset."

    if from_state == 2:
        estimated = df.loc[transition_index, "electric_vehicle_estimated_soc_arrival"]
        if not pd.isna(estimated) and estimated >= 0:
            return float(estimated) / 100.0
    else:
        estimated = df.loc[transition_index + 1, "electric_vehicle_estimated_soc_arrival"]
        if not pd.isna(estimated) and estimated >= 0:
            return float(estimated) / 100.0

    # Fallback to current SOC recorded in kWh.
    current_soc_kwh = df.loc[transition_index + 1, "current_soc"]
    assert not pd.isna(current_soc_kwh), "Current SOC must be available when estimate is missing."
    return float(current_soc_kwh) / cap


def _find_transition(from_state: int):
    for csv_path in glob("data/datasets/citylearn_challenge_2022_phase_all_plus_evs/charger_*_*.csv"):
        df = pd.read_csv(csv_path)
        for idx in range(len(df) - 1):
            if (
                df.loc[idx, "electric_vehicle_charger_state"] == from_state
                and df.loc[idx + 1, "electric_vehicle_charger_state"] == 1
            ):
                return csv_path, idx

    raise AssertionError(f"No {from_state}->1 transition found in dataset.")


@pytest.mark.parametrize("from_state", [2, 3])
def test_ev_soc_matches_dataset_on_arrival(from_state: int):
    csv_path, transition_index = _find_transition(from_state)
    charger_id = csv_path.split("/")[-1].replace(".csv", "")

    env = CityLearnEnv(SCHEMA_PATH, central_agent=True, random_seed=0)
    env.reset()

    # Advance the environment to the timestep of interest.
    for _ in range(transition_index + 1):
        env.step(_zero_actions(env))

    # Identify the EV connected to the selected charger.
    target_ev_id = None

    for building in env.buildings:
        for charger in building.electric_vehicle_chargers or []:
            if charger.charger_id == charger_id:
                sim = charger.charger_simulation
                target_ev_id = sim.electric_vehicle_id[transition_index + 1]
                break
        if target_ev_id:
            break

    assert target_ev_id, "Expected EV id for charger transition not found."

    ev = next(ev for ev in env.electric_vehicles if ev.name == target_ev_id)

    expected_soc = _expected_arrival_soc(csv_path, transition_index, from_state)

    assert pytest.approx(expected_soc, abs=1e-6) == float(ev.battery.soc[env.time_step])


def test_ev_kpi_evaluation_with_evs_and_chargers():
    env = CityLearnEnv(SCHEMA_PATH, central_agent=True, random_seed=0)
    env.reset()

    for _ in range(10):
        env.step(_zero_actions(env))

    df = env.evaluate()

    district_values = df[df["level"] == "district"]["value"]
    assert district_values.notna().any(), "District-level KPI values should contain finite entries when EVs are present."
