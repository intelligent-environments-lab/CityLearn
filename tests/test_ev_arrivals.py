import numpy as np
import pandas as pd
import pytest

from glob import glob

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA_PATH = "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _zero_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


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

    step = env.time_step
    prev_state = sim.electric_vehicle_charger_state[step - 1] if step > 0 else float("nan")
    prev_ev_id = sim.electric_vehicle_id[step - 1] if step > 0 else None

    candidate_index = None
    if prev_state in (2, 3):
        candidate_index = step - 1
    elif 0 <= step < len(sim.electric_vehicle_estimated_soc_arrival):
        candidate_index = step

    estimated_soc = None
    if candidate_index is not None and 0 <= candidate_index < len(sim.electric_vehicle_estimated_soc_arrival):
        candidate_value = sim.electric_vehicle_estimated_soc_arrival[candidate_index]
        if isinstance(candidate_value, (float, np.floating)) and not np.isnan(candidate_value) and candidate_value >= 0:
            estimated_soc = float(candidate_value)

    if estimated_soc is None:
        fallback_index = step if step < len(sim.electric_vehicle_required_soc_departure) else step - 1
        fallback_value = sim.electric_vehicle_required_soc_departure[fallback_index]
        assert (
            isinstance(fallback_value, (float, np.floating))
            and not np.isnan(fallback_value)
            and fallback_value >= 0
        ), "Expected fallback SOC from required departure."
        expected_soc = float(fallback_value)
    else:
        expected_soc = estimated_soc

    assert pytest.approx(expected_soc, abs=1e-6) == float(ev.battery.soc[env.time_step])


def test_ev_kpi_evaluation_with_evs_and_chargers():
    env = CityLearnEnv(SCHEMA_PATH, central_agent=True, random_seed=0)
    env.reset()

    for _ in range(10):
        env.step(_zero_actions(env))

    df = env.evaluate()

    district_values = df[df["level"] == "district"]["value"]
    assert district_values.notna().any(), "District-level KPI values should contain finite entries when EVs are present."


def test_ev_current_soc_overrides_arrival_estimate_when_present():
    csv_path, transition_index = _find_transition(2)
    charger_id = csv_path.split("/")[-1].replace(".csv", "")

    env = CityLearnEnv(SCHEMA_PATH, central_agent=True, random_seed=0)
    env.reset()

    target_charger = None
    for building in env.buildings:
        for charger in building.electric_vehicle_chargers or []:
            if charger.charger_id == charger_id:
                target_charger = charger
                break
        if target_charger is not None:
            break

    assert target_charger is not None, "Expected charger for transition was not found."
    sim = target_charger.charger_simulation
    forced_soc = 0.42
    current_soc = np.full(len(sim.electric_vehicle_charger_state), -0.1, dtype="float32")
    current_soc[transition_index + 1] = forced_soc
    sim.electric_vehicle_current_soc = current_soc

    for _ in range(transition_index + 1):
        env.step(_zero_actions(env))

    connected_ev = target_charger.connected_electric_vehicle
    assert connected_ev is not None, "Expected EV to be connected at the transition step."
    assert float(connected_ev.battery.soc[env.time_step]) == pytest.approx(forced_soc, abs=1e-6)
