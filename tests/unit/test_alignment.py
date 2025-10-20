"""Tests for timestep alignment and EV charger integration."""

from pathlib import Path
import math
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


DATASET = Path(__file__).resolve().parents[2] / 'data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'


def _finite(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.all(np.isfinite(value))

    return math.isfinite(float(value))


def test_timestep_and_chargers():
    env = CityLearnEnv(str(DATASET), central_agent=True)

    try:
        obs, _ = env.reset()
        zeros = np.zeros(env.action_space[0].shape[0], dtype='float32')

        obs, r0, term, trunc, _ = env.step([zeros])
        assert env.time_step == 1
        assert _finite(r0)

        names = env.action_names[0]
        charger_index = next((i for i, name in enumerate(names) if name.startswith('electric_vehicle_storage_')), None)

        if charger_index is None:
            pytest.skip('Dataset does not expose EV storage actions.')

        actions = np.zeros_like(zeros)
        actions[charger_index] = 0.1

        obs, r1, term, trunc, _ = env.step([actions])
        assert env.time_step == 2
        assert _finite(r1)

        building = env.buildings[0]
        t = env.time_step - 1
        lhs = building.net_electricity_consumption[t]
        rhs = (
            building.cooling_electricity_consumption[t]
            + building.heating_electricity_consumption[t]
            + building.dhw_electricity_consumption[t]
            + building.non_shiftable_load_electricity_consumption[t]
            + building.electrical_storage_electricity_consumption[t]
            + building.solar_generation[t]
            + building.chargers_electricity_consumption[t]
            + building.washing_machines_electricity_consumption[t]
        )
        assert abs(lhs - rhs) < 1e-4

    finally:
        env.close()
