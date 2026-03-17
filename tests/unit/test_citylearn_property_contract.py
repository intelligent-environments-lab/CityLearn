from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[2] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def test_without_storage_district_series_are_consistent_ndarrays():
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=6,
        render_mode="none",
        random_seed=0,
    )

    try:
        env.reset()
        emissions = env.net_electricity_consumption_emission_without_storage
        costs = env.net_electricity_consumption_cost_without_storage
        energy = env.net_electricity_consumption_without_storage

        assert isinstance(emissions, np.ndarray)
        assert isinstance(costs, np.ndarray)
        assert isinstance(energy, np.ndarray)
        assert emissions.shape == costs.shape == energy.shape
    finally:
        env.close()
