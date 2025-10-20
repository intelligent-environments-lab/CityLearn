"""Tests verifying sub-hour simulation support."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.base import EpisodeTracker
from citylearn.citylearn import CityLearnEnv
from citylearn.data import EnergySimulation
from citylearn.energy_model import StorageDevice


def _make_tracker(length: int) -> EpisodeTracker:
    tracker = EpisodeTracker(0, length - 1)
    tracker.next_episode(length, rolling_episode_split=False, random_episode_split=False, random_seed=0)
    return tracker


def test_energy_simulation_ratio_matches_dataset_cadence():
    seconds_per_time_step = 900  # 15 minutes
    num_steps = 4
    minutes = [0, 15, 30, 45]

    sim = EnergySimulation(
        month=[1] * num_steps,
        hour=[0] * num_steps,
        minutes=minutes,
        day_type=[1] * num_steps,
        indoor_dry_bulb_temperature=[20.0] * num_steps,
        non_shiftable_load=[1.0] * num_steps,
        dhw_demand=[0.0] * num_steps,
        cooling_demand=[0.0] * num_steps,
        heating_demand=[0.0] * num_steps,
        solar_generation=[0.0] * num_steps,
        seconds_per_time_step=seconds_per_time_step,
        time_step_ratios=[],
    )

    assert sim.time_step_ratios[0] == pytest.approx(1.0)


def test_energy_simulation_ratio_subhour_control_from_hourly_dataset():
    seconds_per_time_step = 900  # 15 minutes control step
    num_steps = 4

    sim = EnergySimulation(
        month=[1] * num_steps,
        hour=[0, 1, 2, 3],
        day_type=[1] * num_steps,
        indoor_dry_bulb_temperature=[20.0] * num_steps,
        non_shiftable_load=[1.0] * num_steps,
        dhw_demand=[0.0] * num_steps,
        cooling_demand=[0.0] * num_steps,
        heating_demand=[0.0] * num_steps,
        solar_generation=[0.0] * num_steps,
        seconds_per_time_step=seconds_per_time_step,
        time_step_ratios=[],
    )

    # Hourly dataset, 15-minute control -> ratio should be 0.25
    assert sim.time_step_ratios[0] == pytest.approx(0.25)


def test_storage_charge_scaling_respects_time_ratio():
    tracker = _make_tracker(4)
    storage = StorageDevice(
        capacity=10.0,
        efficiency=1.0,
        loss_coefficient=0.0,
        initial_soc=0.0,
        time_step_ratio=0.25,
        episode_tracker=tracker,
        seconds_per_time_step=900,
    )
    storage.reset()

    energy_actual = 2.5  # kWh to add over a 15-minute step at full power
    dataset_energy = energy_actual / storage.time_step_ratio
    storage.charge(dataset_energy)

    assert storage.energy_balance[0] == pytest.approx(energy_actual)


def test_env_supports_subhour_seconds_per_time_step():
    schema = 'data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'
    env = CityLearnEnv(
        schema,
        central_agent=True,
        episode_time_steps=2,
        seconds_per_time_step=900,
    )

    obs, _ = env.reset()
    assert env.seconds_per_time_step == 900
    assert obs is not None

    zeros = [np.zeros(env.action_space[0].shape[0], dtype='float32')]
    env.step(zeros)
    env.close()
