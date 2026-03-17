"""Tests verifying sub-hour simulation support."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.base import EpisodeTracker
from citylearn.citylearn import CityLearnEnv
from citylearn.data import EnergySimulation
from citylearn.energy_model import Battery, ElectricDevice, StorageDevice


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


def test_energy_simulation_time_step_ratio_default_not_shared_across_instances():
    num_steps = 4

    sim_hourly = EnergySimulation(
        month=[1] * num_steps,
        hour=[0, 1, 2, 3],
        day_type=[1] * num_steps,
        indoor_dry_bulb_temperature=[20.0] * num_steps,
        non_shiftable_load=[1.0] * num_steps,
        dhw_demand=[0.0] * num_steps,
        cooling_demand=[0.0] * num_steps,
        heating_demand=[0.0] * num_steps,
        solar_generation=[0.0] * num_steps,
        seconds_per_time_step=3600,
    )
    sim_subhour = EnergySimulation(
        month=[1] * num_steps,
        hour=[0, 1, 2, 3],
        day_type=[1] * num_steps,
        indoor_dry_bulb_temperature=[20.0] * num_steps,
        non_shiftable_load=[1.0] * num_steps,
        dhw_demand=[0.0] * num_steps,
        cooling_demand=[0.0] * num_steps,
        heating_demand=[0.0] * num_steps,
        solar_generation=[0.0] * num_steps,
        seconds_per_time_step=900,
    )

    assert len(sim_hourly.time_step_ratios) == 1
    assert len(sim_subhour.time_step_ratios) == 1
    assert sim_hourly.time_step_ratios[0] == pytest.approx(1.0)
    assert sim_subhour.time_step_ratios[0] == pytest.approx(0.25)


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


def test_electric_device_set_electricity_consumption_is_absolute():
    tracker = _make_tracker(4)
    device = ElectricDevice(
        nominal_power=10.0,
        time_step_ratio=0.25,
        seconds_per_time_step=900,
        episode_tracker=tracker,
    )
    device.reset()

    # Additive updates use dataset-resolution values when ratio != 1.
    device.update_electricity_consumption(4.0)
    assert device.electricity_consumption[0] == pytest.approx(1.0)

    # Absolute setter must overwrite, not accumulate.
    device.set_electricity_consumption(2.5)
    assert device.electricity_consumption[0] == pytest.approx(2.5)

    device.set_electricity_consumption(0.5)
    assert device.electricity_consumption[0] == pytest.approx(0.5)


def test_battery_electricity_consumption_tracks_energy_balance_in_subhour():
    tracker = _make_tracker(4)
    battery = Battery(
        capacity=100.0,
        nominal_power=50.0,
        initial_soc=0.5,
        efficiency=1.0,
        loss_coefficient=0.0,
        capacity_loss_coefficient=0.0,
        power_efficiency_curve=[[0.0, 1.0], [1.0, 1.0]],
        capacity_power_curve=[[0.0, 1.0], [1.0, 1.0]],
        time_step_ratio=0.25,
        seconds_per_time_step=900,
        episode_tracker=tracker,
    )
    battery.reset()

    # Dataset-resolution command corresponding to 2.5 kWh over a 15-minute step.
    battery.charge(10.0)

    assert battery.energy_balance[0] == pytest.approx(2.5)
    assert battery.electricity_consumption[0] == pytest.approx(2.5)


def test_battery_degradation_uses_step_energy_without_extra_ratio_scaling():
    tracker = _make_tracker(4)
    battery = Battery(
        capacity=100.0,
        nominal_power=50.0,
        initial_soc=0.5,
        efficiency=1.0,
        loss_coefficient=0.0,
        capacity_loss_coefficient=1e-5,
        power_efficiency_curve=[[0.0, 1.0], [1.0, 1.0]],
        capacity_power_curve=[[0.0, 1.0], [1.0, 1.0]],
        time_step_ratio=0.25,
        seconds_per_time_step=900,
        episode_tracker=tracker,
    )
    battery.reset()
    battery.charge(10.0)

    expected = (
        battery.capacity_loss_coefficient
        * battery.capacity
        * abs(battery.energy_balance[0])
        / (2.0 * max(battery.degraded_capacity, 1e-10))
    )
    assert battery.degrade() == pytest.approx(expected)


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
