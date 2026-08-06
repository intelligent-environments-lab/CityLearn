"""Tests verifying that storage behaviour does not depend on the length of a time step."""

import pytest

pytest.importorskip("gymnasium")

from citylearn.base import EpisodeTracker
from citylearn.energy_model import Battery, StorageDevice, StorageTank

FLAT_CURVE = [[0.0, 1.0], [1.0, 1.0]]
HALF_RATING_CURVE = [[0.0, 0.5], [1.0, 0.5]]
QUARTER_HOUR = 900


def _make_tracker(length: int) -> EpisodeTracker:
    tracker = EpisodeTracker(0, length - 1)
    tracker.next_episode(length, rolling_episode_split=False, random_episode_split=False, random_seed=0)

    return tracker


def _make_battery(capacity_power_curve=None, power_efficiency_curve=None) -> Battery:
    battery = Battery(
        capacity=1000.0,
        nominal_power=50.0,
        initial_soc=0.0,
        efficiency=1.0,
        loss_coefficient=0.0,
        capacity_loss_coefficient=0.0,
        depth_of_discharge=1.0,
        power_efficiency_curve=FLAT_CURVE if power_efficiency_curve is None else power_efficiency_curve,
        capacity_power_curve=FLAT_CURVE if capacity_power_curve is None else capacity_power_curve,
        seconds_per_time_step=QUARTER_HOUR,
        episode_tracker=_make_tracker(8),
    )
    battery.reset()

    return battery


def test_standby_loss_is_deducted_per_hour_not_per_step():
    storage = StorageDevice(
        capacity=100.0,
        efficiency=1.0,
        loss_coefficient=0.01,
        initial_soc=1.0,
        seconds_per_time_step=QUARTER_HOUR,
        episode_tracker=_make_tracker(8),
    )
    storage.reset()

    for _ in range(4):
        storage.charge(0.0)
        storage.next_time_step()

    # An hour of standing still costs the same charge however finely it is sliced. Deducting
    # the hourly loss on every step instead leaves 0.99**4 = 0.9606.
    assert storage.soc[3] == pytest.approx(0.99)


def test_efficiency_curve_is_read_at_the_fraction_of_rated_power():
    battery = _make_battery(power_efficiency_curve=[[0.0, 0.5], [1.0, 1.0]])
    battery.charge(50.0*0.25)  # the full 50 kW rating, for a quarter of an hour

    # power_efficiency_curve is indexed by fraction of rated power, so this is a fraction of
    # 1.0. Dividing the step's kWh by a kW rating reads the curve at 0.25 and gives 0.625.
    assert battery.efficiency == pytest.approx(1.0)


def test_capacity_power_curve_limits_power_not_step_energy():
    battery = _make_battery(capacity_power_curve=HALF_RATING_CURVE)
    battery.charge(100.0)

    # The taper caps a rate in kW, so a quarter-hour step may take a quarter of it. Comparing
    # that limit against kWh directly lets 25.0 kWh through.
    assert battery.energy_balance[0] == pytest.approx(50.0*0.5*0.25)


def test_nominal_power_limits_power_not_step_energy():
    battery = _make_battery()
    battery.charge(100.0)

    # Likewise for nominal_power. Treating the rating as an energy allowance lets 50.0 kWh
    # through in a quarter of an hour.
    assert battery.energy_balance[0] == pytest.approx(50.0*0.25)


def test_storage_tank_max_input_power_limits_power():
    tank = StorageTank(
        capacity=100.0,
        efficiency=1.0,
        loss_coefficient=0.0,
        initial_soc=0.0,
        max_input_power=10.0,
        seconds_per_time_step=QUARTER_HOUR,
        episode_tracker=_make_tracker(8),
    )
    tank.reset()
    tank.charge(100.0)

    # max_input_power is in kW too, and was being applied to the step's energy.
    assert tank.energy_balance[0] == pytest.approx(10.0*0.25)
