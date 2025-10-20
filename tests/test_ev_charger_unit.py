import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.base import EpisodeTracker
from citylearn.data import ChargerSimulation
from citylearn.energy_model import Battery
from citylearn.electric_vehicle import ElectricVehicle
from citylearn.electric_vehicle_charger import Charger


def _make_tracker(length: int = 3) -> EpisodeTracker:
    tracker = EpisodeTracker(0, length - 1)
    tracker.next_episode(length, False, False, 0)
    return tracker


def _make_simulation(length: int, ev_id: str) -> ChargerSimulation:
    return ChargerSimulation(
        electric_vehicle_charger_state=[1] * length,
        electric_vehicle_id=[ev_id] * length,
        electric_vehicle_battery_capacity_khw=[100.0] * length,
        current_soc=[50.0] * length,
        electric_vehicle_departure_time=[-1] * length,
        electric_vehicle_required_soc_departure=[-0.1] * length,
        electric_vehicle_estimated_arrival_time=[-1] * length,
        electric_vehicle_estimated_soc_arrival=[-0.1] * length,
    )


def _make_ev(tracker: EpisodeTracker, initial_soc: float = 0.5) -> ElectricVehicle:
    battery = Battery(
        capacity=100.0,
        nominal_power=50.0,
        initial_soc=initial_soc,
        efficiency=1.0,
        capacity_loss_coefficient=0.0,
        power_efficiency_curve=[[0.0, 1.0], [1.0, 1.0]],
        capacity_power_curve=[[0.0, 1.0], [1.0, 1.0]],
        loss_coefficient=0.0,
        seconds_per_time_step=3600,
        episode_tracker=tracker,
    )
    ev = ElectricVehicle(episode_tracker=tracker, battery=battery, seconds_per_time_step=3600)
    ev.reset()
    return ev


def _make_charger(tracker: EpisodeTracker, sim: ChargerSimulation, ev: ElectricVehicle, **kwargs) -> Charger:
    charger = Charger(
        episode_tracker=tracker,
        charger_simulation=sim,
        charger_id="test",
        max_charging_power=kwargs.get("max_charging_power", 10.0),
        max_discharging_power=kwargs.get("max_discharging_power", 10.0),
        efficiency=kwargs.get("efficiency", 1.0),
        connected_electric_vehicle=ev,
        seconds_per_time_step=3600,
    )
    charger.reset()
    charger.connected_electric_vehicle = ev
    return charger


def test_charger_charging_updates_ev_soc_and_accounting():
    tracker = _make_tracker()
    ev = _make_ev(tracker)
    sim = _make_simulation(3, ev.name)
    charger = _make_charger(tracker, sim, ev, efficiency=0.9)

    initial_soc = ev.battery.soc[ev.time_step]
    charger.update_connected_electric_vehicle_soc(0.5)

    energy_balance = ev.battery.energy_balance[ev.time_step]
    electricity_consumption = charger.electricity_consumption[charger.time_step]

    expected_soc = (initial_soc * ev.battery.capacity + energy_balance) / ev.battery.capacity

    assert np.isclose(ev.battery.soc[ev.time_step], expected_soc)
    assert np.isclose(energy_balance, electricity_consumption * charger.efficiency)


def test_charger_discharging_updates_ev_soc_and_accounting():
    tracker = _make_tracker()
    ev = _make_ev(tracker)
    sim = _make_simulation(3, ev.name)
    charger = _make_charger(tracker, sim, ev, efficiency=0.85)

    initial_soc = ev.battery.soc[ev.time_step]
    charger.update_connected_electric_vehicle_soc(-0.4)

    energy_balance = ev.battery.energy_balance[ev.time_step]
    electricity_consumption = charger.electricity_consumption[charger.time_step]

    expected_soc = (initial_soc * ev.battery.capacity + energy_balance) / ev.battery.capacity

    assert np.isclose(ev.battery.soc[ev.time_step], expected_soc)
    assert np.isclose(energy_balance, electricity_consumption / charger.efficiency)


def test_charger_action_without_connected_vehicle_has_no_effect():
    tracker = _make_tracker()
    ev = _make_ev(tracker)
    sim = _make_simulation(3, ev.name)
    charger = _make_charger(tracker, sim, ev)

    charger.connected_electric_vehicle = None
    charger.update_connected_electric_vehicle_soc(0.8)

    assert np.allclose(charger.electricity_consumption, 0.0)
    expected_commanded_energy = 0.8 * charger.max_charging_power * charger.algorithm_action_based_time_step_hours_ratio
    assert np.isclose(
        charger.past_charging_action_values_kwh[charger.time_step],
        expected_commanded_energy,
    )
