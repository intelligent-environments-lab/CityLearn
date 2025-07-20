import pytest
import numpy as np
from unittest.mock import MagicMock
from citylearn.electric_vehicle import ElectricVehicle
from citylearn.energy_model import Battery
from citylearn.base import EpisodeTracker

@pytest.fixture
def mock_episode_tracker():
    tracker = MagicMock(spec=EpisodeTracker)
    tracker.episode_time_steps = 24
    return tracker

@pytest.fixture
def mock_battery():
    battery = MagicMock(spec=Battery)
    battery.capacity = 100.0
    battery.nominal_power = 10.0
    battery.initial_soc = 0.5
    battery.soc = np.array([0.5] * 24)
    battery.energy_balance = np.zeros(24)
    battery.efficiency = 0.9
    battery.round_trip_efficiency = 0.9**0.5
    battery.time_step = 0
    def force_set_soc_side_effect(new_soc):
        battery.soc[battery.time_step] = new_soc
    battery.force_set_soc.side_effect = force_set_soc_side_effect
    return battery

@pytest.fixture
def electric_vehicle(mock_episode_tracker, mock_battery):
    ev = ElectricVehicle(
        episode_tracker=mock_episode_tracker,
        battery=mock_battery,
        name="TestEV",
        seconds_per_time_step=3600
    )
    ev.time_step = 0
    yield ev
    ev.time_step = 0

def test_initialization(electric_vehicle, mock_battery):
    assert electric_vehicle.name == "TestEV"
    assert electric_vehicle.battery == mock_battery
    assert electric_vehicle.time_step == 0

def test_properties(electric_vehicle, mock_battery):
    assert electric_vehicle.name == "TestEV"
    assert electric_vehicle.battery == mock_battery

def test_setters(electric_vehicle):
    electric_vehicle.name = "NewTestEV"
    assert electric_vehicle.name == "NewTestEV"

    new_battery = MagicMock(spec=Battery)
    electric_vehicle.battery = new_battery
    assert electric_vehicle.battery == new_battery

    with pytest.raises(AttributeError):
        electric_vehicle.battery = None

def test_next_time_step_basic(electric_vehicle, mock_battery):
    electric_vehicle.time_step = 0
    electric_vehicle.next_time_step()
    assert mock_battery.next_time_step.called
    assert electric_vehicle.time_step == 1

def test_reset(electric_vehicle, mock_battery):
    electric_vehicle.time_step = 5
    electric_vehicle.reset()
    assert mock_battery.reset.called
    assert electric_vehicle.time_step == 0

def test_observations(electric_vehicle, mock_battery):
    electric_vehicle._some_observable_variable = np.array([42.0] * 24)  # custom mock data
    mock_battery.soc[0] = 0.6
    obs = electric_vehicle.observations()
    assert obs['electric_vehicle_soc'] == 0.6
    assert obs['some_observable_variable'] == 42.0

def test_as_dict(electric_vehicle, mock_battery):
    mock_battery.capacity = 75.0
    mock_battery.soc[0] = 0.8
    electric_vehicle._temperature = np.array([22.0] * 24)
    result = electric_vehicle.as_dict()
    assert result['name'] == "TestEV"
    assert result['Battery capacity'] == 75.0
    assert result['electric_vehicle_soc'] == 0.8
    assert result['temperature'] == 22.0

def test_render_simulation_end_data(electric_vehicle, mock_battery, mock_episode_tracker):
    mock_battery.soc[0] = 0.5
    mock_battery.soc[1] = 0.6
    mock_battery.capacity = 80.0
    mock_episode_tracker.episode_time_steps = 3
    electric_vehicle.episode_tracker = mock_episode_tracker

    result = electric_vehicle.render_simulation_end_data()
    assert result['simulation_name'] == "TestEV"
    assert len(result['data']) == 3
    assert result['data'][0]['time_step'] == 0
    assert result['data'][0]['battery']['soc'] == 0.5
    assert result['data'][0]['battery']['capacity'] == 80.0
    assert result['data'][1]['battery']['soc'] == 0.6
