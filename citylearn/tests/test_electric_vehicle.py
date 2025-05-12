import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from citylearn.electric_vehicle import ElectricVehicle
from citylearn.data import ElectricVehicleSimulation
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
    battery.capacity = 100.0  # 100 kWh battery
    battery.nominal_power = 10.0  # 10 kW nominal power
    battery.initial_soc = 0.5  # Start at 50% SOC
    battery.soc = np.array([0.5] * 24)  # SOC time series
    battery.energy_balance = np.zeros(24)  # Track energy changes
    battery.efficiency = 0.9  # 90% efficiency
    battery.round_trip_efficiency = 0.9**0.5  # sqrt(efficiency)
    battery.time_step = 0
    
    def force_set_soc_side_effect(new_soc):
        battery.soc[battery.time_step] = new_soc
    
    battery.force_set_soc.side_effect = force_set_soc_side_effect
    return battery

@pytest.fixture
def mock_ev_simulation():
    simulation = MagicMock(spec=ElectricVehicleSimulation)
    
    simulation.electric_vehicle_charger_state = np.array([1] * 24)  # 1 = connected to charger
    simulation.electric_vehicle_required_soc_departure = np.array([0.8] * 24)
    simulation.electric_vehicle_estimated_soc_arrival = np.array([0.4] * 24)
    simulation.electric_vehicle_soc_arrival = np.array([0.4] * 24)
    simulation.electric_vehicle_estimated_arrival_time = np.array([0] * 24)
    simulation.electric_vehicle_departure_time = np.array([23] * 24)
    
    simulation.temperature = np.array([20.0] * 24)
    simulation.day_type = np.array([1] * 24)  # 1 = weekday
    
    return simulation

@pytest.fixture
def electric_vehicle(mock_episode_tracker, mock_battery, mock_ev_simulation):
    ev = ElectricVehicle(
        electric_vehicle_simulation=mock_ev_simulation,
        episode_tracker=mock_episode_tracker,
        battery=mock_battery,
        name="TestEV",
        seconds_per_time_step=3600  # 1 hour
    )
    ev.time_step = 0
    yield ev
    # Clean up
    ev.time_step = 0

# --- BASIC FUNCTIONALITY TESTS ---

def test_initialization(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test that electric vehicle initializes with correct default values"""
    assert electric_vehicle.name == "TestEV"
    assert electric_vehicle.battery == mock_battery
    assert electric_vehicle.electric_vehicle_simulation == mock_ev_simulation
    assert electric_vehicle.time_step == 0

def test_properties(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test property getters"""
    assert electric_vehicle.name == "TestEV"
    assert electric_vehicle.battery == mock_battery
    assert electric_vehicle.electric_vehicle_simulation == mock_ev_simulation

def test_setters(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test property setters"""
    electric_vehicle.name = "NewTestEV"
    assert electric_vehicle.name == "NewTestEV"
    
    new_simulation = MagicMock(spec=ElectricVehicleSimulation)
    electric_vehicle.electric_vehicle_simulation = new_simulation
    assert electric_vehicle.electric_vehicle_simulation == new_simulation
    
    new_battery = MagicMock(spec=Battery)
    electric_vehicle.battery = new_battery
    assert electric_vehicle.battery == new_battery
    
    with pytest.raises(AttributeError):
        electric_vehicle.battery = None

# --- TIME STEP AND RESET TESTS ---

def test_next_time_step_basic(electric_vehicle, mock_battery):
    """Test basic time step advancement"""
    electric_vehicle.time_step = 0
    electric_vehicle.next_time_step()
    
    assert mock_battery.next_time_step.called
    
    assert electric_vehicle.time_step == 1

def test_reset(electric_vehicle, mock_battery):
    """Test reset functionality"""
    electric_vehicle.time_step = 5
    
    electric_vehicle.reset()
    
    assert mock_battery.reset.called
    
    assert electric_vehicle.time_step == 0

def test_next_time_step_with_arrival(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test time step advancement with EV arrival (state transition from not connected to connected)"""
    # Set up time step 0
    electric_vehicle.time_step = 0
    
    mock_ev_simulation.electric_vehicle_charger_state[0] = 2
    
    mock_ev_simulation.electric_vehicle_charger_state[1] = 3
    
    # Set arrival SOC for the current time step
    mock_ev_simulation.electric_vehicle_soc_arrival[0] = 0.3

    mock_ev_simulation.electric_vehicle_soc_arrival[1] = 0.3
    
    # Execute next_time_step - this should trigger the arrival logic
    electric_vehicle.next_time_step()

    # Verify that force_set_soc was called with the arrival SOC
    mock_battery.force_set_soc.assert_called_with(0.3)



# --- OBSERVATION TESTS ---

def test_observations(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test that observations method returns correct data"""
    mock_ev_simulation.temperature[0] = 25.0
    mock_ev_simulation.day_type[0] = 2
    mock_ev_simulation.electric_vehicle_required_soc_departure[0] = 0.9
    mock_battery.soc[0] = 0.6
    
    observations = electric_vehicle.observations()
    
    assert observations['temperature'] == 25.0
    assert observations['day_type'] == 2
    assert observations['electric_vehicle_required_soc_departure'] == 0.9
    assert observations['electric_vehicle_soc'] == 0.6
    
    assert 'electric_vehicle_charger_state' not in observations
    assert 'electric_vehicle_soc_arrival' not in observations

def test_as_dict(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test that as_dict method returns correct representation"""
    mock_ev_simulation.electric_vehicle_charger_state[0] = 1
    mock_battery.capacity = 75.0
    mock_battery.soc[0] = 0.8
    
    result = electric_vehicle.as_dict()
    
    assert result['name'] == "TestEV"
    assert result['EV Charger State'] == 1
    assert result['Battery capacity'] == 75.0
    assert result['electric_vehicle_soc'] == 0.8

def test_render_simulation_end_data(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test that render_simulation_end_data returns correct structured data"""
    mock_battery.soc[0] = 0.5
    mock_battery.soc[1] = 0.6
    mock_battery.capacity = 80.0
    mock_ev_simulation.temperature = np.array([22.0, 23.0, 24.0] + [0] * 21)
    mock_ev_simulation.electric_vehicle_charger_state = np.array([1, 1, 1] + [0] * 21)
    
    mock_episode_tracker = electric_vehicle.episode_tracker
    mock_episode_tracker.episode_time_steps = 3
    
    result = electric_vehicle.render_simulation_end_data()
    
    assert result['simulation_name'] == "TestEV"
    assert len(result['data']) == 3  
    
    assert result['data'][0]['time_step'] == 0
    assert result['data'][0]['simulation']['temperature'] == 22.0
    assert result['data'][0]['simulation']['electric_vehicle_charger_state'] == 1
    assert result['data'][0]['battery']['soc'] == 0.5
    assert result['data'][0]['battery']['capacity'] == 80.0
    
    assert result['data'][1]['time_step'] == 1
    assert result['data'][1]['simulation']['temperature'] == 23.0
    assert result['data'][1]['battery']['soc'] == 0.6

def test_ev_battery_integration(mock_episode_tracker, mock_ev_simulation):
    """Test integration between ElectricVehicle and a real Battery"""
    battery_stub = MagicMock(spec=Battery)
    battery_stub.capacity = 50.0
    battery_stub.efficiency = 0.9
    battery_stub.initial_soc = 0.5
    battery_stub.soc = np.array([0.5] * 24)
    
    ev = ElectricVehicle(
        electric_vehicle_simulation=mock_ev_simulation,
        episode_tracker=mock_episode_tracker,
        battery=battery_stub,
        name="IntegrationTestEV",
        seconds_per_time_step=3600
    )
    
    ev.time_step = 0

    mock_ev_simulation.electric_vehicle_charger_state[0] = 2  # Driving
    mock_ev_simulation.electric_vehicle_charger_state[1] = 3  # About to plug in
    mock_ev_simulation.electric_vehicle_charger_state[2] = 1  # About to plug in
    mock_ev_simulation.electric_vehicle_soc_arrival[1] = 0.3  # Arrival SOC
    
    ev.next_time_step()
    
    battery_stub.force_set_soc.assert_called_with(0.3)

def test_next_time_step_with_invalid_arrival_soc(electric_vehicle, mock_ev_simulation):
    """Test behavior with invalid arrival SOC"""
    electric_vehicle.time_step = 0
    
    mock_ev_simulation.electric_vehicle_charger_state[0] = 2
    
    mock_ev_simulation.electric_vehicle_charger_state[1] = 3
    
    # Set arrival SOC for the current time step
    mock_ev_simulation.electric_vehicle_soc_arrival[0] = 0.3

    mock_ev_simulation.electric_vehicle_soc_arrival[1] = 1.5  # Invalid SOC (>1.0)

    # Execute next_time_step - this should raise an AttributeError
    with pytest.raises(AttributeError):
        electric_vehicle.next_time_step()

def test_next_time_step_with_departure(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test time step advancement with EV departure and SOC variability"""
    # Set up time step 1 (not 0, because departure logic checks time_step > 0)
    electric_vehicle.time_step = 1
    mock_battery.time_step = 1
    
    # Set previous SOC (at time_step 0)
    mock_battery.soc[1] = 0.7
    
    mock_ev_simulation.electric_vehicle_charger_state[1] = 2
    
    mock_ev_simulation.electric_vehicle_charger_state[2] = 3

    mock_ev_simulation.electric_vehicle_charger_state[3] = 2
    
    # Mock numpy functions for deterministic testing
    with patch('numpy.random.normal', return_value=1.1), \
         patch('numpy.clip', side_effect=lambda x, min_val, max_val: 0.77 if x == 0.7 * 1.1 else x):
        electric_vehicle.next_time_step()
    
    mock_battery.force_set_soc.assert_called_with(0.77)        


def test_next_time_step_soc_variability_clipping(electric_vehicle, mock_battery, mock_ev_simulation):
    """Test that SOC variability is properly clipped within bounds"""
    # Set up time step 1 (not 0, because departure logic checks time_step > 0)
    electric_vehicle.time_step = 1
    mock_battery.time_step = 1
    
    # Set previous SOC (at time_step 0) to high value
    mock_battery.soc[1] = 0.95
    
    mock_ev_simulation.electric_vehicle_charger_state[1] = 2
    
    mock_ev_simulation.electric_vehicle_charger_state[2] = 3

    mock_ev_simulation.electric_vehicle_charger_state[3] = 2
    
    # Mock numpy functions for deterministic testing of high value + clipping
    with patch('numpy.random.normal', return_value=1.43), \
         patch('numpy.clip', side_effect=lambda x, min_val, max_val: 
               1.0 if min_val == 0.0 and max_val == 1.0 else  # For SOC clipping
               x):  # For variability factor clipping
        electric_vehicle.next_time_step()
    
    mock_battery.force_set_soc.assert_called_with(1.0)
