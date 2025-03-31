import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from citylearn.electric_vehicle_charger import Charger
from citylearn.base import EpisodeTracker
from citylearn.electric_vehicle import ElectricVehicle
from citylearn.energy_model import Battery

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
    
    # Mock the charge method to properly update SOC and energy_balance
    def charge_side_effect(energy_kwh):
        nonlocal battery
        if battery.time_step == 0:
            prev_soc = battery.initial_soc
        else:
            prev_soc = battery.soc[battery.time_step - 1]
        
        # Apply efficiency
        if energy_kwh >= 0:  # Charging
            effective_energy = energy_kwh * battery.round_trip_efficiency
        else:  # Discharging
            effective_energy = energy_kwh / battery.round_trip_efficiency
        
        new_soc = (prev_soc * battery.capacity + effective_energy) / battery.capacity
        new_soc = max(0.0, min(1.0, new_soc))  # Clamp between 0 and 1
        
        battery.soc[battery.time_step] = new_soc
        battery.energy_balance[battery.time_step] = effective_energy
    
    battery.charge.side_effect = charge_side_effect
    return battery

@pytest.fixture
def mock_electric_vehicle(mock_battery):
    ev = MagicMock(spec=ElectricVehicle)
    ev.battery = mock_battery
    return ev

@pytest.fixture
def charger(mock_episode_tracker):
    charger = Charger(
        episode_tracker=mock_episode_tracker,
        max_charging_power=10.0,
        min_charging_power=1.0,
        max_discharging_power=10.0,
        min_discharging_power=1.0,
        connected_electric_vehicle=None  # Start with no connected vehicle
    )
    charger._Charger__electricity_consumption = np.zeros(24)
    charger._Charger__past_charging_action_values = np.zeros(24)
    charger._Charger__past_connected_evs = [None] * 24
    charger.time_step = 0
    yield charger
    # Cleanup after test
    charger.connected_electric_vehicle = None
    charger.incoming_electric_vehicle = None

# --- TESTS ---

def test_action_value_storage(charger, mock_electric_vehicle):
    """Verify charger stores action values correctly"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    
    test_actions = [0.3, -0.5, 0.0, 1.0]
    for i, action in enumerate(test_actions):
        charger.time_step = i
        charger.update_connected_electric_vehicle_soc(action)
        assert charger._Charger__past_charging_action_values[i] == action

def test_power_clamping(charger, mock_electric_vehicle):
    """Verify power values are clamped correctly"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    charger.seconds_per_time_step = 3600  # 1 hour
    
    # Test cases: (action, expected_power_kw)
    test_cases = [
        (0.5, 5.0),    # 50% of max charging
        (1.5, 10.0),    # Above max -> clamp to max
        (0.05, 1.0),    # Below min -> clamp to min
        (-0.5, -5.0),   # 50% discharging
        (-1.5, -10.0),  # Max discharging
        (-0.05, -1.0)   # Min discharging
    ]
    
    for action, expected_power in test_cases:
        charger.update_connected_electric_vehicle_soc(action)
        # Get the actual power value passed to battery.charge()
        args, _ = mock_electric_vehicle.battery.charge.call_args
        actual_power_kw = args[0] / (charger.seconds_per_time_step / 3600)  # Convert back to kW
        
        assert actual_power_kw == pytest.approx(expected_power)
        mock_electric_vehicle.battery.charge.reset_mock()

def test_electricity_consumption_calculation(charger, mock_electric_vehicle):
    """Verify electricity consumption is calculated correctly"""
    # Setup
    charger.connected_electric_vehicle = mock_electric_vehicle
    charger.seconds_per_time_step = 3600  # 1 hour
    
    # Configure the mock battery to update energy_balance when charge() is called
    def charge_side_effect(energy_kwh):
        mock_electric_vehicle.battery.energy_balance[charger.time_step] = energy_kwh
    
    mock_electric_vehicle.battery.charge.side_effect = charge_side_effect
    mock_electric_vehicle.battery.energy_balance = np.zeros(24)
    
    # Mock efficiency
    with patch.object(charger, 'get_efficiency', return_value=0.9):
        charger.update_connected_electric_vehicle_soc(0.5)  # action_value = 0.5
        
        # Verify calculations:
        # power = 0.5 * 10 = 5 kW
        # energy = 5 * 1 = 5 kWh
        # energy_kwh = 5 * 0.9 = 4.5 kWh (after efficiency loss)
        # electricity_consumption = 4.5 / 0.9 = 5.0 kWh
        
        # Check that charge() was called with 4.5 kWh
        mock_electric_vehicle.battery.charge.assert_called_once_with(pytest.approx(4.5))
        
        # Check electricity consumption calculation
        assert charger._Charger__electricity_consumption[0] == pytest.approx(5.0)

def test_no_action_when_no_ev(charger):
    """Verify no action is taken when no EV is connected"""
    charger.update_connected_electric_vehicle_soc(0.5)
    assert charger._Charger__electricity_consumption[0] == 0
    assert charger._Charger__past_charging_action_values[0] == 0.5

def test_zero_action(charger, mock_electric_vehicle):
    """Verify zero action results in no energy exchange"""

    charger.connected_electric_vehicle = mock_electric_vehicle
    charger.update_connected_electric_vehicle_soc(0.0)
    
    assert charger._Charger__electricity_consumption[0] == 0
    assert not mock_electric_vehicle.battery.charge.called