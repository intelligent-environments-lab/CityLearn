from citylearn.data import ChargerSimulation
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
def mock_ev_simulation():
    simulation = MagicMock(spec=ChargerSimulation)
    simulation.electric_vehicle_charger_state = np.array([1] * 24)
    simulation.electric_vehicle_required_soc_departure = np.array([0.8] * 24)
    simulation.electric_vehicle_estimated_soc_arrival = np.array([0.4] * 24)
    simulation.electric_vehicle_estimated_arrival_time = np.array([0] * 24)
    simulation.electric_vehicle_departure_time = np.array([23] * 24)
    simulation.temperature = np.array([20.0] * 24)
    simulation.day_type = np.array([1] * 24)
    return simulation

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
    
    def charge_side_effect(energy_kwh):
        if battery.time_step == 0:
            prev_soc = battery.initial_soc
        else:
            prev_soc = battery.soc[battery.time_step - 1]
        
        if energy_kwh >= 0:
            effective_energy = energy_kwh * battery.round_trip_efficiency
        else:
            effective_energy = energy_kwh / battery.round_trip_efficiency
        
        new_soc = (prev_soc * battery.capacity + effective_energy) / battery.capacity
        new_soc = max(0.0, min(1.0, new_soc))
        
        battery.soc[battery.time_step] = new_soc
        battery.energy_balance[battery.time_step] = effective_energy
    
    battery.charge.side_effect = charge_side_effect
    return battery

@pytest.fixture
def mock_electric_vehicle(mock_battery, mock_ev_simulation):
    ev = MagicMock(spec=ElectricVehicle)
    ev.battery = mock_battery
    ev.name = "TestEV"
    ev.electric_vehicle_simulation = mock_ev_simulation
    return ev

@pytest.fixture
def charger(mock_episode_tracker, mock_ev_simulation):
    charger = Charger(
        episode_tracker=mock_episode_tracker,
        charger_simulation=mock_ev_simulation,
        charger_id="TestCharger",
        max_charging_power=10.0,
        min_charging_power=1.0,
        max_discharging_power=10.0,
        min_discharging_power=1.0,
        connected_electric_vehicle=None
    )
    charger._Charger__electricity_consumption = np.zeros(24)
    charger._Charger__past_charging_action_values_kwh = np.zeros(24)
    charger._Charger__past_connected_evs = [None] * 24
    charger.time_step = 0
    charger.seconds_per_time_step = 3600
    yield charger
    charger.connected_electric_vehicle = None
    charger.incoming_electric_vehicle = None

# --- BASIC FUNCTIONALITY TESTS ---

def test_initialization(charger):
    """Test that charger initializes with correct default values"""
    assert charger.charger_id == "TestCharger"
    assert charger.max_charging_power == 10.0
    assert charger.min_charging_power == 1.0
    assert charger.max_discharging_power == 10.0
    assert charger.min_discharging_power == 1.0
    assert charger.efficiency == 1.0
    assert charger.connected_electric_vehicle is None
    assert charger.incoming_electric_vehicle is None

def test_plug_car(charger, mock_electric_vehicle):
    """Test plugging in an electric vehicle"""
    charger.plug_car(mock_electric_vehicle)
    assert charger.connected_electric_vehicle == mock_electric_vehicle
    assert charger._Charger__past_connected_evs[0] == mock_electric_vehicle

def test_plug_car_when_occupied(charger, mock_electric_vehicle):
    """Test that plugging fails when charger is already occupied"""
    charger.plug_car(mock_electric_vehicle)
    with pytest.raises(ValueError):
        charger.plug_car(mock_electric_vehicle)

def test_associate_incoming_car(charger, mock_electric_vehicle):
    """Test associating an incoming electric vehicle"""
    charger.associate_incoming_car(mock_electric_vehicle)
    assert charger.incoming_electric_vehicle == mock_electric_vehicle

# --- ACTION AND POWER TESTS ---

def test_action_value_storage(charger, mock_electric_vehicle):
    """Verify charger stores past energies correctly"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    
    test_actions = [0.3, -0.5, 0.0, 1.0]
    max_charging_power = 10
    for i, action in enumerate(test_actions):
        charger.time_step = i
        charger.update_connected_electric_vehicle_soc(action)
        energy = action * max_charging_power 
        assert charger._Charger__past_charging_action_values_kwh[i] == energy

def test_power_clamping(charger, mock_electric_vehicle):
    """Verify power values are clamped correctly"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    
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
        args, _ = mock_electric_vehicle.battery.charge.call_args
        actual_power_kw = args[0]  # Energy in kWh is same as power in kW for 1 hour
        assert actual_power_kw == pytest.approx(expected_power)
        mock_electric_vehicle.battery.charge.reset_mock()

def test_no_action_when_no_ev(charger):
    """Verify no action is taken when no EV is connected"""
    action = 0.5
    max_charging_power = 10
    charger.update_connected_electric_vehicle_soc(action)
    energy = action * max_charging_power 
    assert charger._Charger__electricity_consumption[0] == 0
    assert charger._Charger__past_charging_action_values_kwh[0] == energy

def test_zero_action(charger, mock_electric_vehicle):
    """Verify zero action results in no energy exchange"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    charger.update_connected_electric_vehicle_soc(0.0)
    
    assert charger._Charger__electricity_consumption[0] == 0
    assert not mock_electric_vehicle.battery.charge.called

# --- EFFICIENCY CURVE TESTS ---

def test_efficiency_curve_interpolation():
    """Test efficiency curve interpolation"""
    episode_tracker = MagicMock(spec=EpisodeTracker)
    episode_tracker.episode_time_steps = 24
    
    # Create charger with properly shaped efficiency curves
    charger = Charger(
        charger_simulation = mock_ev_simulation,
        episode_tracker=episode_tracker,
        charge_efficiency_curve=[[0, 0.83],[0.3, 0.83],[0.7, 0.9],[0.8, 0.9],[1, 0.85]],  # [power_levels], [efficiencies]
        discharge_efficiency_curve=[[0, 0.63],[0.3, 0.23],[0.7, 0.5],[0.8, 0.4],[1, 0.75]]
    )
    
    # Verify the curves were properly initialized
    assert charger.charge_efficiency_curve.shape == (2, 5)
    assert charger.discharge_efficiency_curve.shape == (2, 5)
    
    # Test charging efficiency
    assert charger.get_efficiency(0.0, True) == 0.83
    assert charger.get_efficiency(0.7, True) == 0.9
    assert charger.get_efficiency(1.0, True) == 0.85
    assert charger.get_efficiency(0.5, True) == pytest.approx(0.865)  # interpolated
    
    # Test discharging efficiency
    assert charger.get_efficiency(0.0, False) == 0.63
    assert charger.get_efficiency(0.7, False) == 0.5
    assert charger.get_efficiency(1.0, False) == 0.75
    assert charger.get_efficiency(0.5, False) == pytest.approx(0.365)  # interpolated

def test_default_efficiency_without_curve(charger):
    """Test that default efficiency is used when no curve is provided"""
    assert charger.get_efficiency(0.5, True) == 1.0
    assert charger.get_efficiency(0.5, False) == 1.0

# --- TIME STEP AND RESET TESTS ---

def test_next_time_step(charger, mock_electric_vehicle):
    """Test reset functionality"""
    charger.plug_car(mock_electric_vehicle)
    charger.associate_incoming_car(mock_electric_vehicle)
    assert charger.connected_electric_vehicle is not None
    assert charger.incoming_electric_vehicle is not None
    
    charger.time_step = 3

    charger.next_time_step()

    assert charger.time_step == 4
    assert charger.connected_electric_vehicle is None
    assert charger.incoming_electric_vehicle is None

    


def test_reset(charger, mock_electric_vehicle):
    """Test reset functionality"""
    charger.plug_car(mock_electric_vehicle)
    charger.update_connected_electric_vehicle_soc(0.5)
    
    charger.time_step = 3

    charger.reset()
    
    assert charger.time_step == 0
    assert charger.connected_electric_vehicle is None
    assert charger.incoming_electric_vehicle is None
    assert all(ec == 0 for ec in charger._Charger__electricity_consumption)
    assert all(pc == 0 for pc in charger._Charger__past_charging_action_values_kwh)
    assert all(ev is None for ev in charger._Charger__past_connected_evs)


# --- INTEGRATION TESTS ---

def test_charger_battery_integration_with_mocked_efficiency(charger, mock_electric_vehicle, mock_battery):
    """Test integration between charger and battery with mocked efficiency"""
    # Connect the EV to the charger
    charger.plug_car(mock_electric_vehicle)
    
    # Mock the get_efficiency method to always return 0.9
    charger.get_efficiency = lambda power, charging: 0.9
    
    # Initial state checks
    initial_soc = mock_battery.soc[0]
    assert initial_soc == 0.5  # From mock_battery fixture
    
    # Test charging
    charging_action = 0.5  # 50% of max charging power (10kW * 0.5 = 5kW)
    charger.update_connected_electric_vehicle_soc(charging_action)
    
    # Verify battery was charged with 0.9 efficiency
    expected_energy = 5.0 * 0.9  # 5kW * 1 hour * 0.9 efficiency
    expected_soc_change = (expected_energy * mock_battery.round_trip_efficiency) / mock_battery.capacity
    expected_new_soc = initial_soc + expected_soc_change
    assert mock_battery.soc[0] == pytest.approx(expected_new_soc)
    assert mock_battery.energy_balance[0] == pytest.approx(expected_energy * mock_battery.round_trip_efficiency)
    
    # Test discharging
    mock_battery.time_step = 1  # Advance time step
    charger.time_step = 1
    
    discharging_action = -0.3  # 30% of max discharging power (10kW * 0.3 = 3kW)
    charger.update_connected_electric_vehicle_soc(discharging_action)
    
    # Verify battery was discharged with 0.9 efficiency
    expected_energy = -3.0 / 0.9  # -3kW * 1 hour / 0.9 efficiency
    expected_soc_change = (expected_energy / mock_battery.round_trip_efficiency) / mock_battery.capacity
    expected_new_soc = mock_battery.soc[0] + expected_soc_change  # Starts from previous SOC
    
    assert mock_battery.soc[1] == pytest.approx(expected_new_soc)
    assert mock_battery.energy_balance[1] == pytest.approx(expected_energy / mock_battery.round_trip_efficiency)

    # Verify charger recorded the correct electricity consumption
    # Charging: consumption is energy / efficiency (4.269074841227312 / 0.9)
    # Discharging: consumption is energy * efficiency (-3.5136418446315325 * 0.9)
    if charging_action > 0:
        assert charger._Charger__electricity_consumption[0] == pytest.approx(mock_battery.energy_balance[0] / mock_battery.efficiency)
    else:
        assert charger._Charger__electricity_consumption[1] == pytest.approx(mock_battery.energy_balance[1] * mock_battery.efficiency)


        # --- TESTS FOR UNCOVERED LINES ---

def test_as_dict_with_connected_ev(charger, mock_electric_vehicle):
    """Test as_dict method with connected EV"""
    charger.plug_car(mock_electric_vehicle)
    charger.update_connected_electric_vehicle_soc(0.5)
    
    result = charger.as_dict()
    
    assert result["Charger Consumption-kWh"] != "-1.00"
    assert result["Charger Production-kWh"] == "-1.00"
    assert result["EV SOC-%"] == "0.55"  # Initial SOC from mock
    assert result["Is EV Connected"] == True
    assert result["EV Name"] == "TestEV"
    assert result["Charging Action-kWh"] == "5.0"  # 0.5 * 10kW

def test_as_dict_with_incoming_ev(charger, mock_electric_vehicle):
    """Test as_dict method with incoming EV"""
    charger.associate_incoming_car(mock_electric_vehicle)
    
    result = charger.as_dict()
    
    assert result["Incoming EV Name"] == "TestEV"
    assert result["Is EV Connected"] == True  # Because we have incoming EV
    assert result["EV Name"] == "TestEV"

def test_render_simulation_end_data_empty(charger):
    """Test render_simulation_end_data with no EVs"""
    result = charger.render_simulation_end_data()
    
    assert result['name'] == "TestCharger"
    assert len(result['charger_data']) == 23  # num_steps is 24-1
    for data in result['charger_data']:
        assert data['connected_ev'] is None
        assert data['incoming_ev'] is None
        assert data['electricity_consumption'] == 0

def test_render_simulation_end_data_with_ev(charger, mock_electric_vehicle):
    """Test render_simulation_end_data with connected EV"""
    charger.plug_car(mock_electric_vehicle)
    charger.update_connected_electric_vehicle_soc(0.5)
    
    result = charger.render_simulation_end_data()
    
    assert result['name'] == "TestCharger"
    assert len(result['charger_data']) == 23
    # First time step should have EV data
    assert result['charger_data'][0]['connected_ev'] is not None
    assert result['charger_data'][0]['connected_ev']['name'] == "TestEV"
    assert result['charger_data'][0]['electricity_consumption'] != 0

def test_render_simulation_end_data_with_incoming_ev(charger, mock_electric_vehicle):
    """Test render_simulation_end_data with incoming EV"""
    charger.associate_incoming_car(mock_electric_vehicle)
    
    result = charger.render_simulation_end_data()
    
    assert result['name'] == "TestCharger"
    assert len(result['charger_data']) == 23
    # Should show incoming EV but no connected EV
    assert result['charger_data'][0]['connected_ev'] is None
    assert result['charger_data'][0]['incoming_ev'] is not None
    assert result['charger_data'][0]['incoming_ev']['name'] == "TestEV"

def test_time_step_ratio_property(charger):
    """Test time_step_ratio property"""
    charger.time_step_ratio = 2.0
    assert charger.time_step_ratio == 2.0

def test_efficiency_setter_valid(charger):
    """Test efficiency setter with valid values"""
    charger.efficiency = 0.9
    assert charger.efficiency == 0.9

def test_efficiency_setter_invalid(charger):
    """Test efficiency setter with invalid values"""
    with pytest.raises(AssertionError):
        charger.efficiency = 0.0  # Must be > 0
    
    with pytest.raises(AssertionError):
        charger.efficiency = 1.1  # Must be <= 1

def test_electricity_consumption_property(charger):
    """Test electricity_consumption property"""
    consumption = np.array([1.0, 2.0, 3.0])
    charger._Charger__electricity_consumption = consumption
    assert np.array_equal(charger.electricity_consumption, consumption)

def test_past_charging_action_values_kwh_property(charger):
    """Test past_charging_action_values_kwh property"""
    actions = np.array([0.5, -0.3, 1.0])
    charger._Charger__past_charging_action_values_kwh = actions
    assert np.array_equal(charger.past_charging_action_values_kwh, actions)

def test_past_connected_evs_property(charger, mock_electric_vehicle):
    """Test past_connected_evs property"""
    evs = [mock_electric_vehicle, None, mock_electric_vehicle]
    charger._Charger__past_connected_evs = evs
    assert charger.past_connected_evs == evs

def test_charge_efficiency_curve_setter(charger):
    """Test charge_efficiency_curve setter"""
    curve = [[0, 0.8], [1, 0.9]]
    charger.charge_efficiency_curve = curve
    assert charger.charge_efficiency_curve.shape == (2, 2)
    assert np.array_equal(charger.charge_efficiency_curve[0], np.array([0, 1]))
    assert np.array_equal(charger.charge_efficiency_curve[1], np.array([0.8, 0.9]))

def test_discharge_efficiency_curve_setter(charger):
    """Test discharge_efficiency_curve setter"""
    curve = [[0, 0.7], [1, 0.8]]
    charger.discharge_efficiency_curve = curve
    assert charger.discharge_efficiency_curve.shape == (2, 2)
    assert np.array_equal(charger.discharge_efficiency_curve[0], np.array([0, 1]))
    assert np.array_equal(charger.discharge_efficiency_curve[1], np.array([0.7, 0.8]))

def test_connected_electric_vehicle_setter(charger, mock_electric_vehicle):
    """Test connected_electric_vehicle setter"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    assert charger.connected_electric_vehicle == mock_electric_vehicle

def test_incoming_electric_vehicle_setter(charger, mock_electric_vehicle):
    """Test incoming_electric_vehicle setter"""
    charger.incoming_electric_vehicle = mock_electric_vehicle
    assert charger.incoming_electric_vehicle == mock_electric_vehicle

def test_algorithm_action_based_time_step_hours_ratio(mock_episode_tracker):
    """Test algorithm_action_based_time_step_hours_ratio calculation"""
    # Create a new charger with specific seconds_per_time_step
    charger = Charger(
        charger_simulation = mock_ev_simulation,
        episode_tracker=mock_episode_tracker,
        charger_id="TestCharger",
        seconds_per_time_step=1800  # 30 minutes
    )
    
    # Verify the ratio is calculated correctly (1800 seconds = 0.5 hours)
    assert charger.algorithm_action_based_time_step_hours_ratio == 0.5