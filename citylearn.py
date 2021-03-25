import gym
from gym.utils import seeding
import numpy as np
import pandas as pd
import json
from gym import spaces
from energy_models import Battery, HeatPump, ElectricHeater, EnergyStorage, Building
from reward_function import reward_function_sa, reward_function_ma
from pathlib import Path
gym.logger.set_level(40)

# Reference Rule-based controller. Used as a baseline to calculate the costs in CityLearn
# It requires, at least, the hour of the day as input state
class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day = states[0]
        
        multiplier = 0.4
        # Daytime: release stored energy
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 7 and hour_day <= 11:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 12 and hour_day <= 15:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 16 and hour_day <= 18:
            a = [[-0.11 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 19 and hour_day <= 22:
            a = [[-0.06 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        # Early nightime: store DHW and/or cooling energy
        if hour_day >= 23 and hour_day <= 24:
            a = [[0.085 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 1 and hour_day <= 6:
            a = [[0.1383 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        return np.array(a, dtype='object')

def auto_size(buildings):
    for building in buildings.values():
        
        # Autosize guarantees that the DHW device is large enough to always satisfy the maximum DHW demand
        if building.dhw_heating_device.nominal_power == 'autosize':
            
            # If the DHW device is a HeatPump
            if isinstance(building.dhw_heating_device, HeatPump):
                
                #We assume that the heat pump is always large enough to meet the highest heating or cooling demand of the building
                building.dhw_heating_device.nominal_power = np.array(building.sim_results['dhw_demand']/building.dhw_heating_device.cop_heating).max()
                
            # If the device is an electric heater
            elif isinstance(building.dhw_heating_device, ElectricHeater):
                building.dhw_heating_device.nominal_power = (np.array(building.sim_results['dhw_demand'])/building.dhw_heating_device.efficiency).max()
        
        # Autosize guarantees that the cooling device device is large enough to always satisfy the maximum DHW demand
        if building.cooling_device.nominal_power == 'autosize':

            building.cooling_device.nominal_power = (np.array(building.sim_results['cooling_demand'])/building.cooling_device.cop_cooling).max()
        
        # Defining the capacity of the storage devices as a number of times the maximum demand
        building.dhw_storage.capacity = max(building.sim_results['dhw_demand'])*building.dhw_storage.capacity
        building.cooling_storage.capacity = max(building.sim_results['cooling_demand'])*building.cooling_storage.capacity
        
        # Done in order to avoid dividing by 0 if the capacity is 0
        if building.dhw_storage.capacity <= 0.00001:
            building.dhw_storage.capacity = 0.00001
        if building.cooling_storage.capacity <= 0.00001:
            building.cooling_storage.capacity = 0.00001
        
        
def building_loader(data_path, building_attributes, weather_file, solar_profile, carbon_intensity, building_ids, buildings_states_actions, save_memory = True):
    with open(building_attributes) as json_file:
        data = json.load(json_file)

    buildings, observation_spaces, action_spaces = {},[],[]
    s_low_central_agent, s_high_central_agent, appended_states = [], [], []
    a_low_central_agent, a_high_central_agent, appended_actions = [], [], []
    for uid, attributes in zip(data, data.values()):
        if uid in building_ids:
            
            battery = Battery(capacity = attributes['Battery']['capacity'],
                                         capacity_loss_coef = attributes['Battery']['capacity_loss_coefficient'],
                                         loss_coef = attributes['Battery']['loss_coefficient'],
                                         efficiency = attributes['Battery']['efficiency'],
                                         nominal_power = attributes['Battery']['nominal_power'],
                                         power_efficiency_curve = attributes['Battery']['power_efficiency_curve'],
                                         capacity_power_curve = attributes['Battery']['capacity_power_curve'],                              
                                         save_memory = save_memory)        
            
            heat_pump = HeatPump(nominal_power = attributes['Heat_Pump']['nominal_power'], 
                                 eta_tech = attributes['Heat_Pump']['technical_efficiency'], 
                                 t_target_heating = attributes['Heat_Pump']['t_target_heating'], 
                                 t_target_cooling = attributes['Heat_Pump']['t_target_cooling'], save_memory = save_memory)

            electric_heater = ElectricHeater(nominal_power = attributes['Electric_Water_Heater']['nominal_power'], 
                                             efficiency = attributes['Electric_Water_Heater']['efficiency'], save_memory = save_memory)

            chilled_water_tank = EnergyStorage(capacity = attributes['Chilled_Water_Tank']['capacity'],
                                               loss_coef = attributes['Chilled_Water_Tank']['loss_coefficient'], save_memory = save_memory)

            dhw_tank = EnergyStorage(capacity = attributes['DHW_Tank']['capacity'],
                                     loss_coef = attributes['DHW_Tank']['loss_coefficient'], save_memory = save_memory)

            building = Building(buildingId = uid, dhw_storage = dhw_tank, cooling_storage = chilled_water_tank, electrical_storage = battery, dhw_heating_device = electric_heater, cooling_device = heat_pump, save_memory = save_memory)

            data_file = str(uid) + '.csv'
            simulation_data = data_path / data_file
            with open(simulation_data) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['cooling_demand'] = list(data['Cooling Load [kWh]'])
            building.sim_results['dhw_demand'] = list(data['DHW Heating [kWh]'])
            building.sim_results['non_shiftable_load'] = list(data['Equipment Electric Power [kWh]'])
            building.sim_results['month'] = list(data['Month'])
            building.sim_results['day'] = list(data['Day Type'])
            building.sim_results['hour'] = list(data['Hour'])
            building.sim_results['daylight_savings_status'] = list(data['Daylight Savings Status'])
            building.sim_results['t_in'] = list(data['Indoor Temperature [C]'])
            building.sim_results['avg_unmet_setpoint'] = list(data['Average Unmet Cooling Setpoint Difference [C]'])
            building.sim_results['rh_in'] = list(data['Indoor Relative Humidity [%]'])
            
            with open(weather_file) as csv_file:
                weather_data = pd.read_csv(csv_file)
                
            building.sim_results['t_out'] = list(weather_data['Outdoor Drybulb Temperature [C]'])
            building.sim_results['rh_out'] = list(weather_data['Outdoor Relative Humidity [%]'])
            building.sim_results['diffuse_solar_rad'] = list(weather_data['Diffuse Solar Radiation [W/m2]'])
            building.sim_results['direct_solar_rad'] = list(weather_data['Direct Solar Radiation [W/m2]'])
            
            # Reading weather forecasts
            building.sim_results['t_out_pred_6h'] = list(weather_data['6h Prediction Outdoor Drybulb Temperature [C]'])
            building.sim_results['t_out_pred_12h'] = list(weather_data['12h Prediction Outdoor Drybulb Temperature [C]'])
            building.sim_results['t_out_pred_24h'] = list(weather_data['24h Prediction Outdoor Drybulb Temperature [C]'])
            
            building.sim_results['rh_out_pred_6h'] = list(weather_data['6h Prediction Outdoor Relative Humidity [%]'])
            building.sim_results['rh_out_pred_12h'] = list(weather_data['12h Prediction Outdoor Relative Humidity [%]'])
            building.sim_results['rh_out_pred_24h'] = list(weather_data['24h Prediction Outdoor Relative Humidity [%]'])
            
            building.sim_results['diffuse_solar_rad_pred_6h'] = list(weather_data['6h Prediction Diffuse Solar Radiation [W/m2]'])
            building.sim_results['diffuse_solar_rad_pred_12h'] = list(weather_data['12h Prediction Diffuse Solar Radiation [W/m2]'])
            building.sim_results['diffuse_solar_rad_pred_24h'] = list(weather_data['24h Prediction Diffuse Solar Radiation [W/m2]'])
            
            building.sim_results['direct_solar_rad_pred_6h'] = list(weather_data['6h Prediction Direct Solar Radiation [W/m2]'])
            building.sim_results['direct_solar_rad_pred_12h'] = list(weather_data['12h Prediction Direct Solar Radiation [W/m2]'])
            building.sim_results['direct_solar_rad_pred_24h'] = list(weather_data['24h Prediction Direct Solar Radiation [W/m2]'])
            
            # Reading the building attributes
            building.building_type = attributes['Building_Type']
            building.climate_zone = attributes['Climate_Zone']
            building.solar_power_capacity = attributes['Solar_Power_Installed(kW)']

            with open(solar_profile) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['solar_gen'] = list(attributes['Solar_Power_Installed(kW)']*data['Hourly Data: AC inverter power (W)']/1000)
            
            with open(carbon_intensity) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['carbon_intensity'] = list(data['kg_CO2/kWh'])
            
            # Finding the max and min possible values of all the states, which can then be used by the RL agent to scale the states and train any function approximators more effectively
            s_low, s_high = [], []
            for state_name, value in zip(buildings_states_actions[uid]['states'], buildings_states_actions[uid]['states'].values()):
                if value == True:
                    if state_name == "net_electricity_consumption":
                        # lower and upper bounds of net electricity consumption are rough estimates and may not be completely accurate. Scaling this state-variable using these bounds may result in normalized values above 1 or below 0.
                        _net_elec_cons_upper_bound = max(np.array(building.sim_results['non_shiftable_load']) - np.array(building.sim_results['solar_gen']) + np.array(building.sim_results['dhw_demand'])/.8 + np.array(building.sim_results['cooling_demand']) + building.dhw_storage.capacity/.8 + building.cooling_storage.capacity/2)
                        s_low.append(0.)
                        s_high.append(_net_elec_cons_upper_bound)
                        s_low_central_agent.append(0.)
                        s_high_central_agent.append(_net_elec_cons_upper_bound)
                        
                    elif (state_name != 'cooling_storage_soc') and (state_name != 'dhw_storage_soc') and (state_name != 'electrical_storage_soc'):
                        s_low.append(min(building.sim_results[state_name]))
                        s_high.append(max(building.sim_results[state_name]))
                        
                        # Create boundaries of the observation space of a centralized agent (if a central agent is being used instead of decentralized ones). We include all the weather variables used as states, and use the list appended_states to make sure we don't include any repeated states (i.e. weather variables measured by different buildings)
                        if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                            s_low_central_agent.append(min(building.sim_results[state_name]))
                            s_high_central_agent.append(max(building.sim_results[state_name]))
                            
                        elif state_name not in appended_states:
                            s_low_central_agent.append(min(building.sim_results[state_name]))
                            s_high_central_agent.append(max(building.sim_results[state_name]))
                            appended_states.append(state_name)
                    else:
                        s_low.append(0.0)
                        s_high.append(1.0)
                        s_low_central_agent.append(0.0)
                        s_high_central_agent.append(1.0)
            
            '''The energy storage (tank) capacity indicates how many times bigger the tank is compared to the maximum hourly energy demand of the building (cooling or DHW respectively), which sets a lower bound for the action of 1/tank_capacity, as the energy storage device can't provide the building with more energy than it will ever need for a given hour. The heat pump is sized using approximately the maximum hourly energy demand of the building (after accounting for the COP, see function autosize). Therefore, we make the fair assumption that the action also has an upper bound equal to 1/tank_capacity. This boundaries should speed up the learning process of the agents and make them more stable rather than if we just set them to -1 and 1. I.e. if Chilled_Water_Tank.Capacity is 3 (3 times the max. hourly demand of the building in the entire year), its actions will be bounded between -1/3 and 1/3'''
            a_low, a_high = [], []    
            for action_name, value in zip(buildings_states_actions[uid]['actions'], buildings_states_actions[uid]['actions'].values()):
                if value == True:
                    if action_name =='cooling_storage':
                        
                        # Avoid division by 0
                        if attributes['Chilled_Water_Tank']['capacity'] > 0.000001:                            
                            a_low.append(max(-1.0/attributes['Chilled_Water_Tank']['capacity'], -1.0))
                            a_high.append(min(1.0/attributes['Chilled_Water_Tank']['capacity'], 1.0))
                            a_low_central_agent.append(max(-1.0/attributes['Chilled_Water_Tank']['capacity'], -1.0))
                            a_high_central_agent.append(min(1.0/attributes['Chilled_Water_Tank']['capacity'], 1.0))
                        else:
                            a_low.append(-1.0)
                            a_high.append(1.0)
                            a_low_central_agent.append(-1.0)
                            a_high_central_agent.append(1.0)
                    elif action_name =='dhw_storage':
                        if attributes['DHW_Tank']['capacity'] > 0.000001:
                            a_low.append(max(-1.0/attributes['DHW_Tank']['capacity'], -1.0))
                            a_high.append(min(1.0/attributes['DHW_Tank']['capacity'], 1.0))
                            a_low_central_agent.append(max(-1.0/attributes['DHW_Tank']['capacity'], -1.0))
                            a_high_central_agent.append(min(1.0/attributes['DHW_Tank']['capacity'], 1.0))
                        else:
                            a_low.append(-1.0)
                            a_high.append(1.0)
                            a_low_central_agent.append(-1.0)
                            a_high_central_agent.append(1.0)
                            
                    elif action_name =='electrical_storage':
                        a_low.append(-1.0)
                        a_high.append(1.0)
                        a_low_central_agent.append(-1.0)
                        a_high_central_agent.append(1.0)
                        
            building.set_state_space(np.array(s_high), np.array(s_low))
            building.set_action_space(np.array(a_high), np.array(a_low))
            
            observation_spaces.append(building.observation_space)
            action_spaces.append(building.action_space)
            
            buildings[uid] = building
    
    observation_space_central_agent = spaces.Box(low=np.float32(np.array(s_low_central_agent)), high=np.float32(np.array(s_high_central_agent)), dtype=np.float32)
    action_space_central_agent = spaces.Box(low=np.float32(np.array(a_low_central_agent)), high=np.float32(np.array(a_high_central_agent)), dtype=np.float32)
        
    for building in buildings.values():

        # If the DHW device is a HeatPump
        if isinstance(building.dhw_heating_device, HeatPump):
                
            # Calculating COPs of the heat pumps for every hour
            building.dhw_heating_device.cop_heating = building.dhw_heating_device.eta_tech*(building.dhw_heating_device.t_target_heating + 273.15)/(building.dhw_heating_device.t_target_heating - weather_data['Outdoor Drybulb Temperature [C]'])
            building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating < 0] = 20.0
            building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating > 20] = 20.0
            building.dhw_heating_device.cop_heating = building.dhw_heating_device.cop_heating.to_numpy()

        building.cooling_device.cop_cooling = building.cooling_device.eta_tech*(building.cooling_device.t_target_cooling + 273.15)/(weather_data['Outdoor Drybulb Temperature [C]'] - building.cooling_device.t_target_cooling)
        building.cooling_device.cop_cooling[building.cooling_device.cop_cooling < 0] = 20.0
        building.cooling_device.cop_cooling[building.cooling_device.cop_cooling > 20] = 20.0
        building.cooling_device.cop_cooling = building.cooling_device.cop_cooling.to_numpy()
        
        building.reset()
        
    auto_size(buildings)

    return buildings, observation_spaces, action_spaces, observation_space_central_agent, action_space_central_agent

class CityLearn(gym.Env):  
    def __init__(self, data_path, building_attributes, weather_file, solar_profile, building_ids, carbon_intensity = None, buildings_states_actions = None, simulation_period = (0,8759), cost_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption'], central_agent = False, save_memory = True, verbose = 0):
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)
        
        self.data_path = data_path
        self.buildings_states_actions_filename = buildings_states_actions
        self.buildings_net_electricity_demand = []
        self.building_attributes = building_attributes
        self.solar_profile = solar_profile
        self.carbon_intensity = carbon_intensity
        self.building_ids = building_ids
        self.cost_function = cost_function
        self.cost_rbc = None
        self.weather_file = weather_file
        self.central_agent = central_agent
        self.loss = []
        self.verbose = verbose
        
        params_loader = {'data_path':data_path,
                         'building_attributes':self.data_path / self.building_attributes,
                         'weather_file':self.data_path / self.weather_file,
                         'solar_profile':self.data_path / self.solar_profile,
                         'carbon_intensity':self.data_path / self.carbon_intensity,
                         'building_ids':building_ids,
                         'buildings_states_actions':self.buildings_states_actions,
                         'save_memory':save_memory}
        
        self.buildings, self.observation_spaces, self.action_spaces, self.observation_space, self.action_space = building_loader(**params_loader)
        
        self.simulation_period = simulation_period
        self.uid = None
        self.n_buildings = len([i for i in self.buildings])
        self.reset()
        
    def get_state_action_spaces(self):
        return self.observation_spaces, self.action_spaces
            
    def next_hour(self):
        self.time_step = next(self.hour)
        for building in self.buildings.values():
            building.time_step = self.time_step
            
    def get_building_information(self):
        
        np.seterr(divide='ignore', invalid='ignore')
        # Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand
        building_info = {}
        n_years = (self.simulation_period[1] - self.simulation_period[0] + 1)/8760
        for uid, building in self.buildings.items():
            building_info[uid] = {}
            building_info[uid]['building_type'] = building.building_type
            building_info[uid]['climate_zone'] = building.climate_zone
            building_info[uid]['solar_power_capacity (kW)'] = round(building.solar_power_capacity, 3)
            building_info[uid]['Annual_DHW_demand (kWh)'] = round(sum(building.sim_results['dhw_demand'])/n_years, 3)
            building_info[uid]['Annual_cooling_demand (kWh)'] = round(sum(building.sim_results['cooling_demand'])/n_years, 3)
            building_info[uid]['Annual_nonshiftable_electrical_demand (kWh)'] = round(sum(building.sim_results['non_shiftable_load'])/n_years, 3)
            
            building_info[uid]['Correlations_DHW'] = {}
            building_info[uid]['Correlations_cooling_demand'] = {}
            building_info[uid]['Correlations_non_shiftable_load'] = {}
            
            for uid_corr, building_corr in self.buildings.items():
                if uid_corr != uid:
                    building_info[uid]['Correlations_DHW'][uid_corr] = round((np.corrcoef(np.array(building.sim_results['dhw_demand']), np.array(building_corr.sim_results['dhw_demand'])))[0][1], 3)
                    building_info[uid]['Correlations_cooling_demand'][uid_corr] = round((np.corrcoef(np.array(building.sim_results['cooling_demand']), np.array(building_corr.sim_results['cooling_demand'])))[0][1], 3)
                    building_info[uid]['Correlations_non_shiftable_load'][uid_corr] = round((np.corrcoef(np.array(building.sim_results['non_shiftable_load']), np.array(building_corr.sim_results['non_shiftable_load'])))[0][1], 3)
        
        return building_info
        
    def step(self, actions):
                
        self.buildings_net_electricity_demand = []
        self.current_carbon_intensity = list(self.buildings.values())[0].sim_results['carbon_intensity'][self.time_step]
        electric_demand = 0
        elec_consumption_electrical_storage = 0
        elec_consumption_dhw_storage = 0
        elec_consumption_cooling_storage = 0
        elec_consumption_dhw_total = 0
        elec_consumption_cooling_total = 0
        elec_consumption_appliances = 0
        elec_generation = 0
        
        if self.central_agent:
            # If the agent is centralized, all the actions for all the buildings are provided as an ordered list of numbers. The order corresponds to the order of the buildings as they appear on the file building_attributes.json, and only considering the buildings selected for the simulation by the user (building_ids).
            for uid, building in self.buildings.items():
            
                if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                    # Cooling
                    _electric_demand_cooling = building.set_storage_cooling(actions[0])
                    actions = actions[1:]
                    elec_consumption_cooling_storage += building._electric_consumption_cooling_storage
                else:
                    _electric_demand_cooling = 0

                if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                    # DHW
                    _electric_demand_dhw = building.set_storage_heating(actions[0])
                    actions = actions[1:]
                    elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                else:
                    _electric_demand_dhw = 0

                # Total heating and cooling electrical loads
                elec_consumption_cooling_total += _electric_demand_cooling
                elec_consumption_dhw_total += _electric_demand_dhw

                # Electrical appliances
                _non_shiftable_load = building.get_non_shiftable_load()
                elec_consumption_appliances += _non_shiftable_load

                # Solar generation
                _solar_generation = building.get_solar_power()
                elec_generation += _solar_generation

                # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
                building_electric_demand = round(_electric_demand_cooling + _electric_demand_dhw + _non_shiftable_load - _solar_generation, 4)

                # Electricity consumed by every building
                building.current_net_electricity_demand = building_electric_demand
                self.buildings_net_electricity_demand.append(-building_electric_demand) # >0 if solar generation > electricity consumption

                # Total electricity consumption
                electric_demand += building_electric_demand
                
            assert len(actions) == 0, 'Some of the actions provided were not used'
            
        else:
            
            assert len(actions) == self.n_buildings, "The length of the list of actions should match the length of the list of buildings."

            for a, (uid, building) in zip(actions, self.buildings.items()):

                assert sum(self.buildings_states_actions[uid]['actions'].values()) == len(a), "The number of input actions for building "+str(uid)+" must match the number of actions defined in the list of building attributes."
                
                
                # Getting input actions and stablishing associations between the components of the action array and their corresponding actions.
                if self.buildings_states_actions[uid]['actions']['electrical_storage']:
                    
                    if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                        # Cooling
                        _electric_demand_cooling = building.set_storage_cooling(a[0])
                        elec_consumption_cooling_storage += building._electric_consumption_cooling_storage

                        # 'Electrical Storage' & 'Cooling Storage' & 'DHW Storage'
                        if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                            # DHW
                            _electric_demand_dhw = building.set_storage_heating(a[1])
                            elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                            
                            # Electrical
                            _electric_demand_electrical_storage = building.set_storage_electrical(a[2])
                            elec_consumption_electrical_storage += _electric_demand_electrical_storage

                        # 'Electrical Storage' & 'Cooling Storage'
                        else:
                            _electric_demand_dhw = building.set_storage_heating(0.0)
                            # Electrical
                            _electric_demand_electrical_storage = building.set_storage_electrical(a[1])
                            elec_consumption_electrical_storage += _electric_demand_electrical_storage
                    else:
                        _electric_demand_cooling = building.set_storage_cooling(0.0)
                        # 'Electrical Storage' & 'DHW Storage'
                        if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                            # DHW
                            _electric_demand_dhw = building.set_storage_heating(a[0])
                            elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                            
                            # Electrical
                            _electric_demand_electrical_storage = building.set_storage_electrical(a[1])
                            elec_consumption_electrical_storage += _electric_demand_electrical_storage

                        # 'Electrical Storage'
                        else:
                            _electric_demand_dhw = building.set_storage_heating(0.0)
                            # Electrical
                            _electric_demand_electrical_storage = building.set_storage_electrical(a[0])
                            elec_consumption_electrical_storage += _electric_demand_electrical_storage
                        
                else:
                    
                    _electric_demand_electrical_storage = 0.0
                    
                    if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                        # Cooling
                        _electric_demand_cooling = building.set_storage_cooling(a[0])
                        elec_consumption_cooling_storage += building._electric_consumption_cooling_storage

                        if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                            # DHW
                            _electric_demand_dhw = building.set_storage_heating(a[1])
                            elec_consumption_dhw_storage += building._electric_consumption_dhw_storage

                        else:
                            _electric_demand_dhw = building.set_storage_heating(0.0)

                    else:
                        _electric_demand_cooling = building.set_storage_cooling(0.0)
                        # DHW
                        _electric_demand_dhw = building.set_storage_heating(a[0])
                        elec_consumption_dhw_storage += building._electric_consumption_dhw_storage

                # Total heating and cooling electrical loads
                elec_consumption_cooling_total += _electric_demand_cooling
                elec_consumption_dhw_total += _electric_demand_dhw

                # Electrical appliances
                _non_shiftable_load = building.get_non_shiftable_load()
                elec_consumption_appliances += _non_shiftable_load

                # Solar generation
                _solar_generation = building.get_solar_power()
                elec_generation += _solar_generation

                # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
                building_electric_demand = round(_electric_demand_electrical_storage + _electric_demand_cooling + _electric_demand_dhw + _non_shiftable_load - _solar_generation, 4)

                # Electricity consumed by every building
                building.current_net_electricity_demand = building_electric_demand
                self.buildings_net_electricity_demand.append(-building_electric_demand)    

                # Total electricity consumption
                electric_demand += building_electric_demand
            
        self.next_hour()
        
        if self.central_agent:
            s, s_appended = [], []
            for uid, building in self.buildings.items():
                
                # If the agent is centralized, we append the states avoiding repetition. I.e. if multiple buildings share the outdoor temperature as a state, we only append it once to the states of the central agent. The variable s_appended is used for this purpose.
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if value == True:
                        if state_name not in s_appended:
                            if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                                s.append(building.sim_results[state_name][self.time_step])
                            elif state_name == 'net_electricity_consumption':
                                s.append(building.current_net_electricity_demand)
                            elif state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                                s.append(building.sim_results[state_name][self.time_step])
                                s_appended.append(state_name)
                            elif state_name == 'cooling_storage_soc':
                                s.append(building.cooling_storage._soc/building.cooling_storage.capacity)
                            elif state_name == 'dhw_storage_soc':
                                s.append(building.dhw_storage._soc/building.dhw_storage.capacity)
            self.state = np.array(s)
            rewards = reward_function_sa(self.buildings_net_electricity_demand)
            self.cumulated_reward_episode += rewards
            
        else:
            # If the controllers are decentralized, we append all the states to each associated agent's list of states.
            self.state = []
            for uid, building in self.buildings.items():
                s = []
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if value == True:
                        if state_name == 'net_electricity_consumption':
                            s.append(building.current_net_electricity_demand)
                        elif (state_name != 'cooling_storage_soc') and (state_name != 'dhw_storage_soc') and (state_name != 'electrical_storage_soc'):
                            s.append(building.sim_results[state_name][self.time_step])
                        elif state_name == 'cooling_storage_soc':
                            s.append(building.cooling_storage._soc/building.cooling_storage.capacity)
                        elif state_name == 'dhw_storage_soc':
                            s.append(building.dhw_storage._soc/building.dhw_storage.capacity)
                        elif state_name == 'electrical_storage_soc':
                            s.append(building.electrical_storage._soc/building.electrical_storage.capacity)

                self.state.append(np.array(s))
            self.state = np.array(self.state, dtype='object')
            
            
            
            rewards = self.reward_function.get_rewards(self.buildings_net_electricity_demand, self.current_carbon_intensity)
            self.cumulated_reward_episode += sum(rewards)
            
        # Control variables which are used to display the results and the behavior of the buildings at the district level.
        self.carbon_emissions.append(np.float32(max(0, electric_demand)*self.current_carbon_intensity))
        self.net_electric_consumption.append(np.float32(electric_demand))
        self.electric_consumption_electric_storage.append(np.float32(elec_consumption_electrical_storage))
        self.electric_consumption_dhw_storage.append(np.float32(elec_consumption_dhw_storage))
        self.electric_consumption_cooling_storage.append(np.float32(elec_consumption_cooling_storage))
        self.electric_consumption_dhw.append(np.float32(elec_consumption_dhw_total))
        self.electric_consumption_cooling.append(np.float32(elec_consumption_cooling_total))
        self.electric_consumption_appliances.append(np.float32(elec_consumption_appliances))
        self.electric_generation.append(np.float32(elec_generation))
        self.net_electric_consumption_no_storage.append(np.float32(electric_demand-elec_consumption_cooling_storage-elec_consumption_dhw_storage-elec_consumption_electrical_storage))
        self.net_electric_consumption_no_pv_no_storage.append(np.float32(electric_demand + elec_generation - elec_consumption_cooling_storage - elec_consumption_dhw_storage-elec_consumption_electrical_storage))
        
        terminal = self._terminal()
        return (self._get_ob(), rewards, terminal, {})
    
    def reset_baseline_cost(self):
        self.cost_rbc = None
        
    def reset(self):
        
        #Initialization of variables
        self.hour = iter(np.array(range(self.simulation_period[0], self.simulation_period[1] + 1)))
        self.next_hour()
            
        self.carbon_emissions = []
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        self.electric_consumption_electric_storage = []
        self.electric_consumption_dhw_storage = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_electrical_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_cooling = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
        
        self.cumulated_reward_episode = 0
        self.current_carbon_intensity = 0
        
        if self.central_agent:
            s, s_appended = [], []
            for uid, building in self.buildings.items():
                building.reset()
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if state_name not in s_appended:
                        if value == True:
                            if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                                s.append(building.sim_results[state_name][self.time_step])
                            elif state_name == 'net_electricity_consumption':
                                s.append(building.current_net_electricity_demand)
                            elif state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                                s.append(building.sim_results[state_name][self.time_step])
                                s_appended.append(state_name)
                            elif state_name == 'cooling_storage_soc':
                                s.append(0.0)
                            elif state_name == 'dhw_storage_soc':
                                s.append(0.0)
            self.state = np.array(s)
        else:
            self.reward_function = reward_function_ma(len(self.building_ids), self.get_building_information())
            
            self.state = []
            for uid, building in self.buildings.items():
                building.reset()
                s = []
                for state_name, value in zip(self.buildings_states_actions[uid]['states'], self.buildings_states_actions[uid]['states'].values()):
                    if value == True:
                        if state_name == 'net_electricity_consumption':
                            s.append(building.current_net_electricity_demand)
                        elif (state_name != 'cooling_storage_soc') and (state_name != 'dhw_storage_soc') and (state_name != 'electrical_storage_soc'):
                            s.append(building.sim_results[state_name][self.time_step])
                        elif state_name == 'cooling_storage_soc':
                            s.append(0.0)
                        elif state_name == 'dhw_storage_soc':
                            s.append(0.0)
                        elif state_name == 'electrical_storage_soc':
                            s.append(0.0)

                self.state.append(np.array(s, dtype=np.float32))
                
            self.state = np.array(self.state, dtype='object')
            
        return self._get_ob()
    
    def _get_ob(self):            
        return self.state
    
    def _terminal(self):
        is_terminal = bool(self.time_step >= self.simulation_period[1])
        if is_terminal:
            for building in self.buildings.values():
                building.terminate()
                
            # When the simulation is over, convert all the control variables to numpy arrays so they are easier to plot.
            self.carbon_emissions = np.array(self.carbon_emissions)
            self.net_electric_consumption = np.array(self.net_electric_consumption)
            self.net_electric_consumption_no_storage = np.array(self.net_electric_consumption_no_storage)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_pv_no_storage)
            self.electric_consumption_electric_storage = np.array(self.electric_consumption_electric_storage)
            self.electric_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
            self.electric_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
            self.electric_consumption_dhw = np.array(self.electric_consumption_dhw)
            self.electric_consumption_cooling = np.array(self.electric_consumption_cooling)
            self.electric_consumption_appliances = np.array(self.electric_consumption_appliances)
            self.electric_generation = np.array(self.electric_generation)
#             self.loss.append([i for i in self.get_baseline_cost().values()])
            
            if self.verbose == 1:
                print('Cumulated reward: '+str(self.cumulated_reward_episode))
            
        return is_terminal
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_buildings_net_electric_demand(self):
        return self.buildings_net_electricity_demand
    
    def cost(self):
        
        # Running the reference rule-based controller to find the baseline cost
        if self.cost_rbc is None:
            env_rbc = CityLearn(self.data_path, self.building_attributes, self.weather_file, self.solar_profile, self.building_ids, carbon_intensity = self.carbon_intensity, buildings_states_actions = self.buildings_states_actions_filename, simulation_period = self.simulation_period, cost_function = self.cost_function, central_agent = False)
            _, actions_spaces = env_rbc.get_state_action_spaces()

            #Instantiatiing the control agent(s)
            agent_rbc = RBC_Agent(actions_spaces)

            state = env_rbc.reset()
            done = False
            while not done:
                action = agent_rbc.select_action([list(env_rbc.buildings.values())[0].sim_results['hour'][env_rbc.time_step]])
                next_state, rewards, done, _ = env_rbc.step(action)
                state = next_state
                
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                self.cost_rbc, self.cost_rbc_last_yr = env_rbc.get_baseline_cost()
            else:
                self.cost_rbc = env_rbc.get_baseline_cost()
        
        # Compute the costs normalized by the baseline costs
        cost, cost_last_yr, c_score, c_score_last_yr = {}, {}, [], []
        if 'ramping' in self.cost_function:
            cost['ramping'] = np.abs((self.net_electric_consumption - np.roll(self.net_electric_consumption,1))[1:]).sum()/self.cost_rbc['ramping']
            c_score.append(cost['ramping'])
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['ramping_last_yr'] = np.abs((self.net_electric_consumption[-8760:] - np.roll(self.net_electric_consumption[-8760:],1))[1:]).sum()/self.cost_rbc_last_yr['ramping_last_yr']
                c_score_last_yr.append(cost_last_yr['ramping_last_yr'])
            
        # Finds the load factor for every month (average monthly demand divided by its maximum peak), and averages all the load factors across the 12 months. The metric is one minus the load factor.
        if '1-load_factor' in self.cost_function:
            cost['1-load_factor'] = np.mean([1-np.mean(self.net_electric_consumption[i:i+int(8760/12)])/ np.max(self.net_electric_consumption[i:i+int(8760/12)]) for i in range(0,len(self.net_electric_consumption), int(8760/12))])/self.cost_rbc['1-load_factor']
            c_score.append(cost['1-load_factor'])
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['1-load_factor_last_yr'] = np.mean([1-np.mean(self.net_electric_consumption[-8760:][i:i+int(8760/12)])/ np.max(self.net_electric_consumption[-8760:][i:i+int(8760/12)]) for i in range(0,len(self.net_electric_consumption[-8760:]), int(8760/12))])/self.cost_rbc_last_yr['1-load_factor_last_yr']
                c_score_last_yr.append(cost_last_yr['1-load_factor_last_yr'])
           
        # Average of all the daily peaks of the 365 day of the year. The peaks are calculated using the net energy demand of the whole district of buildings.
        if 'average_daily_peak' in self.cost_function:
            cost['average_daily_peak'] = np.mean([self.net_electric_consumption[i:i+24].max() for i in range(0,len(self.net_electric_consumption),24)])/self.cost_rbc['average_daily_peak']
            c_score.append(cost['average_daily_peak'])
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['average_daily_peak_last_yr'] = np.mean([self.net_electric_consumption[-8760:][i:i+24].max() for i in range(0,len(self.net_electric_consumption[-8760:]),24)])/self.cost_rbc_last_yr['average_daily_peak_last_yr']
                c_score_last_yr.append(cost_last_yr['average_daily_peak_last_yr'])
            
        # Peak demand of the district for the whole year period.
        if 'peak_demand' in self.cost_function:
            cost['peak_demand'] = self.net_electric_consumption.max()/self.cost_rbc['peak_demand']
            c_score.append(cost['peak_demand'])
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['peak_demand_last_yr'] = self.net_electric_consumption[-8760:].max()/self.cost_rbc_last_yr['peak_demand_last_yr']
                c_score_last_yr.append(cost_last_yr['peak_demand_last_yr'])
            
        # Positive net electricity consumption for the whole district. It is clipped at a min. value of 0 because the objective is to minimize the energy consumed in the district, not to profit from the excess generation. (Island operation is therefore incentivized)
        if 'net_electricity_consumption' in self.cost_function:
            cost['net_electricity_consumption'] = self.net_electric_consumption.clip(min=0).sum()/self.cost_rbc['net_electricity_consumption']
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['net_electricity_consumption_last_yr'] = self.net_electric_consumption[-8760:].clip(min=0).sum()/self.cost_rbc_last_yr['net_electricity_consumption_last_yr']
            
        if 'carbon_emissions' in self.cost_function:
            cost['carbon_emissions'] = self.carbon_emissions.sum()/self.cost_rbc['carbon_emissions']
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['carbon_emissions_last_yr'] = self.carbon_emissions[-8760:].sum()/self.cost_rbc_last_yr['carbon_emissions_last_yr']
            
        # Not used for the challenge
        if 'quadratic' in self.cost_function:
            cost['quadratic'] = (self.net_electric_consumption.clip(min=0)**2).sum()/self.cost_rbc['quadratic']
            c_score.append(cost['quadratic'])
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['quadratic_last_yr'] = (self.net_electric_consumption[-8760:].clip(min=0)**2).sum()/self.cost_rbc_last_yr['quadratic_last_yr']
                c_score_last_yr.append(cost_last_yr['quadratic_last_yr'])
        
        cost['total'] = np.mean([c for c in cost.values()])
        
        if c_score != []:
            cost['coordination_score'] = np.mean(c_score)
            
        if c_score_last_yr != []:
            cost_last_yr['coordination_score_last_yr'] = np.mean(c_score_last_yr)
        
        if self.simulation_period[1] - self.simulation_period[0] > 8760:
            cost_last_yr['total_last_yr'] = np.mean([c for c in cost_last_yr.values()])
            return cost, cost_last_yr
        
        return cost
    
    def get_baseline_cost(self):
        
        # Computes the costs for the Rule-based controller, which are used to normalized the actual costs.
        cost, cost_last_yr = {}, {}
        if 'ramping' in self.cost_function:
            cost['ramping'] = np.abs((self.net_electric_consumption - np.roll(self.net_electric_consumption,1))[1:]).sum()
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['ramping_last_yr'] = np.abs((self.net_electric_consumption[-8760:] - np.roll(self.net_electric_consumption[-8760:],1))[1:]).sum()
            
        if '1-load_factor' in self.cost_function:
            cost['1-load_factor'] = np.mean([1 - np.mean(self.net_electric_consumption[i:i+int(8760/12)])/ np.max(self.net_electric_consumption[i:i+int(8760/12)]) for i in range(0, len(self.net_electric_consumption), int(8760/12))])
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['1-load_factor_last_yr'] = np.mean([1-np.mean(self.net_electric_consumption[-8760:][i:i+int(8760/12)])/ np.max(self.net_electric_consumption[-8760:][i:i+int(8760/12)]) for i in range(0,len(self.net_electric_consumption[-8760:]), int(8760/12))])
           
        if 'average_daily_peak' in self.cost_function:
            cost['average_daily_peak'] = np.mean([self.net_electric_consumption[i:i+24].max() for i in range(0, len(self.net_electric_consumption), 24)])
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['average_daily_peak_last_yr'] = np.mean([self.net_electric_consumption[-8760:][i:i+24].max() for i in range(0,len(self.net_electric_consumption[-8760:]),24)])
            
        if 'peak_demand' in self.cost_function:
            cost['peak_demand'] = self.net_electric_consumption.max()
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['peak_demand_last_yr'] = self.net_electric_consumption[-8760:].max()
            
        if 'net_electricity_consumption' in self.cost_function:
            cost['net_electricity_consumption'] = self.net_electric_consumption.clip(min=0).sum()
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['net_electricity_consumption_last_yr'] = self.net_electric_consumption[-8760:].clip(min=0).sum()
            
        if 'carbon_emissions' in self.cost_function:
            cost['carbon_emissions'] = self.carbon_emissions.sum()
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['carbon_emissions_last_yr'] = self.carbon_emissions[-8760:].sum()
            
        if 'quadratic' in self.cost_function:
            cost['quadratic'] = (self.net_electric_consumption.clip(min=0)**2).sum()
            
            if self.simulation_period[1] - self.simulation_period[0] > 8760:
                cost_last_yr['quadratic_last_yr'] = (self.net_electric_consumption[-8760:].clip(min=0)**2).sum()
                
        if self.simulation_period[1] - self.simulation_period[0] > 8760:
            return cost, cost_last_yr
            
        return cost