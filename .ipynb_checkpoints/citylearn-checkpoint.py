import gym
from gym.utils import seeding
import numpy as np
import pandas as pd
import json
from energy_models import HeatPump, ElectricHeater, EnergyStorage, Building
from pathlib import Path

# Reference Rule-based controller. Used as a baseline to calculate the costs in CityLearn
# It requires, at least, the hour of the day as input state
class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day = states[0]
        
        # Daytime: release stored energy
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 9 and hour_day <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        # Early nightime: store DHW and/or cooling energy
        if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
            a = []
            for i in range(len(self.actions_spaces)):
                if len(self.actions_spaces[i].sample()) == 2:
                    a.append([0.091, 0.091])
                else:
                    a.append([0.091])
   
        self.action_tracker.append(a)
        
        return np.array(a)

def auto_size(buildings):
    for building in buildings:
        
        # Autosize guarantees that the DHW device is large enough to always satisfy the maximum DHW demand
        if building.dhw_heating_device.nominal_power == 'autosize':
            
            # If the DHW device is a HeatPump
            if isinstance(building.dhw_heating_device, HeatPump):
                
                # Calculating COPs of the heat pumps for every hour
                building.dhw_heating_device.cop_heating = building.dhw_heating_device.eta_tech*(building.dhw_heating_device.t_target_heating  + 273.15)/(building.dhw_heating_device.t_target_heating - (building.sim_results['t_out']))
                building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating < 0] = 20.0
                building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating > 20] = 20.0
                
                #We assume that the heat pump is always large enough to meet the highest heating or cooling demand of the building
                building.dhw_heating_device.nominal_power = max(building.sim_results['dhw_demand']/building.dhw_heating_device.cop_heating)
                
            # If the device is an electric heater
            elif isinstance(building.dhw_heating_device, ElectricHeater):
                building.dhw_heating_device.nominal_power = max(building.sim_results['dhw_demand']/building.dhw_heating_device.efficiency)
        
        # Autosize guarantees that the cooling device device is large enough to always satisfy the maximum DHW demand
        if building.cooling_device.nominal_power == 'autosize':
            building.cooling_device.cop_cooling = building.cooling_device.eta_tech*(building.cooling_device.t_target_cooling + 273.15)/(building.sim_results['t_out'] - building.cooling_device.t_target_cooling)
            building.cooling_device.cop_cooling[building.cooling_device.cop_cooling < 0] = 20.0
            building.cooling_device.cop_cooling[building.cooling_device.cop_cooling > 20] = 20.0

            building.cooling_device.nominal_power = max(building.sim_results['cooling_demand']/building.cooling_device.cop_cooling)
        
        # Defining the capacity of the storage devices as a number of times the maximum demand
        building.dhw_storage.capacity = max(building.sim_results['dhw_demand'])*building.dhw_storage.capacity
        building.cooling_storage.capacity = max(building.sim_results['cooling_demand'])*building.cooling_storage.capacity
        
        # Done in order to avoid dividing by 0 if the capacity is 0
        if building.dhw_storage.capacity <= 0.00001:
            building.dhw_storage.capacity = 0.00001
        if building.cooling_storage.capacity <= 0.00001:
            building.cooling_storage.capacity = 0.00001
        
        
def building_loader(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions):
    with open(building_attributes) as json_file:
        data = json.load(json_file)

    buildings, observation_spaces, action_spaces = [],[],[]
    for uid, attributes in zip(data, data.values()):
        if uid in building_ids:
            heat_pump = HeatPump(nominal_power = attributes['Heat_Pump']['nominal_power'], 
                                 eta_tech = attributes['Heat_Pump']['technical_efficiency'], 
                                 t_target_heating = attributes['Heat_Pump']['t_target_heating'], 
                                 t_target_cooling = attributes['Heat_Pump']['t_target_cooling'])

            electric_heater = ElectricHeater(nominal_power = attributes['Electric_Water_Heater']['nominal_power'], 
                                             efficiency = attributes['Electric_Water_Heater']['efficiency'])

            chilled_water_tank = EnergyStorage(capacity = attributes['Chilled_Water_Tank']['capacity'],
                                               loss_coeff = attributes['Chilled_Water_Tank']['loss_coefficient'])

            dhw_tank = EnergyStorage(capacity = attributes['DHW_Tank']['capacity'],
                                     loss_coeff = attributes['DHW_Tank']['loss_coefficient'])

            building = Building(buildingId = uid, dhw_storage = dhw_tank, cooling_storage = chilled_water_tank, dhw_heating_device = electric_heater, cooling_device = heat_pump)

            data_file = str(uid) + '.csv'
            simulation_data = data_path / data_file
            with open(simulation_data) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['cooling_demand'] = data['Cooling Load [kWh]']
            building.sim_results['dhw_demand'] = data['DHW Heating [kWh]']
            building.sim_results['non_shiftable_load'] = data['Equipment Electric Power [kWh]']
            building.sim_results['month'] = data['Month']
            building.sim_results['day'] = data['Day Type']
            building.sim_results['hour'] = data['Hour']
            building.sim_results['daylight_savings_status'] = data['Daylight Savings Status']
            building.sim_results['t_in'] = data['Indoor Temperature [C]']
            building.sim_results['avg_unmet_setpoint'] = data['Average Unmet Cooling Setpoint Difference [C]']
            building.sim_results['rh_in'] = data['Indoor Relative Humidity [%]']
            
            with open(weather_file) as csv_file:
                weather_data = pd.read_csv(csv_file)
                
            building.sim_results['t_out'] = weather_data['Outdoor Drybulb Temperature [C]']
            building.sim_results['rh_out'] = weather_data['Outdoor Relative Humidity [%]']
            building.sim_results['diffuse_solar_rad'] = weather_data['Diffuse Solar Radiation [W/m2]']
            building.sim_results['direct_solar_rad'] = weather_data['Direct Solar Radiation [W/m2]']
            
            # Reading weather forecasts
            building.sim_results['t_out_pred_6h'] = weather_data['6h Prediction Outdoor Drybulb Temperature [C]']
            building.sim_results['t_out_pred_12h'] = weather_data['12h Prediction Outdoor Drybulb Temperature [C]']
            building.sim_results['t_out_pred_24h'] = weather_data['24h Prediction Outdoor Drybulb Temperature [C]']
            
            building.sim_results['rh_out_pred_6h'] = weather_data['6h Prediction Outdoor Relative Humidity [%]']
            building.sim_results['rh_out_pred_12h'] = weather_data['12h Prediction Outdoor Relative Humidity [%]']
            building.sim_results['rh_out_pred_24h'] = weather_data['24h Prediction Outdoor Relative Humidity [%]']
            
            building.sim_results['diffuse_solar_rad_pred_6h'] = weather_data['6h Prediction Diffuse Solar Radiation [W/m2]']
            building.sim_results['diffuse_solar_rad_pred_12h'] = weather_data['12h Prediction Diffuse Solar Radiation [W/m2]']
            building.sim_results['diffuse_solar_rad_pred_24h'] = weather_data['24h Prediction Diffuse Solar Radiation [W/m2]']
            
            building.sim_results['direct_solar_rad_pred_6h'] = weather_data['6h Prediction Direct Solar Radiation [W/m2]']
            building.sim_results['direct_solar_rad_pred_12h'] = weather_data['12h Prediction Direct Solar Radiation [W/m2]']
            building.sim_results['direct_solar_rad_pred_24h'] = weather_data['24h Prediction Direct Solar Radiation [W/m2]']
            
            # Reading the building attributes
            building.building_type = attributes['Building_Type']
            building.climate_zone = attributes['Climate_Zone']
            building.solar_power_capacity = attributes['Solar_Power_Installed(kW)']

            with open(solar_profile) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['solar_gen'] = attributes['Solar_Power_Installed(kW)']*data['Hourly Data: AC inverter power (W)']/1000
            
            # Finding the max and min possible values of all the states, which can then be used by the RL agent to scale the states and train any function approximators more effectively
            s_low, s_high = [], []
            for state_name, value in zip(buildings_states_actions[uid]['states'], buildings_states_actions[uid]['states'].values()):
                if value == True:
                    if state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                        s_low.append(building.sim_results[state_name].min())
                        s_high.append(building.sim_results[state_name].max())
                    else:
                        s_low.append(0.0)
                        s_high.append(1.0)
            
            a_low, a_high = [], []         
            for state_name, value in zip(buildings_states_actions[uid]['actions'], buildings_states_actions[uid]['actions'].values()):
                if value == True:
                    a_low.append(0.0)
                    a_high.append(1.0)

            building.set_state_space(np.array(s_high), np.array(s_low))
            building.set_action_space(np.array(a_high), np.array(a_low))
            
            observation_spaces.append(building.observation_space)
            action_spaces.append(building.action_space)
            buildings.append(building)
        
    auto_size(buildings)

    return buildings, observation_spaces, action_spaces

class CityLearn(gym.Env):  
    def __init__(self, data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = None, simulation_period = (0,8759), cost_function = ['quadratic']):
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)
        
        self.buildings_states_actions_filename = buildings_states_actions
        self.building_attributes = building_attributes
        self.solar_profile = solar_profile
        self.building_ids = building_ids
        self.cost_function = cost_function
        self.cost_rbc = None
        self.data_path = data_path
        self.weather_file = weather_file
        
        self.buildings, self.observation_spaces, self.action_spaces = building_loader(data_path, building_attributes, weather_file, solar_profile, building_ids, self.buildings_states_actions)
        
        self.simulation_period = simulation_period
        self.uid = None
        self.n_buildings = len(self.buildings)
        self.reset()
        
    def get_state_action_spaces(self):
        return self.observation_spaces, self.action_spaces
        
    def next_hour(self):
        self.time_step = next(self.hour)
        for building in self.buildings:
            building.time_step = self.time_step
            
    def get_building_information(self):
        
        # Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand
        building_info = {}
        for building in self.buildings:
            building_info[building.buildingId] = {}
            building_info[building.buildingId]['building_type'] = building.building_type
            building_info[building.buildingId]['climate_zone'] = building.climate_zone
            building_info[building.buildingId]['solar_power_capacity (kW)'] = building.solar_power_capacity
            building_info[building.buildingId]['Annual_DHW_demand (kWh)'] = building.sim_results['dhw_demand'].sum()
            building_info[building.buildingId]['Annual_cooling_demand (kWh)'] = building.sim_results['cooling_demand'].sum()
            building_info[building.buildingId]['Annual_nonshiftable_electrical_demand (kWh)'] = building.sim_results['non_shiftable_load'].sum()
            
            building_info[building.buildingId]['Correlations_DHW'] = {}
            building_info[building.buildingId]['Correlations_cooling_demand'] = {}
            building_info[building.buildingId]['Correlations_non_shiftable_load'] = {}
            for building_corr in self.buildings:
                if building_corr != building:
                    building_info[building.buildingId]['Correlations_DHW'][building_corr.buildingId] = building.sim_results['dhw_demand'].corr(building_corr.sim_results['dhw_demand'])
                    building_info[building.buildingId]['Correlations_cooling_demand'][building_corr.buildingId] = building.sim_results['cooling_demand'].corr(building_corr.sim_results['cooling_demand'])
                    building_info[building.buildingId]['Correlations_non_shiftable_load'][building_corr.buildingId] = building.sim_results['non_shiftable_load'].corr(building_corr.sim_results['non_shiftable_load'])
                    
        return building_info
        
    def step(self, actions):
        
        assert len(actions) == self.n_buildings, "The length of the list of actions should match the length of the list of buildings."
        
        rewards = []
        self.state = []
        electric_demand = 0
        elec_consumption_dhw_storage = 0
        elec_consumption_cooling_storage = 0
        elec_consumption_dhw_building = 0
        elec_consumption_cooling_building = 0
        elec_consumption_appliances = 0
        elec_generation = 0
        for a, building in zip(actions,self.buildings):
            uid = building.buildingId
            assert sum(list(self.buildings_states_actions[uid]['actions'].values())) == len(a), "The number of input actions for building "+str(uid)+" must match the number of actions defined in the list of building attributes."
            
            building_electric_demand = 0

            if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                
                # Cooling
                building_electric_demand += building.set_storage_cooling(a[0])
                elec_consumption_cooling_storage += building.electricity_consumption_cooling_storage
                if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                    
                    # DHW
                    building_electric_demand += building.set_storage_heating(a[1])
                    elec_consumption_dhw_storage += building.electricity_consumption_dhw_storage
            else:
                
                # DHW
                building_electric_demand += building.set_storage_heating(a[0])
                elec_consumption_dhw_storage += building.electricity_consumption_dhw_storage

            # Electrical appliances
            building_electric_demand += building.get_non_shiftable_load()
            
            # Solar generation
            building_electric_demand -= building.get_solar_power()
            
            elec_consumption_cooling_building += building.get_cooling_electric_demand()
            elec_consumption_dhw_building += building.get_dhw_electric_demand() 
            elec_consumption_appliances += building.get_non_shiftable_load()
            elec_generation += building.get_solar_power()
                                    
            # Electricity consumed by every building
            rewards.append(-building_electric_demand)    
            
            # Total electricity consumption
            electric_demand += building_electric_demand
            
        self.next_hour()
            
        for a, building in zip(actions,self.buildings):
            uid = building.buildingId
            
            # Possible states: type of day, hour of day, daylight savings status, outdoor temperature, outdoor Relative Humidity, diffuse solar radiation, direct solar radiation, average indoor temperature, average unmet temperature setpoint difference, average indoor relative humidity, state of charge of cooling device, state of charge of DHW device.
            s = []
            for state_name, value in zip(self.buildings_states_actions[uid]['states'], self.buildings_states_actions[uid]['states'].values()):
                if value == True:
                    if state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                        s.append(building.sim_results[state_name][self.time_step])
                    elif state_name == 'cooling_storage_soc':
                        s.append(building.cooling_storage.soc/building.cooling_storage.capacity)
                    elif state_name == 'dhw_storage_soc':
                        s.append(building.dhw_storage.soc/building.dhw_storage.capacity)
        
            self.state.append(np.array(s))
            
        self.net_electric_consumption = np.append(self.net_electric_consumption,electric_demand)
        self.electric_consumption_dhw_storage = np.append(self.electric_consumption_dhw_storage,elec_consumption_dhw_storage)
        self.electric_consumption_cooling_storage = np.append(self.electric_consumption_cooling_storage,elec_consumption_cooling_storage)
        self.electric_consumption_dhw = np.append(self.electric_consumption_dhw,elec_consumption_dhw_building)
        self.electric_consumption_cooling = np.append(self.electric_consumption_cooling,elec_consumption_cooling_building)
        self.electric_consumption_appliances = np.append(self.electric_consumption_appliances,elec_consumption_appliances)
        self.electric_generation = np.append(self.electric_generation,elec_generation)
        self.net_electric_consumption_no_storage = np.append(self.net_electric_consumption_no_storage,electric_demand-elec_consumption_cooling_storage-elec_consumption_dhw_storage)
        self.net_electric_consumption_no_pv_no_storage = np.append(self.net_electric_consumption_no_pv_no_storage,electric_demand + elec_generation - elec_consumption_cooling_storage - elec_consumption_dhw_storage)

        terminal = self._terminal()
        return (self._get_ob(), rewards, terminal, {})
    
    def reset(self):
        
        #Initialization of variables
        self.hour = iter(np.array(range(self.simulation_period[0], self.simulation_period[1] + 1)))
        self.next_hour()
            
        self.net_electric_consumption = np.array([])
        self.net_electric_consumption_no_storage = np.array([])
        self.net_electric_consumption_no_pv_no_storage = np.array([])
        self.electric_consumption_dhw_storage = np.array([])
        self.electric_consumption_cooling_storage = np.array([])
        self.electric_consumption_dhw = np.array([])
        self.electric_consumption_cooling = np.array([])
        self.electric_consumption_appliances = np.array([])
        self.electric_generation = np.array([])
        
        
        self.cost_rbc = None
        
        self.state = []
        for building in self.buildings:
            uid = building.buildingId
            s = []
            for state_name, value in zip(self.buildings_states_actions[uid]['states'], self.buildings_states_actions[uid]['states'].values()):
                if value == True:
                    if state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                        s.append(building.sim_results[state_name][self.time_step])
                    elif state_name == 'cooling_storage_soc':
                        s.append(0.0)
                    elif state_name == 'dhw_storage_soc':
                        s.append(0.0)
            
            self.state.append(np.array(s, dtype=np.float32))
            building.reset()
            
        return self._get_ob()
    
    def _get_ob(self):
        return np.array([s for s in [s_var for s_var in self.state]])
    
    def _terminal(self):
        return bool(self.time_step >= self.simulation_period[1])
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def cost(self):
        
        # Running the reference rule-based controller to find the baseline cost
        if self.cost_rbc is None:
            env_rbc = CityLearn(self.data_path, self.building_attributes, self.weather_file, self.solar_profile, self.building_ids, buildings_states_actions = self.buildings_states_actions_filename, simulation_period = self.simulation_period, cost_function = self.cost_function)
            _, actions_spaces = env_rbc.get_state_action_spaces()

            #Instantiatiing the control agent(s)
            agent_rbc = RBC_Agent(actions_spaces)

            state = env_rbc.reset()
            done = False
            while not done:
                action = agent_rbc.select_action([env_rbc.buildings[0].sim_results['hour'][env_rbc.buildings[0].time_step]])
                next_state, rewards, done, _ = env_rbc.step(action)
                state = next_state
            self.cost_rbc = env_rbc.get_baseline_cost()
        
        # Compute the costs normalized by the baseline costs
        cost = {}
        if 'ramping' in self.cost_function:
            cost['ramping'] = np.abs((self.net_electric_consumption - np.roll(self.net_electric_consumption,1))[1:]).sum()/self.cost_rbc['ramping']
            
        # Finds the load factor for every month (average monthly demand divided by its maximum peak), and averages all the load factors across the 12 months. The metric is one minus the load factor.
        if '1-load_factor' in self.cost_function:
            cost['1-load_factor'] = np.mean([1-np.mean(self.net_electric_consumption[i:i+int(8760/12)])/ np.max(self.net_electric_consumption[i:i+int(8760/12)]) for i in range(0,len(self.net_electric_consumption), int(8760/12))])/self.cost_rbc['1-load_factor']
           
        # Average of all the daily peaks of the 365 day of the year. The peaks are calculated using the net energy demand of the whole district of buildings.
        if 'average_daily_peak' in self.cost_function:
            cost['average_daily_peak'] = np.mean([self.net_electric_consumption[i:i+24].max() for i in range(0,len(self.net_electric_consumption),24)])/self.cost_rbc['average_daily_peak']
            
        # Peak demand of the district for the whole year period.
        if 'peak_demand' in self.cost_function:
            cost['peak_demand'] = self.net_electric_consumption.max()/self.cost_rbc['peak_demand']
            
        # Positive net electricity consumption for the whole district. It is clipped at a min. value of 0 because the objective is to minimize the energy consumed in the district, not to profit from the excess generation. (Island operation is therefore incentivized)
        if 'net_electricity_consumption' in self.cost_function:
            cost['net_electricity_consumption'] = self.net_electric_consumption.clip(min=0).sum()/self.cost_rbc['net_electricity_consumption']
            
        # Not used for the challenge
        if 'quadratic' in self.cost_function:
            cost['quadratic'] = (self.net_electric_consumption.clip(min=0)**2).sum()/self.cost_rbc['quadratic']
            
        cost['total'] = np.mean([c for c in cost.values()])
            
        return cost
    
    def get_baseline_cost(self):
        
        # Computes the costs for the Rule-based controller, which are used to normalized the actual costs.
        cost = {}
        if 'ramping' in self.cost_function:
            cost['ramping'] = np.abs((self.net_electric_consumption - np.roll(self.net_electric_consumption,1))[1:]).sum()
            
        if '1-load_factor' in self.cost_function:
            cost['1-load_factor'] = np.mean([1 - np.mean(self.net_electric_consumption[i:i+int(8760/12)])/ np.max(self.net_electric_consumption[i:i+int(8760/12)]) for i in range(0, len(self.net_electric_consumption), int(8760/12))])
           
        if 'average_daily_peak' in self.cost_function:
            cost['average_daily_peak'] = np.mean([self.net_electric_consumption[i:i+24].max() for i in range(0, len(self.net_electric_consumption), 24)])
            
        if 'peak_demand' in self.cost_function:
            cost['peak_demand'] = self.net_electric_consumption.max()
            
        if 'net_electricity_consumption' in self.cost_function:
            cost['net_electricity_consumption'] = self.net_electric_consumption.clip(min=0).sum()
            
        if 'quadratic' in self.cost_function:
            cost['quadratic'] = (self.net_electric_consumption.clip(min=0)**2).sum()
            
        return cost
        


        
