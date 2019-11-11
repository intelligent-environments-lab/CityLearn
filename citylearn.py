import gym
from gym.utils import seeding
import numpy as np
import pandas as pd
import json
from energy_models import HeatPump, ElectricHeater, EnergyStorage, Building

def auto_size(buildings):  
    for building in buildings:
        
        if building.dhw_heating_device.nominal_power == 'autosize':
            #Calculating COPs of the heat pumps for every hour
            if isinstance(building.dhw_heating_device, HeatPump):
                building.dhw_heating_device.cop_heating = building.dhw_heating_device.eta_tech*building.dhw_heating_device.t_target_heating/(building.dhw_heating_device.t_target_heating - (building.sim_results['t_out'] + 273.15))
                building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating < 0] = 20.0
                building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating > 20] = 20.0
                #We assume that the heat pump is large enough to meet the highest heating or cooling demand of the building (Tindoor always satisfies Tsetpoint)
                building.dhw_heating_device.nominal_power = max(building.sim_results['dhw_demand']/building.dhw_heating_device.cop_heating)
            else:
                building.dhw_heating_device.nominal_power = max(building.sim_results['dhw_demand']/building.dhw_heating_device.efficiency)
            
        if building.cooling_device.nominal_power == 'autosize':
            building.cooling_device.cop_cooling = building.cooling_device.eta_tech*building.cooling_device.t_target_cooling/(building.sim_results['t_out'] + 273.15 - building.cooling_device.t_target_cooling)
            building.cooling_device.cop_cooling[building.cooling_device.cop_cooling < 0] = 20.0
            building.cooling_device.cop_cooling[building.cooling_device.cop_cooling > 20] = 20.0

            #We assume that the heat pump is large enough to meet the highest heating or cooling demand of the building (Tindoor always satisfies Tsetpoint)
            building.cooling_device.nominal_power = max(building.sim_results['cooling_demand']/building.cooling_device.cop_cooling)
        
        #Defining the capacity of the heat tanks based on the average non-zero heat demand
        if building.dhw_storage.capacity == 'autosize':
            building.dhw_storage.capacity = max(building.sim_results['dhw_demand'])*3
        if building.cooling_storage.capacity == 'autosize':    
            building.cooling_storage.capacity = max(building.sim_results['cooling_demand'])*3
        
        
def building_loader(building_attributes, solar_profile, building_ids, buildings_states_actions):
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

            with open('data//'+uid+'.csv') as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['cooling_demand'] = data['Cooling Load [kWh]']
            building.sim_results['dhw_demand'] = data['DHW Heating [kWh]']
            building.sim_results['non_shiftable_load'] = data['Equipment Electric Power [kWh]']
            building.sim_results['day'] = data['Day Type']
            building.sim_results['hour'] = data['Hour']
            building.sim_results['daylight_savings_status'] = data['Daylight Savings Status']
            building.sim_results['t_out'] = data['Outdoor Drybulb Temperature [C]']
            building.sim_results['rh_out'] = data['Outdoor Relative Humidity [%]']
            building.sim_results['diffuse_solar_rad'] = data['Diffuse Solar Radiation [W/m2]']
            building.sim_results['direct_solar_rad'] = data['Direct Solar Radiation [W/m2]']
            building.sim_results['t_in'] = data['Indoor Temperature [C]']
            building.sim_results['avg_unmet_setpoint'] = data['Average Unmet Cooling Setpoint Difference [C]']
            building.sim_results['rh_in'] = data['Indoor Relative Humidity [%]']

            with open(solar_profile) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['solar_gen'] = attributes['Solar_Power_Installed(kW)']*data['Hourly Data: AC inverter power (W)']/1000
            
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
    def __init__(self, building_attributes, solar_profile, building_ids, buildings_states_actions = None, simulation_period = (0,8759), cost_function = ['quadratic']):
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)
         
        self.cost_function = cost_function
        self.buildings, self.observation_spaces, self.action_spaces = building_loader(building_attributes, solar_profile, building_ids, self.buildings_states_actions)
        self.action_track = {}
        for building in self.buildings:
            uid = building.buildingId
            self.action_track[uid] = []
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
        
    def step(self, actions):
        assert len(actions) == self.n_buildings, "The length of the list of actions should match the length of the list of buildings."
        
        self.next_hour()
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
            assert sum(list(self.buildings_states_actions[uid]['actions'].values())) == len(a)
            
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
                
            self.action_track[uid].append(a)
                    
            #Electricity consumed by every building
            rewards.append(-building_electric_demand)    
            
            #Total electricity consumption
            electric_demand += building_electric_demand
            
            #Possible states: type of day, hour of day, daylight savings status, outdoor temperature, outdoor Relative Humidity, diffuse solar radiation, direct solar radiation, average indoor temperature, average unmet temperature setpoint difference, average indoor relative humidity, state of charge of cooling device, state of charge of DHW device.
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
        
        terminal = self._terminal()
        return (self._get_ob(), rewards, terminal, {})
    
    def reset(self):
        #Initialization of variables
        self.hour = iter(np.array(range(self.simulation_period[0], self.simulation_period[1] + 1)))
        self.time_step = next(self.hour)
        for building in self.buildings:
            building.time_step = self.time_step
            
        self.net_electric_consumption = np.array([])
        self.electric_consumption_dhw_storage = np.array([])
        self.electric_consumption_cooling_storage = np.array([])
        self.electric_consumption_dhw = np.array([])
        self.electric_consumption_cooling = np.array([])
        self.electric_consumption_appliances = np.array([])
        self.electric_generation = np.array([])
        
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
        cost = 0
        
        if 'quadratic' in self.cost_function:
            cost += np.sqrt((self.net_electric_consumption.clip(min=0)**2).sum())
            
        if 'ramping' in self.cost_function:
            pass
            
        if 'load_factor' in self.cost_function:
            pass
           
        if 'peak_to_valley' in self.cost_function:
            pass
            
        if 'peak_demand' in self.cost_function:
            pass
            
        if 'net_electricity_consumption' in self.cost_function:
            pass
            
        return cost


        
