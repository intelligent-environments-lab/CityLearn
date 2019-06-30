import gym
import numpy as np
import pandas as pd

class CityLearn(gym.Env):  
    def __init__(self, demand_file, weather_file, buildings = None, time_resolution = 1, simulation_period = (3500,6000)):
        self.time_resolution = time_resolution
        self.buildings = buildings
        self.simulation_period = simulation_period
        self.hour = iter(np.array(range(simulation_period[0], simulation_period[1] + 1)))
        self.time_step = next(self.hour)
        self.total_electric_consumption = []
        self.action_track = {}
        self.uid = None
        self.n_buildings = 0
        for uid in buildings:
            self.action_track[uid] = []
            self.n_buildings += 1
            self.last_building_uid = uid
        
    def __call__(self, current_building_id):
        self.uid = current_building_id
        
    def next_hour(self):
        self.time_step = [next(self.hour) for j in range(self.time_resolution)][-1]
        for uid in self.buildings:
            self.buildings[uid].time_step = self.time_step
        
    def step(self, action):
        action = action/self.time_resolution
        uid = self.uid
                
        self.action_track[uid].append(action)
        electric_demand = 0
        reward = 0
        for i in range(self.time_step, self.time_step + self.time_resolution):                
            #Heating
            electric_demand += self.buildings[uid].set_storage_heating(0)
            #Cooling
            electric_demand += self.buildings[uid].set_storage_cooling(action)

            #Electricity consumed
            reward = reward - electric_demand
            
        self.total_electric_consumption.append(electric_demand)
            
        #States: hour, Tout, Tin, Thermal_energy_stored    
        s1 = self.buildings[uid].sim_results['hour'][i]
        s2 = self.buildings[uid].sim_results['t_out'][i]
        s3 = self.buildings[uid].cooling_storage.soc/self.buildings[uid].cooling_storage.capacity
        self.state = np.array([s1, s2, s3])

        terminal = self._terminal()
        return (self.state, reward, terminal, {})
    
    def reset(self):
        #Initialization of variables
        self.action_track = {}
        self.hour = iter(np.array(range(self.simulation_period[0], self.simulation_period[1] + 1)))
        self.time_step = next(self.hour)
        self.state = np.array([0.0,0.0,0.0], dtype=np.float32)
        self.total_electric_consumption = []
        
        for uid in self.buildings:
            self.buildings[uid].reset()
            self.action_track[uid] = []
            
        return self._get_ob()
    
    def _get_ob(self):
        s = self.state
        return np.array([s[0], s[1], s[2]], dtype=np.float32)
    
    def _terminal(self):
        return bool(self.time_step >= self.simulation_period[1] and self.uid == self.last_building_uid)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
		

def building_loader(demand_file, weather_file, buildings):
    demand = pd.read_csv(demand_file,sep="\t")
    weather = pd.read_csv(weather_file,sep="\t")
    weather = weather.rename(columns = {' Ta':'Ta',' h':'h'})

    #Obtaining all the building IDs from the header of the document with the results
    ids = []
    for col_name in demand.columns:
        if "(" in col_name:
            ids.append(col_name.split('(')[0])

    #Creating a list with unique values of the IDs
    unique_id_str = sorted(list(set(ids)))
    unique_id = [int(i) for i in unique_id_str if int(i) in buildings]
    unique_id_file = ["(" + str(uid) for uid in unique_id]
    
    #Filling out the variables of the buildings with the output files from CitySim
    for uid, building_id in zip(unique_id, unique_id_file):
        sim_var_name = {sim_var: [col for col in demand.columns if building_id in col if sim_var in col] for sim_var in list(['Qs','ElectricConsumption','Ta'])}

        #Summing up and averaging the values by the number of floors of the building
        qs = demand[sim_var_name['Qs']]
        buildings[uid].sim_results['cooling_demand'] = -qs[qs<0].sum(axis = 1)/1000
        buildings[uid].sim_results['heating_demand'] = qs[qs>0].sum(axis = 1)/1000
        buildings[uid].sim_results['non_shiftable_load'] = demand[sim_var_name['ElectricConsumption']]
        buildings[uid].sim_results['t_in'] = demand[sim_var_name['Ta']].mean(axis = 1)
        buildings[uid].sim_results['t_out'] = weather['Ta']
        buildings[uid].sim_results['hour'] = weather['h']

def auto_size(buildings, t_target_heating, t_target_cooling):
    for uid in buildings:
        #Calculating COPs of the heat pumps for every hour
        buildings[uid].heating_device.cop_heating = buildings[uid].heating_device.eta_tech*buildings[uid].heating_device.t_target_heating/(buildings[uid].heating_device.t_target_heating - (buildings[uid].sim_results['t_out'] + 273.15))
        buildings[uid].heating_device.cop_heating[buildings[uid].heating_device.cop_heating < 0] = 20.0
        buildings[uid].heating_device.cop_heating[buildings[uid].heating_device.cop_heating > 20] = 20.0
        buildings[uid].cooling_device.cop_cooling = buildings[uid].cooling_device.eta_tech*buildings[uid].cooling_device.t_target_cooling/(buildings[uid].sim_results['t_out'] + 273.15 - buildings[uid].cooling_device.t_target_cooling)
        buildings[uid].cooling_device.cop_cooling[buildings[uid].cooling_device.cop_cooling < 0] = 20.0
        buildings[uid].cooling_device.cop_cooling[buildings[uid].cooling_device.cop_cooling > 20] = 20.0
        
        #We assume that the heat pump is large enough to meet the highest heating or cooling demand of the building (Tindoor always satisfies Tsetpoint)
        buildings[uid].heating_device.nominal_power = max(buildings[uid].sim_results['heating_demand']/(buildings[uid].heating_device.cop_heating))
        buildings[uid].cooling_device.nominal_power = max(buildings[uid].sim_results['cooling_demand']/(buildings[uid].cooling_device.cop_cooling))
        
        #Defining the capacity of the heat tanks based on the average non-zero heat demand
        buildings[uid].heating_storage.capacity = max(buildings[uid].sim_results['heating_demand']/(buildings[uid].heating_device.cop_heating))*3
        buildings[uid].cooling_storage.capacity = max(buildings[uid].sim_results['cooling_demand']/(buildings[uid].cooling_device.cop_cooling))*7
