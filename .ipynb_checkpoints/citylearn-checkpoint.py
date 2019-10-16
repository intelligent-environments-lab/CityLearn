import gym
from gym.utils import seeding
import numpy as np
import pandas as pd

class CityLearn(gym.Env):  
    def __init__(self, demand_file, weather_file, buildings = None, time_resolution = 1, simulation_period = (3500,6000)):
        self.action_track = {}
        self.buildings = buildings
        for building in self.buildings:
            uid = building.buildingId
            self.action_track[uid] = []
        self.time_resolution = time_resolution
        self.simulation_period = simulation_period
        self.uid = None
        self.n_buildings = len(buildings)
        self.reset()
        
    def next_hour(self):
        self.time_step = [next(self.hour) for j in range(self.time_resolution)][-1]
        for building in self.buildings:
            building.time_step = self.time_step
        
    def step(self, actions):
        assert len(actions) == self.n_buildings, "The length of the list of actions should match the length of the list of buildings."
        self.next_hour()
        rewards = []
        self.state = []
        electric_demand = 0
        for a_bld, building in zip(actions,self.buildings):
            uid = building.buildingId
            building_electric_demand = 0
            for a in a_bld:
                a = a/self.time_resolution
                self.action_track[uid].append(a)
            
                reward = 0
                
                for _ in range(self.time_resolution):                
                    #Heating
                    building_electric_demand += building.set_storage_heating(0)
                    #Cooling
                    building_electric_demand += building.set_storage_cooling(a)
                    
            #Electricity consumed by every building
            rewards.append(-building_electric_demand)    
            
            #Total electricity consumption
            electric_demand += building_electric_demand
            
            #States: hour, Tout, Tin, Thermal_energy_stored    
            s1 = building.sim_results['hour'][self.time_step]
            s2 = building.sim_results['t_out'][self.time_step]
            s3 = building.cooling_storage.soc/building.cooling_storage.capacity
        
            self.state.append(np.array([s1, s2, s3]))
            
        self.total_electric_consumption.append(electric_demand)
        
        terminal = self._terminal()
        return (self._get_ob(), rewards, terminal, {})
    
    def reset(self):
        #Initialization of variables
        self.hour = iter(np.array(range(self.simulation_period[0], self.simulation_period[1] + 1)))
        self.time_step = next(self.hour)
        for building in self.buildings:
            building.time_step = self.time_step
            
        self.total_electric_consumption = []
        
        self.state = []
        for building in self.buildings:
            uid = building.buildingId
            s1 = building.sim_results['hour'][self.time_step]
            s2 = building.sim_results['t_out'][self.time_step]
            s3 = 0.0
            self.state.append(np.array([s1,s2,s3], dtype=np.float32))
            building.reset()
        return self._get_ob()
    
    def _get_ob(self):
        return np.array([s for s in [s_var for s_var in self.state]], dtype=np.float32)
    
    def _terminal(self):
        return bool(self.time_step >= self.simulation_period[1])
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def cost(self):
        return np.sqrt((np.array(self.total_electric_consumption)**2).sum())


def building_loader(demand_file, weather_file, buildings):
    demand = pd.read_csv(demand_file,sep="\t")
    weather = pd.read_csv(weather_file,sep="\t")
    weather = weather.rename(columns = {' Ta':'Ta',' h':'h'})

    i = 0
    building_ids = {}
    for building in buildings:
        building_ids[building.buildingId] = i
        i += 1
        
    #Obtaining all the building IDs from the header of the document with the results
    ids = []
    for col_name in demand.columns:
        if "(" in col_name:
            ids.append(col_name.split('(')[0])

    #Creating a list with unique values of the IDs
    unique_id_str = sorted(list(set(ids)))
    unique_id = [int(i) for i in unique_id_str if int(i) in building_ids]
    unique_id_file = ["(" + str(uid) for uid in unique_id]
    
    #Filling out the variables of the buildings with the output files from CitySim
    for uid, building_id in zip(unique_id, unique_id_file):
        building_list_indx = building_ids[uid]
        sim_var_name = {sim_var: [col for col in demand.columns if building_id in col if sim_var in col] for sim_var in list(['Qs','ElectricConsumption','Ta'])}

        #Summing up and averaging the values by the number of floors of the building
        qs = demand[sim_var_name['Qs']]
        buildings[building_list_indx].sim_results['cooling_demand'] = -qs[qs<0].sum(axis = 1)/1000
        buildings[building_list_indx].sim_results['heating_demand'] = qs[qs>0].sum(axis = 1)/1000
        buildings[building_list_indx].sim_results['non_shiftable_load'] = demand[sim_var_name['ElectricConsumption']]
        buildings[building_list_indx].sim_results['t_in'] = demand[sim_var_name['Ta']].mean(axis = 1)
        buildings[building_list_indx].sim_results['t_out'] = weather['Ta']
        buildings[building_list_indx].sim_results['hour'] = weather['h']

def auto_size(buildings, t_target_heating, t_target_cooling):  
    for building in buildings:
        #Calculating COPs of the heat pumps for every hour
        building.heating_device.cop_heating = building.heating_device.eta_tech*building.heating_device.t_target_heating/(building.heating_device.t_target_heating - (building.sim_results['t_out'] + 273.15))
        building.heating_device.cop_heating[building.heating_device.cop_heating < 0] = 20.0
        building.heating_device.cop_heating[building.heating_device.cop_heating > 20] = 20.0
        building.cooling_device.cop_cooling = building.cooling_device.eta_tech*building.cooling_device.t_target_cooling/(building.sim_results['t_out'] + 273.15 - building.cooling_device.t_target_cooling)
        building.cooling_device.cop_cooling[building.cooling_device.cop_cooling < 0] = 20.0
        building.cooling_device.cop_cooling[building.cooling_device.cop_cooling > 20] = 20.0
        
        #We assume that the heat pump is large enough to meet the highest heating or cooling demand of the building (Tindoor always satisfies Tsetpoint)
        building.heating_device.nominal_power = max(building.sim_results['heating_demand']/building.heating_device.cop_heating)
        building.cooling_device.nominal_power = max(building.sim_results['cooling_demand']/building.cooling_device.cop_cooling)
        
        #Defining the capacity of the heat tanks based on the average non-zero heat demand
        building.heating_storage.capacity = max(building.sim_results['heating_demand'])*3
        building.cooling_storage.capacity = max(building.sim_results['cooling_demand'])*3
