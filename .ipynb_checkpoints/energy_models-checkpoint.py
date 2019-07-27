import gym
from gym import core, spaces
import numpy as np

class HeatPump:
    def __init__(self, nominal_power = None, eta_tech = None, t_target_heating = None, t_target_cooling = None):
        """
        Args:
            nominal_power (float): Maximum amount of electric power that the heat pump can consume from the power grid (given by the nominal power of the compressor)
            eta_tech (float): Technical efficiency
            t_target_heating (float): Temperature of the sink where the heating energy is released
            t_target_cooling (float): Temperature of the sink where the cooling energy is released
        """
        
        #Parameters
        self.nominal_power = nominal_power
        self.eta_tech = eta_tech
        
        #Variables
        self.max_cooling = None
        self.max_heating = None
        self.cop_heating = None
        self.cop_cooling = None
        self.t_target_heating = t_target_heating
        self.t_target_cooling = t_target_cooling
        self.t_source_heating = None
        self.t_source_cooling = None
        self.cop_heating_list = []
        self.cop_cooling_list = []
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
                   
    def get_max_cooling_power(self, max_electric_power = None, t_source_cooling = None, t_target_cooling = None):
        """
        Args:
            max_electric_power (float): Maximum amount of electric power that the heat pump can consume from the power grid
            t_source_cooling (float): Temperature of the sisource from where the cooling energy is taken
            t_target_cooling (float): Temperature of the sink where the cooling energy will be released
            
        Returns:
            max_cooling (float): maximum amount of cooling energy that the heatpump can provide
        """
        
        if t_target_cooling is not None:
            self.t_target_cooling = t_target_cooling
            
        if t_source_cooling is not None:
            self.t_source_cooling = t_source_cooling

        #Caping the COP (coefficient of performance) to 1.0 - 20.0
        if self.t_source_cooling - self.t_target_cooling > 0.01:
            self.cop_cooling = self.eta_tech*(self.t_target_cooling + 273.15)/(self.t_source_cooling - self.t_target_cooling)
        else:
            self.cop_cooling = 20.0
        self.cop_cooling = max(min(self.cop_cooling, 20.0), 1.0)
        
        self.cop_cooling_list.append(self.cop_cooling)
        if max_electric_power is None:
            self.max_cooling = self.nominal_power*self.cop_cooling
        else:
            self.max_cooling = min(max_electric_power, self.nominal_power)*self.cop_cooling
        return self.max_cooling
    
    def get_max_heating_power(self, max_electric_power = None, t_source_heating = None, t_target_heating = None):
        """Method that calculates the heating COP and the maximum heating power available
        Args:
            max_electric_power (float): Maximum amount of electric power that the heat pump can consume from the power grid
            t_source_heating (float): Temperature of the source from where the heating energy is taken
            t_target_heating (float): Temperature of the sink where the heating energy will be released
            
        Returns:
            max_heating (float): maximum amount of heating energy that the heatpump can provide
        """
        
        if t_target_heating is not None:
            self.t_target_heating = t_target_heating
            
        if t_source_heating is not None:
            self.t_source_heating = t_source_heating
        
        #Caping the COP (coefficient of performance) to 1.0 - 20.0
        if self.t_target_heating - self.t_source_heating > 0.01:
            self.cop_heating = self.eta_tech*(self.t_target_heating + 273.15)/(self.t_target_heating - self.t_source_heating)
        else:
            self.cop_heating = 20.0
        
        self.cop_heating = max(min(self.cop_heating, 20.0), 1.0)
        self.cop_heating_list.append(self.cop_heating)
        if max_electric_power is None:
            self.max_heating = self.nominal_power*self.cop_heating
        else:
            self.max_heating = min(max_electric_power, self.nominal_power)*self.cop_heating
        return self.max_heating
            
    def get_electric_consumption_cooling(self, cooling_supply = 0):
        """Method that calculates the cooling COP and the maximum cooling power available
        Args:
            cooling_supply (float): Amount of cooling energy that the heat pump is going to supply
            
        Returns:
            _elec_consumption_cooling (float): electricity consumption for cooling
        """
        
        _elec_consumption_cooling = cooling_supply/self.cop_cooling
        self.electrical_consumption_cooling.append(_elec_consumption_cooling)
        return _elec_consumption_cooling
    
    def get_electric_consumption_heating(self, heat_supply = 0):
        """
        Args:
            cooling_supply (float): Amount of cooling energy that the heat pump is going to supply
            
        Returns:
            _elec_consumption_heating (float): electricity consumption for heating
        """
            
        _elec_consumption_heating = heat_supply/self.cop_heating
        self.electrical_consumption_heating.append(_elec_consumption_heating)
        return _elec_consumption_heating
    
    def reset(self):
        self.t_source_heating = None
        self.t_source_cooling = None
        self.max_cooling = None
        self.max_heating = None
        self.cop_heating = None
        self.cop_cooling = None
        self.cop_heating_list = []
        self.cop_cooling_list = []
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
       
    
class EnergyStorage:
    def __init__(self, capacity = None, max_power_output = None, max_power_charging = None, efficiency = 1, loss_coeff = 0):
        self.capacity = capacity
        self.max_power_output = max_power_output
        self.max_power_charging = max_power_charging
        self.efficiency = efficiency
        self.loss_coeff = loss_coeff
        self.soc_list = [0]
        self.soc = 0 #State of charge
        self.energy_balance_list = [0] #Positive for energy entering the storage
        self.energy_balance = 0
        
    def charge(self, energy):
        soc_init = self.soc*(1-self.loss_coeff)
        losses =  - energy*(1 - self.efficiency)
        energy = energy/self.efficiency
        
        if self.max_power_charging is not None:
            energy =  min(energy, self.max_power_charging)
            
        if self.max_power_output is not None:
            energy = max(-max_power_output, energy)
            
        self.soc = max(0, soc_init + energy*self.efficiency)  
        if self.capacity is not None:
            self.soc = min(self.soc, self.capacity)
            
        self.energy_balance = self.soc - soc_init - losses
        self.energy_balance_list.append(self.energy_balance)
        self.soc_list.append(self.soc)
        return self.energy_balance
    
    def reset(self):
        self.soc_list = [0]
        self.soc = 0 #State of charge
        self.energy_balance_list = [0] #Positive for energy entering the storage
        self.energy_balance = 0
		
class Building:  
    def __init__(self, buildingId, heating_storage = None, cooling_storage = None, electrical_storage = None, heating_device = None, cooling_device = None):
        self.buildingId = buildingId
        self.heating_storage = heating_storage
        self.cooling_storage = cooling_storage
        self.electrical_storage = electrical_storage
        self.heating_device = heating_device
        self.cooling_device = cooling_device
        self.observation_spaces = None
        self.action_spaces = None
        self.time_step = 0
        self.sim_results = {} #'cooling_demand','heating_demand','non_shiftable_load','t_in','t_out','hour'
        self.electricity_consumption_heating = []
        self.electricity_consumption_cooling = []
        self.electricity_consumption_hvac = []
        
    def state_action_space(self, high_state, low_state, max_action, min_action):
        #Defining state space: hour, Tout, Tin, Thermal_energy_stored
        self.observation_spaces = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

        #Defining action space: new desired energy stored in the tank
        self.action_spaces = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

    def set_storage_heating(self, action):
        heat_power_avail = self.heating_device.get_max_heating_power(t_source_heating = self.sim_results['t_out'][self.time_step]) - self.sim_results['heating_demand'][self.time_step]
        heating_energy_balance = self.heating_storage.charge(min(heat_power_avail, action*self.heating_storage.capacity)) 
        heating_energy_balance = max(0,heating_energy_balance)
        elec_demand_heating = self.heating_device.get_electric_consumption_heating(heat_supply = heating_energy_balance)
        self.electricity_consumption_heating.append(elec_demand_heating)
        return elec_demand_heating
        
    def set_storage_cooling(self, action):
        cooling_power_avail = self.cooling_device.get_max_cooling_power(t_source_cooling = self.sim_results['t_out'][self.time_step]) - self.sim_results['cooling_demand'][self.time_step]
        cooling_energy_balance = self.cooling_storage.charge(min(cooling_power_avail, action*self.cooling_storage.capacity)) 
        cooling_energy_balance = max(0,cooling_energy_balance)
        elec_demand_cooling = self.cooling_device.get_electric_consumption_cooling(cooling_supply = cooling_energy_balance)
        self.electricity_consumption_cooling.append(elec_demand_cooling)
        return elec_demand_cooling
    
    def reset(self):
        self.time_step = 0
        self.electricity_consumption_heating = []
        self.electricity_consumption_cooling = []
        self.electricity_consumption_hvac = []
        if self.heating_storage is not None:
            self.heating_storage.reset()
        if self.cooling_storage is not None:
            self.cooling_storage.reset()
        if self.electrical_storage is not None:
            self.electrical_storage.reset()
        if self.heating_device is not None:
            self.heating_device.reset()
        if self.cooling_device is not None:
            self.cooling_device.reset()