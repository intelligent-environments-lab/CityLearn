from gym import spaces
import numpy as np

class Building:  
    def __init__(self, buildingId, dhw_storage = None, cooling_storage = None, electrical_storage = None, dhw_heating_device = None, cooling_device = None, save_memory = True):
        """
        Args:
            buildingId (int)
            dhw_storage (EnergyStorage)
            cooling_storage (EnergyStorage)
            electrical_storage (Battery)
            dhw_heating_device (ElectricHeater or HeatPump)
            cooling_device (HeatPump)
        """
        
        # Building attributes
        self.building_type = None
        self.climate_zone = None
        self.solar_power_capacity = None
        
        self.buildingId = buildingId
        self.dhw_storage = dhw_storage
        self.cooling_storage = cooling_storage
        self.electrical_storage = electrical_storage
        self.dhw_heating_device = dhw_heating_device
        self.cooling_device = cooling_device
        self.observation_space = None
        self.action_space = None
        self.time_step = 0
        self.sim_results = {}
        self.save_memory = save_memory
        
        if self.dhw_storage is not None:
            self.dhw_storage.reset()
        if self.cooling_storage is not None:
            self.cooling_storage.reset()
        if self.electrical_storage is not None:
            self.electrical_storage.reset()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.reset()
        if self.cooling_device is not None:
            self.cooling_device.reset()
            
        self._electric_consumption_cooling_storage = 0.0
        self._electric_consumption_dhw_storage = 0.0
        
        self.cooling_demand_building = []
        self.dhw_demand_building = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
           
        self.electric_consumption_cooling = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_dhw_storage = []
        
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        
        self.cooling_device_to_building = []
        self.cooling_storage_to_building = []
        self.cooling_device_to_storage = []
        self.cooling_storage_soc = []

        self.dhw_heating_device_to_building = []
        self.dhw_storage_to_building = []
        self.dhw_heating_device_to_storage = []
        self.dhw_storage_soc = []
        
        self.electrical_storage_electric_consumption = []
        self.electrical_storage_soc = []
        
    def set_state_space(self, high_state, low_state):
        # Setting the state space and the lower and upper bounds of each state-variable
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
    
    def set_action_space(self, max_action, min_action):
        # Setting the action space and the lower and upper bounds of each action-variable
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)
        
    def set_storage_electrical(self, action):
        """
        Args:
            action (float): Amount of heating energy stored (added) in that time-step as a ratio of the maximum capacity of the energy storage device. 
            -1 =< action < 0 : Energy Storage Unit releases energy into the building and its State of Charge decreases
            0 < action <= 1 : Energy Storage Unit receives energy from the energy supply device and its State of Charge increases
            The actions are always subject to the constraints of the power capacity of the heating supply unit, the DHW demand of the
            building (which limits the maximum amount of DHW that the energy storage can provide to the building), and the state of charge of the
            energy storage unit itself
        Return:
            elec_demand_heating (float): electricity consumption needed for space heating and heating storage
        """

        electrical_energy_balance = self.electrical_storage.charge(action*self.electrical_storage.capacity)
        
        if self.save_memory == False:
            self.electrical_storage_electric_consumption.append(electrical_energy_balance)
            self.electrical_storage_soc.append(self.electrical_storage._soc)
        
        self.electrical_storage.time_step += 1
        
        return electrical_energy_balance
    

    def set_storage_heating(self, action):
        """
        Args:
            action (float): Amount of heating energy stored (added) in that time-step as a ratio of the maximum capacity of the energy storage device. 
            -1 =< action < 0 : Energy Storage Unit releases energy into the building and its State of Charge decreases
            0 < action <= 1 : Energy Storage Unit receives energy from the energy supply device and its State of Charge increases
            The actions are always subject to the constraints of the power capacity of the heating supply unit, the DHW demand of the
            building (which limits the maximum amount of DHW that the energy storage can provide to the building), and the state of charge of the
            energy storage unit itself
        Return:
            elec_demand_heating (float): electricity consumption needed for space heating and heating storage
        """
        
        # Heating power that could be possible to supply to the storage device to increase its State of Charge once the heating demand of the building has been satisfied
        heat_power_avail = self.dhw_heating_device.get_max_heating_power() - self.sim_results['dhw_demand'][self.time_step]
        
        # The storage device is charged (action > 0) or discharged (action < 0) taking into account the max power available and that the storage device cannot be discharged by an amount of energy greater than the energy demand of the building. 
        heating_energy_balance = self.dhw_storage.charge(max(-self.sim_results['dhw_demand'][self.time_step], min(heat_power_avail, action*self.dhw_storage.capacity)))
        
        if self.save_memory == False:
            self.dhw_heating_device_to_storage.append(max(0, heating_energy_balance))
            self.dhw_storage_to_building.append(-min(0, heating_energy_balance))
            self.dhw_heating_device_to_building.append(self.sim_results['dhw_demand'][self.time_step] + min(0, heating_energy_balance))
            self.dhw_storage_soc.append(self.dhw_storage._soc)
        
        # The energy that the energy supply device must provide is the sum of the energy balance of the storage unit (how much net energy it will lose or get) plus the energy supplied to the building. A constraint is added to guarantee it's always positive.
        heating_energy_balance = max(0, heating_energy_balance + self.sim_results['dhw_demand'][self.time_step])
        
        # Electricity consumed by the energy supply unit
        elec_demand_heating = self.dhw_heating_device.set_total_electric_consumption_heating(heat_supply = heating_energy_balance)
        
        # Electricity consumption used (if +) or saved (if -) due to the change in the state of charge of the energy storage device 
        self._electric_consumption_dhw_storage = elec_demand_heating - self.dhw_heating_device.get_electric_consumption_heating(heat_supply = self.sim_results['dhw_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_consumption_dhw.append(elec_demand_heating)
            self.electric_consumption_dhw_storage.append(self._electric_consumption_dhw_storage)
        
        self.dhw_heating_device.time_step += 1
        
        return elec_demand_heating
    
        
    def set_storage_cooling(self, action):
        """
            Args:
                action (float): Amount of cooling energy stored (added) in that time-step as a ratio of the maximum capacity of the energy storage device. 
                1 =< action < 0 : Energy Storage Unit releases energy into the building and its State of Charge decreases
                0 < action <= -1 : Energy Storage Unit receives energy from the energy supply device and its State of Charge increases
                The actions are always subject to the constraints of the power capacity of the cooling supply unit, the cooling demand of the
                building (which limits the maximum amount of cooling energy that the energy storage can provide to the building), and the state of charge of the energy storage unit itself
            Return:
                elec_demand_cooling (float): electricity consumption needed for space cooling and cooling storage
        """
    
        # Cooling power that could be possible to supply to the storage device to increase its State of Charge once the heating demand of the building has been satisfied
        cooling_power_avail = self.cooling_device.get_max_cooling_power() - self.sim_results['cooling_demand'][self.time_step]
        
        # The storage device is charged (action > 0) or discharged (action < 0) taking into account the max power available and that the storage device cannot be discharged by an amount of energy greater than the energy demand of the building.
        cooling_energy_balance = self.cooling_storage.charge(max(-self.sim_results['cooling_demand'][self.time_step], min(cooling_power_avail, action*self.cooling_storage.capacity))) 
        
        if self.save_memory == False:
            self.cooling_device_to_storage.append(max(0, cooling_energy_balance))
            self.cooling_storage_to_building.append(-min(0, cooling_energy_balance))
            self.cooling_device_to_building.append(self.sim_results['cooling_demand'][self.time_step] + min(0, cooling_energy_balance))
            self.cooling_storage_soc.append(self.cooling_storage._soc)
        
        # The energy that the energy supply device must provide is the sum of the energy balance of the storage unit (how much net energy it will lose or get) plus the energy supplied to the building. A constraint is added to guarantee it's always positive.
        cooling_energy_balance = max(0, cooling_energy_balance + self.sim_results['cooling_demand'][self.time_step])
        
        # Electricity consumed by the energy supply unit
        elec_demand_cooling = self.cooling_device.set_total_electric_consumption_cooling(cooling_supply = cooling_energy_balance)
        
        # Electricity consumption used (if +) or saved (if -) due to the change in the state of charge of the energy storage device 
        self._electric_consumption_cooling_storage = elec_demand_cooling - self.cooling_device.get_electric_consumption_cooling(cooling_supply = self.sim_results['cooling_demand'][self.time_step])
        
        if self.save_memory == False:
            self.electric_consumption_cooling.append(np.float32(elec_demand_cooling))
            self.electric_consumption_cooling_storage.append(np.float32(self._electric_consumption_cooling_storage))
            
        self.cooling_device.time_step += 1

        return elec_demand_cooling
    

    def get_non_shiftable_load(self):
        return self.sim_results['non_shiftable_load'][self.time_step]
    
    def get_solar_power(self):
        return self.sim_results['solar_gen'][self.time_step]
    
    def get_dhw_electric_demand(self):
        return self.dhw_heating_device._electrical_consumption_heating
        
    def get_cooling_electric_demand(self):
        return self.cooling_device._electrical_consumption_cooling
    
    def reset(self):
        
        self.current_net_electricity_demand = self.sim_results['non_shiftable_load'][self.time_step] - self.sim_results['solar_gen'][self.time_step]
        
        if self.dhw_storage is not None:
            self.dhw_storage.reset()
        if self.cooling_storage is not None:
            self.cooling_storage.reset()
        if self.electrical_storage is not None:
            self.electrical_storage.reset()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.reset()
            self.current_net_electricity_demand += self.dhw_heating_device.get_electric_consumption_heating(self.sim_results['dhw_demand'][self.time_step]) 
        if self.cooling_device is not None:
            self.cooling_device.reset()
            self.current_net_electricity_demand += self.cooling_device.get_electric_consumption_cooling(self.sim_results['cooling_demand'][self.time_step])
            
        self._electric_consumption_cooling_storage = 0.0
        self._electric_consumption_dhw_storage = 0.0
        
        self.cooling_demand_building = []
        self.dhw_demand_building = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
           
        self.electric_consumption_cooling = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_dhw_storage = []
        
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        
        self.cooling_device_to_building = []
        self.cooling_storage_to_building = []
        self.cooling_device_to_storage = []
        self.cooling_storage_soc = []

        self.dhw_heating_device_to_building = []
        self.dhw_storage_to_building = []
        self.dhw_heating_device_to_storage = []
        self.dhw_storage_soc = []
        
        self.electrical_storage_electric_consumption = []
        self.electrical_storage_soc = []
        
    def terminate(self):
        
        if self.dhw_storage is not None:
            self.dhw_storage.terminate()
        if self.cooling_storage is not None:
            self.cooling_storage.terminate()
        if self.electrical_storage is not None:
            self.electrical_storage.terminate()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.terminate()
        if self.cooling_device is not None:
            self.cooling_device.terminate()
            
        if self.save_memory == False:
            
            self.cooling_demand_building = np.array(self.sim_results['cooling_demand'][:self.time_step])
            self.dhw_demand_building = np.array(self.sim_results['dhw_demand'][:self.time_step])
            self.electric_consumption_appliances = np.array(self.sim_results['non_shiftable_load'][:self.time_step])
            self.electric_generation = np.array(self.sim_results['solar_gen'][:self.time_step])
            
            elec_consumption_dhw = 0
            elec_consumption_dhw_storage = 0
            if self.dhw_heating_device.time_step == self.time_step and self.dhw_heating_device is not None:
                elec_consumption_dhw = np.array(self.electric_consumption_dhw)
                elec_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
                
            elec_consumption_cooling = 0
            elec_consumption_cooling_storage = 0
            if self.cooling_device.time_step == self.time_step and self.cooling_device is not None:
                elec_consumption_cooling = np.array(self.electric_consumption_cooling)
                elec_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
                
            self.net_electric_consumption = np.array(self.electric_consumption_appliances) + elec_consumption_cooling + elec_consumption_dhw - np.array(self.electric_generation) 
            self.net_electric_consumption_no_storage = np.array(self.electric_consumption_appliances) + (elec_consumption_cooling - elec_consumption_cooling_storage) + (elec_consumption_dhw - elec_consumption_dhw_storage) - np.array(self.electric_generation)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_storage) + np.array(self.electric_generation)
                
            self.cooling_demand_building = np.array(self.cooling_demand_building)
            self.dhw_demand_building = np.array(self.dhw_demand_building)
            self.electric_consumption_appliances = np.array(self.electric_consumption_appliances)
            self.electric_generation = np.array(self.electric_generation)
               
            self.electric_consumption_cooling = np.array(self.electric_consumption_cooling)
            self.electric_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
            self.electric_consumption_dhw = np.array(self.electric_consumption_dhw)
            self.electric_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
            
            self.net_electric_consumption = np.array(self.net_electric_consumption)
            self.net_electric_consumption_no_storage = np.array(self.net_electric_consumption_no_storage)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_pv_no_storage)
            
            self.cooling_device_to_building = np.array(self.cooling_device_to_building)
            self.cooling_storage_to_building = np.array(self.cooling_storage_to_building)
            self.cooling_device_to_storage = np.array(self.cooling_device_to_storage)
            self.cooling_storage_soc = np.array(self.cooling_storage_soc)
    
            self.dhw_heating_device_to_building = np.array(self.dhw_heating_device_to_building)
            self.dhw_storage_to_building = np.array(self.dhw_storage_to_building)
            self.dhw_heating_device_to_storage = np.array(self.dhw_heating_device_to_storage)
            self.dhw_storage_soc = np.array(self.dhw_storage_soc)
            
            self.electrical_storage_electric_consumption = np.array(self.electrical_storage_electric_consumption)
            self.electrical_storage_soc = np.array(self.electrical_storage_soc)
        

class HeatPump:
    def __init__(self, nominal_power = None, eta_tech = None, t_target_heating = None, t_target_cooling = None, save_memory = True):
        """
        Args:
            nominal_power (float): Maximum amount of electric power that the heat pump can consume from the power grid (given by the nominal power of the compressor)
            eta_tech (float): Technical efficiency
            t_target_heating (float): Temperature at which the heating energy is released
            t_target_cooling (float): Temperature at which the cooling energy is released
        """
        #Parameters
        self.nominal_power = nominal_power
        self.eta_tech = eta_tech
        
        #Variables
        self.max_cooling = None
        self.max_heating = None
        self._cop_heating = None
        self._cop_cooling = None
        self.t_target_heating = t_target_heating
        self.t_target_cooling = t_target_cooling
        self.t_source_heating = None
        self.t_source_cooling = None
        self.cop_heating = []
        self.cop_cooling = []
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
        self.time_step = 0
        self.save_memory = save_memory
                   
    def get_max_cooling_power(self, max_electric_power = None):
        """
        Args:
            max_electric_power (float): Maximum amount of electric power that the heat pump can consume from the power grid
            
        Returns:
            max_cooling (float): maximum amount of cooling energy that the heatpump can provide
        """

        if max_electric_power is None:
            self.max_cooling = self.nominal_power*self.cop_cooling[self.time_step]
        else:
            self.max_cooling = min(max_electric_power, self.nominal_power)*self.cop_cooling[self.time_step]
        return self.max_cooling
    
    def get_max_heating_power(self, max_electric_power = None):
        """
        Method that calculates the heating COP and the maximum heating power available
        Args:
            max_electric_power (float): Maximum amount of electric power that the heat pump can consume from the power grid
            
        Returns:
            max_heating (float): maximum amount of heating energy that the heatpump can provide
        """
        
        if max_electric_power is None:
            self.max_heating = self.nominal_power*self.cop_cooling[self.time_step]
        else:
            self.max_heating = min(max_electric_power, self.nominal_power)*self.cop_cooling[self.time_step]
            
        return self.max_heating
    
    def set_total_electric_consumption_cooling(self, cooling_supply = 0):
        """
        Method that calculates the total electricity consumption of the heat pump given an amount of cooling energy to be supplied to both the building and the storage unit
        Args:
            cooling_supply (float): Total amount of cooling energy that the heat pump is going to supply
            
        Returns:
            _electrical_consumption_cooling (float): electricity consumption for cooling
        """
        
        self.cooling_supply.append(cooling_supply)
        self._electrical_consumption_cooling = cooling_supply/self.cop_cooling[self.time_step]
        
        if self.save_memory == False:
            self.electrical_consumption_cooling.append(np.float32(self._electrical_consumption_cooling))
            
        return self._electrical_consumption_cooling
            
    def get_electric_consumption_cooling(self, cooling_supply = 0):
        """
        Method that calculates the electricity consumption of the heat pump given an amount of cooling energy
        Args:
            cooling_supply (float): Amount of cooling energy
            
        Returns:
            _electrical_consumption_cooling (float): electricity consumption for that amount of cooling
        """
        
        _elec_consumption_cooling = cooling_supply/self.cop_cooling[self.time_step]
        return _elec_consumption_cooling
    
    def set_total_electric_consumption_heating(self, heat_supply = 0):
        """
        Method that calculates the electricity consumption of the heat pump given an amount of heating energy to be supplied
        Args:
            heat_supply (float): Amount of heating energy that the heat pump is going to supply
            
        Returns:
            _elec_consumption_heating (float): electricity consumption for heating
        """
        
        self.heat_supply.append(heat_supply)
        self._electrical_consumption_heating = heat_supply/self.cop_heating[self.time_step]
        
        if self.save_memory == False:
            self.electrical_consumption_heating.append(np.float32(self._electrical_consumption_heating))
            
        return self._electrical_consumption_heating
    
    def get_electric_consumption_heating(self, heat_supply = 0):
        """
        Method that calculates the electricity consumption of the heat pump given an amount of heating energy to be supplied
        Args:
            heat_supply (float): Amount of heating energy that the heat pump is going to supply
            
        Returns:
            _elec_consumption_heating (float): electricity consumption for heating
        """
        
        _elec_consumption_heating = heat_supply/self.cop_heating[self.time_step]
        return _elec_consumption_heating
    
    def reset(self):
        self.t_source_heating = None
        self.t_source_cooling = None
        self.max_cooling = None
        self.max_heating = None
        self._cop_heating = None
        self._cop_cooling = None
        self._electrical_consumption_cooling = 0
        self._electrical_consumption_heating = 0
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
        self.time_step = 0
        
    def terminate(self):
        if self.save_memory == False:
            self.cop_heating = self.cop_heating[:self.time_step]
            self.cop_cooling = self.cop_cooling[:self.time_step]
            self.electrical_consumption_cooling = np.array(self.electrical_consumption_cooling)
            self.electrical_consumption_heating = np.array(self.electrical_consumption_heating)
            self.heat_supply = np.array(self.heat_supply)
            self.cooling_supply = np.array(self.cooling_supply)

class ElectricHeater:
    def __init__(self, nominal_power = None, efficiency = None, save_memory = True):
        """
        Args:
            nominal_power (float): Maximum amount of electric power that the electric heater can consume from the power grid
            efficiency (float): efficiency
        """
        
        #Parameters
        self.nominal_power = nominal_power
        self.efficiency = efficiency
        
        #Variables
        self.max_heating = None
        self.electrical_consumption_heating = []
        self._electrical_consumption_heating = 0
        self.heat_supply = []
        self.time_step = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.electrical_consumption_heating = np.array(self.electrical_consumption_heating)
            self.heat_supply = np.array(self.heat_supply)
        
    def get_max_heating_power(self, max_electric_power = None, t_source_heating = None, t_target_heating = None):
        """Method that calculates the maximum heating power available
        Args:
            max_electric_power (float): Maximum amount of electric power that the electric heater can consume from the power grid
            t_source_heating (float): Not used by the electric heater
            t_target_heating (float): Not used by electric heater
            
        Returns:
            max_heating (float): maximum amount of heating energy that the electric heater can provide
        """
        
        if max_electric_power is None:
            self.max_heating = self.nominal_power*self.efficiency
        else:
            self.max_heating = self.max_electric_power*self.efficiency
        
        return self.max_heating
    
    def set_total_electric_consumption_heating(self, heat_supply = 0):
        """
        Args:
            heat_supply (float): Amount of heating energy that the electric heater is going to supply
            
        Returns:
            _electrical_consumption_heating (float): electricity consumption for heating
        """
        
        self.heat_supply.append(heat_supply)
        self._electrical_consumption_heating = heat_supply/self.efficiency
        
        if self.save_memory == False:
            self.electrical_consumption_heating.append(np.float32(self._electrical_consumption_heating))
            
        return self._electrical_consumption_heating
    
    def get_electric_consumption_heating(self, heat_supply = 0):
        """
        Args:
            heat_supply (float): Amount of heating energy that the electric heater is going to supply
            
        Returns:
            _electrical_consumption_heating (float): electricity consumption for heating
        """
        
        _electrical_consumption_heating = heat_supply/self.efficiency
        return _electrical_consumption_heating
    
    def reset(self):
        self.max_heating = None
        self.electrical_consumption_heating = []
        self.heat_supply = []
    
class EnergyStorage:
    def __init__(self, capacity = None, max_power_output = None, max_power_charging = None, efficiency = 1, loss_coef = 0, save_memory = True):
        """
        Generic energy storage class. It can be used as a chilled water storage tank or a DHW storage tank
        Args:
            capacity (float): Maximum amount of energy that the storage unit is able to store (kWh)
            max_power_output (float): Maximum amount of power that the storage unit can output (kW)
            max_power_charging (float): Maximum amount of power that the storage unit can use to charge (kW)
            efficiency (float): Efficiency factor of charging and discharging the storage unit (from 0 to 1)
            loss_coef (float): Loss coefficient used to calculate the amount of energy lost every hour (from 0 to 1)
        """
            
        self.capacity = capacity
        self.max_power_output = max_power_output
        self.max_power_charging = max_power_charging
        self.efficiency = efficiency**0.5
        self.loss_coef = loss_coef
        self.soc = []
        self._soc = 0 # State of Charge
        self.energy_balance = []
        self._energy_balance = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.energy_balance = np.array(self.energy_balance)
            self.soc =  np.array(self.soc)
        
    def charge(self, energy):
        """Method that controls both the energy CHARGE and DISCHARGE of the energy storage device
        energy < 0 -> Discharge
        energy > 0 -> Charge
        Args:
            energy (float): Amount of energy stored in that time-step (Wh)
        Return:
            energy_balance (float): 
        """
        
        #The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        soc_init = self._soc*(1-self.loss_coef)
        
        #Charging    
        if energy >= 0:
            if self.max_power_charging is not None:
                energy =  min(energy, self.max_power_charging)
            self._soc = soc_init + energy*self.efficiency
            
        #Discharging
        else:
            if self.max_power_output is not None:
                energy = max(-max_power_output, energy)
            self._soc = max(0, soc_init + energy/self.efficiency)  
            
        if self.capacity is not None:
            self._soc = min(self._soc, self.capacity)
          
        # Calculating the energy balance with its external environmrnt (amount of energy taken from or relseased to the environment)
        
        #Charging    
        if energy >= 0:
            self._energy_balance = (self._soc - soc_init)/self.efficiency
            
        #Discharging
        else:
            self._energy_balance = (self._soc - soc_init)*self.efficiency
        
        if self.save_memory == False:
            self.energy_balance.append(np.float32(self._energy_balance))
            self.soc.append(np.float32(self._soc))
            
        return self._energy_balance
    
    def reset(self):
        self.soc = []
        self._soc = 0 #State of charge
        self.energy_balance = [] #Positive for energy entering the storage
        self._energy_balance = 0
        self.time_step = 0
        

class Battery:
    def __init__(self, capacity, nominal_power = None, capacity_loss_coef = None, power_efficiency_curve = None, capacity_power_curve = None, efficiency = None, loss_coef = 0, save_memory = True):
        """
        Generic energy storage class. It can be used as a chilled water storage tank or a DHW storage tank
        Args:
            capacity (float): Maximum amount of energy that the storage unit is able to store (kWh)
            max_power_charging (float): Maximum amount of power that the storage unit can use to charge (kW)
            efficiency (float): Efficiency factor of charging and discharging the storage unit (from 0 to 1)
            loss_coef (float): Loss coefficient used to calculate the amount of energy lost every hour (from 0 to 1)
            power_efficiency_curve (float): Charging/Discharging efficiency as a function of the power released or consumed
            capacity_power_curve (float): Max. power of the battery as a function of its current state of charge
            capacity_loss_coef (float): Battery degradation. Storage capacity lost in each charge and discharge cycle (as a fraction of the total capacity)
        """
            
        self.capacity = capacity
        self.c0 = capacity
        self.nominal_power = nominal_power
        self.capacity_loss_coef = capacity_loss_coef
        
        if power_efficiency_curve is not None:
            self.power_efficiency_curve = np.array(power_efficiency_curve).T
        else:
            self.power_efficiency_curve = power_efficiency_curve
            
        if capacity_power_curve is not None:
            self.capacity_power_curve = np.array(capacity_power_curve).T
        else:
            self.capacity_power_curve = capacity_power_curve
            
        self.efficiency = efficiency**0.5
        self.loss_coef = loss_coef
        self.max_power = None
        self._eff = []
        self._energy = []
        self._max_power = []
        self.soc = []
        self._soc = 0 # State of Charge
        self.energy_balance = []
        self._energy_balance = 0
        self.save_memory = save_memory
        
    def terminate(self):
        if self.save_memory == False:
            self.energy_balance = np.array(self.energy_balance)
            self.soc =  np.array(self.soc)
        
    def charge(self, energy):
        """Method that controls both the energy CHARGE and DISCHARGE of the energy storage device
        energy < 0 -> Discharge
        energy > 0 -> Charge
        Args:
            energy (float): Amount of energy stored in that time-step (Wh)
        Return:
            energy_balance (float): 
        """
        
        #The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        soc_init = self._soc*(1-self.loss_coef)
        if self.capacity_power_curve is not None:
            soc_normalized = soc_init/self.capacity
            # Calculating the maximum power rate at which the battery can be charged or discharged
            idx = max(0, np.argmax(soc_normalized <= self.capacity_power_curve[0]) - 1)
            
            self.max_power = self.nominal_power*(self.capacity_power_curve[1][idx] + (self.capacity_power_curve[1][idx+1] - self.capacity_power_curve[1][idx]) * (soc_normalized - self.capacity_power_curve[0][idx])/(self.capacity_power_curve[0][idx+1] - self.capacity_power_curve[0][idx]))
        
        else:
            self.max_power = self.nominal_power
          
        #Charging    
        if energy >= 0:
            if self.nominal_power is not None:
                
                energy =  min(energy, self.max_power)
                if self.power_efficiency_curve is not None:
                    # Calculating the maximum power rate at which the battery can be charged or discharged
                    energy_normalized = np.abs(energy)/self.nominal_power
                    idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
                    self.efficiency = self.power_efficiency_curve[1][idx] + (energy_normalized - self.power_efficiency_curve[0][idx])*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx])/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
                    self.efficiency = self.efficiency**0.5
                 
            self._soc = soc_init + energy*self.efficiency
            
        #Discharging
        else:
            if self.nominal_power is not None:
                energy = max(-self.max_power, energy)
                
            if self.power_efficiency_curve is not None:
                
                # Calculating the maximum power rate at which the battery can be charged or discharged
                energy_normalized = np.abs(energy)/self.nominal_power
                idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
                self.efficiency = self.power_efficiency_curve[1][idx] + (energy_normalized - self.power_efficiency_curve[0][idx])*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx])/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
                self.efficiency = self.efficiency**0.5
                    
            self._soc = max(0, soc_init + energy/self.efficiency)
            
        if self.capacity is not None:
            self._soc = min(self._soc, self.capacity)
          
        # Calculating the energy balance with its external environment (amount of energy taken from or relseased to the environment)
        
        #Charging    
        if energy >= 0:
            self._energy_balance = (self._soc - soc_init)/self.efficiency
            
        #Discharging
        else:
            self._energy_balance = (self._soc - soc_init)*self.efficiency
            
        # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge       
        self.capacity -= self.capacity_loss_coef*self.c0*np.abs(self._energy_balance)/(2*self.capacity)
        
        if self.save_memory == False:
            self.energy_balance.append(np.float32(self._energy_balance))
            self.soc.append(np.float32(self._soc))
            
        return self._energy_balance
    
    def reset(self):
        self.soc = []
        self._soc = 0 #State of charge
        self.energy_balance = [] #Positive for energy entering the storage
        self._energy_balance = 0
        self.time_step = 0