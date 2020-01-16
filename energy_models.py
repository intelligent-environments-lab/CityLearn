from gym import spaces
import numpy as np

class Building:  
    def __init__(self, buildingId, dhw_storage = None, cooling_storage = None, electrical_storage = None, dhw_heating_device = None, cooling_device = None):
        """
        Args:
            buildingId (int)
            dhw_storage (EnergyStorage)
            cooling_storage (EnergyStorage)
            electrical_storage (EnergyStorage)
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
        self.electricity_consumption_cooling_storage = 0.0
        self.electricity_consumption_dhw_storage = 0.0
        self.electricity_consumption_heating = []
        self.electricity_consumption_cooling = []
        
    def set_state_space(self, high_state, low_state):
        # Setting the state space and the lower and upper bounds of each state-variable
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
    
    def set_action_space(self, max_action, min_action):
        # Setting the action space and the lower and upper bounds of each action-variable
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

    def set_storage_heating(self, action):
        """
        Args:
            action (float): Amount of heating energy stored (added) in that time-step as a ratio of the maximum capacity of the energy storage device. 
            1 =< action < 0 : Energy Storage Unit releases energy into the building and its State of Charge decreases
            0 < action <= -1 : Energy Storage Unit receives energy from the energy supply device and its State of Charge increases
            The actions are always subject to the constraints of the power capacity of the heating supply unit, the DHW demand of the
            building (which limits the maximum amount of DHW that the energy storage can provide to the building), and the state of charge of the
            energy storage unit itself
        Return:
            elec_demand_heating (float): electricity consumption needed for space heating and heating storage
        """
        
        # Heating power that could be possible to supply to the storage device to increase its State of Charge once the heating demand of the building has been satisfied
        heat_power_avail = self.dhw_heating_device.get_max_heating_power(t_source_heating = self.sim_results['t_out'][self.time_step]) - self.sim_results['dhw_demand'][self.time_step]
        
        # The storage device is charged (action > 0) or discharged (action < 0) taking into account the max power available and that the storage device cannot be discharged by an amount of energy greater than the energy demand of the building. 
        heating_energy_balance = self.dhw_storage.charge(max(-self.sim_results['dhw_demand'][self.time_step], min(heat_power_avail, action*self.dhw_storage.capacity))) 
        
        # The energy that the energy supply device must provide is the sum of the energy balance of the storage unit (how much net energy it will lose or get) plus the energy supplied to the building. A constraint is added to guarantee it's always positive.
        heating_energy_balance = max(0, heating_energy_balance + self.sim_results['dhw_demand'][self.time_step])
        
        # Electricity consumed by the energy supply unit
        elec_demand_heating = self.dhw_heating_device.set_total_electric_consumption_heating(heat_supply = heating_energy_balance)
        
        # Electricity consumption used (if +) or saved (if -) due to the change in the state of charge of the energy storage device 
        self.electricity_consumption_dhw_storage = elec_demand_heating - self.dhw_heating_device.get_electric_consumption_heating(heat_supply = self.sim_results['dhw_demand'][self.time_step])
        
        self.electricity_consumption_heating.append(elec_demand_heating)
        
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
        cooling_power_avail = self.cooling_device.get_max_cooling_power(t_source_cooling = self.sim_results['t_out'][self.time_step]) - self.sim_results['cooling_demand'][self.time_step]
        
        # The storage device is charged (action > 0) or discharged (action < 0) taking into account the max power available and that the storage device cannot be discharged by an amount of energy greater than the energy demand of the building. 
        cooling_energy_balance = self.cooling_storage.charge(max(-self.sim_results['cooling_demand'][self.time_step], min(cooling_power_avail, action*self.cooling_storage.capacity))) 
        
        # The energy that the energy supply device must provide is the sum of the energy balance of the storage unit (how much net energy it will lose or get) plus the energy supplied to the building. A constraint is added to guarantee it's always positive.
        cooling_energy_balance = max(0, cooling_energy_balance + self.sim_results['cooling_demand'][self.time_step])
        
        # Electricity consumed by the energy supply unit
        elec_demand_cooling = self.cooling_device.set_total_electric_consumption_cooling(cooling_supply = cooling_energy_balance)
        
        # Electricity consumption used (if +) or saved (if -) due to the change in the state of charge of the energy storage device 
        self.electricity_consumption_cooling_storage = elec_demand_cooling - self.cooling_device.get_electric_consumption_cooling(cooling_supply = self.sim_results['cooling_demand'][self.time_step])
        
        self.electricity_consumption_cooling.append(elec_demand_cooling)
        
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
        self.electricity_consumption_heating = []
        self.electricity_consumption_cooling = []             

class HeatPump:
    def __init__(self, nominal_power = None, eta_tech = None, t_target_heating = None, t_target_cooling = None):
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
            t_source_cooling (float): Temperature of the source from where the cooling energy is taken
            t_target_cooling (float): Temperature of the sink where the cooling energy will be released
            
        Returns:
            max_cooling (float): maximum amount of cooling energy that the heatpump can provide
        """
        
        if t_target_cooling is not None:
            self.t_target_cooling = t_target_cooling
            
        if t_source_cooling is not None:
            self.t_source_cooling = t_source_cooling

        # Calculating the COP (coefficient of performance) and clipping it between 1.0 and 20.0
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
        """
        Method that calculates the heating COP and the maximum heating power available
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
        
        # Calculating the COP (coefficient of performance) and clipping it between 1.0 and 20.0
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
    
    def set_total_electric_consumption_cooling(self, cooling_supply = 0):
        """
        Method that calculates the total electricity consumption of the heat pump given an amount of cooling energy to be supplied to both the building and the storage unit
        Args:
            cooling_supply (float): Total amount of cooling energy that the heat pump is going to supply
            
        Returns:
            _elec_consumption_cooling (float): electricity consumption for cooling
        """
        
        self.cooling_supply.append(cooling_supply)
        self._elec_consumption_cooling = cooling_supply/self.cop_cooling
        self.electrical_consumption_cooling.append(self._elec_consumption_cooling)
        return self._elec_consumption_cooling
            
    def get_electric_consumption_cooling(self, cooling_supply = 0):
        """
        Method that calculates the electricity consumption of the heat pump given an amount of cooling energy
        Args:
            cooling_supply (float): Amount of cooling energy
            
        Returns:
            _elec_consumption_cooling (float): electricity consumption for that amount of cooling
        """
        
        _elec_consumption_cooling = cooling_supply/self.cop_cooling
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
        self._elec_consumption_heating = heat_supply/self.cop_heating
        self.electrical_consumption_heating.append(self._elec_consumption_heating)
        return self._elec_consumption_heating
    
    def get_electric_consumption_heating(self, heat_supply = 0):
        """
        Method that calculates the electricity consumption of the heat pump given an amount of heating energy to be supplied
        Args:
            heat_supply (float): Amount of heating energy that the heat pump is going to supply
            
        Returns:
            _elec_consumption_heating (float): electricity consumption for heating
        """
        
        _elec_consumption_heating = heat_supply/self.cop_heating
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
        self._electrical_consumption_cooling = 0
        self._electrical_consumption_heating = 0
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []

class ElectricHeater:
    def __init__(self, nominal_power = None, efficiency = None):
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
        self.electrical_consumption_heating.append(self._electrical_consumption_heating)
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
    def __init__(self, capacity = None, max_power_output = None, max_power_charging = None, efficiency = 1, loss_coeff = 0):
        """
        Generic energy storage class. It can be used as a chilled water storage tank or a DHW storage tank
        Args:
            capacity (float): Maximum amount of energy that the storage unit is able to store (kWh)
            max_power_output (float): Maximum amount of power that the storage unit can output (kW)
            max_power_charging (float): Maximum amount of power that the storage unit can use to charge (kW)
            efficiency (float): Efficiency factor of charging and discharging the storage unit (from 0 to 1)
            loss_coeff (float): Loss coefficient used to calculate the amount of energy lost every hour (from 0 to 1)
        """
            
        self.capacity = capacity
        self.max_power_output = max_power_output
        self.max_power_charging = max_power_charging
        self.efficiency = efficiency
        self.loss_coeff = loss_coeff
        self.soc_list = []
        self.soc = 0 # State of Charge
        self.energy_balance_list = []
        self.energy_balance = 0
        
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
        soc_init = self.soc*(1-self.loss_coeff)
        
        #Charging    
        if energy >= 0:
            if self.max_power_charging is not None:
                energy =  min(energy, self.max_power_charging)
            self.soc = max(0, soc_init + energy*self.efficiency) 
            
        #Discharging
        else:
            if self.max_power_output is not None:
                energy = max(-max_power_output, energy/self.efficiency)
                self.soc = max(0, soc_init + energy)  
            else:
                self.soc = max(0, soc_init + energy/self.efficiency)  
            
        if self.capacity is not None:
            self.soc = min(self.soc, self.capacity)
          
        #Calculating the energy balance with the electrical grid (amount of energy taken from or relseased to the power grid)
        
        #Charging    
        if energy >= 0:
            self.energy_balance = (self.soc - soc_init)/self.efficiency
            
        #Discharging
        else:
            self.energy_balance = (self.soc - soc_init)*self.efficiency
        
        self.energy_balance_list.append(self.energy_balance)
        self.soc_list.append(self.soc)
        return self.energy_balance
    
    def reset(self):
        self.soc_list = []
        self.soc = 0 #State of charge
        self.energy_balance_list = [] #Positive for energy entering the storage
        self.energy_balance = 0