from typing import Iterable, List, Union
import numpy as np
from citylearn.base import Environment
np.seterr(divide = 'ignore', invalid = 'ignore')

class Device(Environment):
    def __init__(self, efficiency: float = 1.0):
        super().__init__()
        self.efficiency = efficiency

    @property
    def efficiency(self) -> float:
        return self.__efficiency

    @efficiency.setter
    def efficiency(self, efficiency: float):
        assert efficiency > 0, 'efficiency must be >= 0.'
        self.__efficiency = efficiency

    def autosize(self):
        raise NotImplementedError

class ElectricDevice(Device):
    def __init__(self, nominal_power: float = None, **kwargs):
        super().__init__(**kwargs)
        self.nominal_power = nominal_power

    @property
    def nominal_power(self) -> float:
        return self.__nominal_power

    @property
    def electricity_consumption(self) -> List[float]:
        return self.__electricity_consumption

    @property
    def available_nominal_power(self) -> float:
        return None if self.nominal_power is None else self.nominal_power - self.electricity_consumption[self.time_step]

    @nominal_power.setter
    def nominal_power(self, nominal_power: float):
        assert nominal_power is None or nominal_power >= 0, 'nominal_power must be >= 0.'
        self.__nominal_power = nominal_power if nominal_power > 0 else 0.00001

    def update_electricity_consumption(self, electricity_consumption: float):
        assert electricity_consumption >= 0, 'electricity_consumption must be >= 0.'
        self.__electricity_consumption[self.time_step] += electricity_consumption

    def next_time_step(self):
        super().next_time_step()
        self.__electricity_consumption.append(0.0)

    def reset(self):
        super().reset()
        self.__electricity_consumption = [0.0]

class HeatPump(ElectricDevice):
    def __init__(self, nominal_power: float, efficiency: float = 0.2, t_target_heating: float = 45.0, t_target_cooling: float = 8.0, **kwargs):
        """
        Args:
            nominal_power (float): Maximum amount of electric power that the heat pump can consume from the power grid (given by the nominal power of the compressor)
            efficiency (float): Technical efficiency
            t_target_heating (float): Temperature at which the heating energy is released
            t_target_cooling (float): Temperature at which the cooling energy is released
        """
        super().__init__(nominal_power = nominal_power, efficiency = efficiency, **kwargs)
        self.t_target_heating = t_target_heating
        self.t_target_cooling = t_target_cooling

    @property
    def t_target_heating(self) -> float:
        return self.__t_target_heating

    @property
    def t_target_cooling(self) -> float:
        return self.__t_target_cooling

    @t_target_heating.setter
    def t_target_heating(self, t_target_heating: float):
        self.__t_target_heating = t_target_heating

    @t_target_cooling.setter
    def t_target_cooling(self, t_target_cooling: float):
        self.__t_target_cooling = t_target_cooling

    def get_cop(self, outdoor_drybulb_temperature: Union[float, Iterable[float]],  heating: bool) -> Union[float, Iterable[float]]:
        c_to_k = lambda x: x + 273.15
        outdoor_drybulb_temperature = np.array(outdoor_drybulb_temperature)

        if heating:
            cop = self.efficiency*c_to_k(self.t_target_heating)/(self.t_target_heating - outdoor_drybulb_temperature)
        else:
            cop = self.efficiency*c_to_k(self.t_target_cooling)/(outdoor_drybulb_temperature - self.t_target_cooling)
        
        cop = np.array(cop)
        cop[cop < 0] = 20
        cop[cop > 20] = 20
        return cop

    def get_max_output_power(self, outdoor_drybulb_temperature: Union[float, Iterable[float]], heating: bool, max_electric_power: Union[float, Iterable[float]] = None) -> Union[float, Iterable[float]]:
        """
        Method that calculates the heating COP and the maximum heating power available at current time step.
        Args:
            max_electric_power (float): Maximum amount of electric power that the heat pump can consume from the power grid
            
        Returns:
            max_heating (float): maximum amount of heating energy that the heatpump can provide
        """

        cop = self.get_cop(outdoor_drybulb_temperature, heating)

        if max_electric_power is None: 
            return self.available_nominal_power*cop  
        else:
            return np.minimum(max_electric_power, self.available_nominal_power)*cop 

    def get_input_power(self, output_power: Union[float, Iterable[float]], outdoor_drybulb_temperature: Union[float, Iterable[float]], heating: bool) -> Union[float, Iterable[float]]:
        return output_power/self.get_cop(outdoor_drybulb_temperature, heating)

    def autosize(self, outdoor_drybulb_temperature: Iterable[float], cooling_demand: Iterable[float] = None, heating_demand: Iterable[float] = None):
        if cooling_demand is not None:
            cooling_nominal_power = np.array(cooling_demand)/self.get_cop(outdoor_drybulb_temperature, False)
        else:
            cooling_nominal_power = 0
        
        if heating_demand is not None:
            heating_nominal_power = np.array(heating_demand)/self.get_cop(outdoor_drybulb_temperature, True)
        else:
            heating_nominal_power = 0

        self.nominal_power = max(cooling_nominal_power + heating_nominal_power)

class ElectricHeater(ElectricDevice):
    def __init__(self, nominal_power: float, efficiency: float = 0.9, **kwargs):
        """
        Args:
            nominal_power (float): Maximum amount of electric power that the electric heater can consume from the power grid
            efficiency (float): efficiency
        """

        super().__init__(nominal_power = nominal_power, efficiency = efficiency, **kwargs)

    def get_max_output_power(self, max_electric_power: Union[float, Iterable[float]] = None) -> Union[float, Iterable[float]]:
        """Method that calculates the maximum heating power available  at current time step.
        Args:
            max_electric_power (float): Maximum amount of electric power that the electric heater can consume from the power grid
        Returns:
            max_heating (float): maximum amount of heating energy that the electric heater can provide
        """
        if max_electric_power is None:
            return self.available_nominal_power*self.efficiency
        else:
            return np.minimum(max_electric_power, self.available_nominal_power)*self.efficiency

    def get_input_power(self, output_power: Union[float, Iterable[float]]) -> Union[float, Iterable[float]]:
        return np.array(output_power)/self.efficiency

    def autosize(self, demand: Iterable[float]):
        self.nominal_power = max(np.array(demand)/self.efficiency)

class PV(Device):
    def __init__(self, capacity: float, **kwargs):
        super().__init__(**kwargs)
        self.capacity = capacity

    @property
    def capacity(self) -> float:
        return self.__capacity

    @capacity.setter
    def capacity(self, capacity: float):
        assert capacity >= 0, 'capacity must be >= 0.'
        self.__capacity = capacity

    def get_generation(self, inverter_ac_power_per_w: Union[float, Iterable[float]]) -> Union[float, Iterable[float]]:
        return self.capacity*np.array(inverter_ac_power_per_w)/1000

    def autosize(self, demand: Iterable[float]):
        self.capacity = max(np.array(demand)/self.efficiency)

class StorageDevice(Device):
    def __init__(self, capacity: float, efficiency: float = 0.9, loss_coef: float = 0.006, initial_soc: float = 0, efficiency_scaling: float = 0.5, **kwargs):
        self.efficiency_scaling = efficiency_scaling
        self.capacity = capacity
        self.loss_coef = loss_coef
        self.initial_soc = initial_soc
        super().__init__(efficiency = efficiency**self.efficiency_scaling, **kwargs)

    @property
    def capacity(self) -> float:
        return self.__capacity

    @property
    def loss_coef(self) -> float:
        return self.__loss_coef

    @property
    def initial_soc(self) -> float:
        return self.__initial_soc

    @property
    def soc(self) -> List[float]:
        return self.__soc[1:]

    @property
    def soc_init(self) -> float:
        return self.__soc[-1]*(1 - self.loss_coef)

    @property
    def efficiency_scaling(self) -> float:
        return self.__efficiency_scaling

    @property
    def energy_balance(self) -> List[float]:
        # actual energy charged/discharged irrespective of what is determined in the step function after 
        # taking into account storage design limits e.g. maximum power input/output, capacity
        energy_balance = np.array(self.soc, dtype = float) - np.array(self.__soc[0:-1], dtype = float)*(1 - self.loss_coef)
        energy_balance[energy_balance >= 0.0] /= self.efficiency
        energy_balance[energy_balance < 0.0] *= self.efficiency
        return energy_balance.tolist()

    @capacity.setter
    def capacity(self, capacity: float):
        assert capacity >= 0, 'capacity must be >= 0.'
        self.__capacity = capacity if capacity > 0 else 0.00001

    @loss_coef.setter
    def loss_coef(self, loss_coef: float):
        assert 0 <= loss_coef <= 1, 'initial_soc must be >= 0 and <= 1.'
        self.__loss_coef = loss_coef

    @initial_soc.setter
    def initial_soc(self, initial_soc: float):
        assert 0 <= initial_soc <= self.capacity, 'initial_soc must be >= 0 and <= capacity.'
        self.__initial_soc = initial_soc

    @efficiency_scaling.setter
    def efficiency_scaling(self, efficiency_scaling: float):
        self.__efficiency_scaling = efficiency_scaling

    def charge(self, energy: float):
        """Method that controls both the energy CHARGE and DISCHARGE of the energy storage device
        energy < 0 -> Discharge
        energy > 0 -> Charge
        Args:
            energy (float): Amount of energy stored in that time-step (Wh)
        """
        
        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        soc = min(self.soc_init + energy*self.efficiency, self.capacity) if energy >= 0 else max(0, self.soc_init + energy/self.efficiency)
        self.__soc.append(soc)

    def autosize(self, demand: Iterable[float], multiplier: int = 1):
        self.capacity = max(demand)*multiplier

    def reset(self):
        super().reset()
        self.__soc = [self.initial_soc]

class StorageTank(StorageDevice):
    def __init__(self, capacity: float, max_output_power: float = None, max_input_power: float = None, **kwargs):
        """
        Generic energy storage class. It can be used as a chilled water storage tank or a DHW storage tank
        Args:
            capacity (float): Maximum amount of energy that the storage unit is able to store (kWh)
            max_output_power (float): Maximum amount of power that the storage unit can output (kW)
            max_input_power (float): Maximum amount of power that the storage unit can use to charge (kW)
            efficiency (float): Efficiency factor of charging and discharging the storage unit (from 0 to 1)
            loss_coef (float): Loss coefficient used to calculate the amount of energy lost every hour (from 0 to 1)
        """
        super().__init__(capacity = capacity, **kwargs)
        self.max_output_power = max_output_power
        self.max_input_power = max_input_power

    @property
    def max_output_power(self) -> float:
        return self.__max_output_power

    @property
    def max_input_power(self) -> float:
        return self.__max_input_power

    @max_output_power.setter
    def max_output_power(self, max_output_power: float):
        
        self.__max_output_power = max_output_power

    @max_input_power.setter
    def max_input_power(self, max_input_power: float):
        self.__max_input_power = max_input_power

    def charge(self, energy: float):
        """Method that controls both the energy CHARGE and DISCHARGE of the energy storage device
        energy < 0 -> Discharge
        energy > 0 -> Charge
        Args:
            energy (float): Amount of energy stored in that time-step (Wh)
        """

        if energy >= 0:    
            energy = energy if self.max_input_power is None else min(energy, self.max_input_power)
        else:
            energy = energy if self.max_output_power is None else max(-self.max_output_power, energy)
        
        super().charge(energy)

class Battery(ElectricDevice, StorageDevice):
    def __init__(self, capacity: float, nominal_power: float, efficiency: float = 0.9, capacity_loss_coef: float = 1e-5, power_efficiency_curve: List[List[float]] = [[0,0.83],[0.3,0.83],[0.7,0.9],[0.8,0.9],[1,0.85]], capacity_power_curve: List[List[float]] = [[0.0 , 1],[0.8, 1],[1.0 ,0.2]], **kwargs):
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

        self.__efficiency_history = []
        self.__capacity_history = []
        super().__init__(capacity = capacity, nominal_power = nominal_power, efficiency = efficiency, **kwargs)
        self.capacity_loss_coef = capacity_loss_coef
        self.power_efficiency_curve = power_efficiency_curve
        self.capacity_power_curve = capacity_power_curve
        
    @StorageDevice.capacity.getter
    def capacity(self) -> float:
        return self.capacity_history[-1]

    @StorageDevice.efficiency.getter
    def efficiency(self) -> float:
        return self.efficiency_history[-1]

    @ElectricDevice.electricity_consumption.getter
    def electricity_consumption(self) -> List[float]:
        return self.energy_balance

    @property
    def capacity_loss_coef(self) -> float:
        return self.__capacity_loss_coef

    @property
    def power_efficiency_curve(self) -> List[List[float]]:
        return self.__power_efficiency_curve

    @property
    def capacity_power_curve(self) -> List[List[float]]:
        return self.__capacity_power_curve

    @property
    def efficiency_history(self) -> List[float]:
        return self.__efficiency_history

    @property
    def capacity_history(self) -> List[float]:
        return self.__capacity_history

    @property
    def energy_balance(self) -> List[float]:
        energy_balance = np.array(super().energy_balance, dtype = float)
        efficiency_history = np.array(self.efficiency_history[1:], dtype = float)
        energy_balance[energy_balance >= 0.0] *= self.efficiency/efficiency_history[energy_balance >= 0.0]
        energy_balance[energy_balance < 0.0] /= self.efficiency*efficiency_history[energy_balance < 0.0]
        return energy_balance.tolist()

    @StorageDevice.capacity.setter
    def capacity(self, capacity: float):
        StorageDevice.capacity.fset(self, capacity)
        self.__capacity_history.append(capacity)

    @StorageDevice.efficiency.setter
    def efficiency(self, efficiency: float):
        StorageDevice.efficiency.fset(self, efficiency)
        self.__efficiency_history.append(efficiency)

    @capacity_loss_coef.setter
    def capacity_loss_coef(self, capacity_loss_coef: float):
        self.__capacity_loss_coef = capacity_loss_coef

    @power_efficiency_curve.setter
    def power_efficiency_curve(self, power_efficiency_curve: List[List[float]]):
        self.__power_efficiency_curve = power_efficiency_curve if power_efficiency_curve is None else np.array(power_efficiency_curve).T

    @capacity_power_curve.setter
    def capacity_power_curve(self, capacity_power_curve: List[List[float]]):
        self.__capacity_power_curve = capacity_power_curve if capacity_power_curve is None else np.array(capacity_power_curve).T

    def charge(self, energy: float):
        """Method that controls both the energy CHARGE and DISCHARGE of the energy storage device
        energy < 0 -> Discharge
        energy > 0 -> Charge
        Args:
            energy (float): Amount of energy stored in that time-step (Wh)
        Return:
            energy_balance (float): 
        """

        energy = min(energy, self.get_max_input_power()) if energy >= 0 else max(-self.get_max_output_power(), energy)
        self.__efficiency_history.append(self.efficiency)
        super().charge(energy)
        self.__capacity_history.append(self.capacity - self.degrade())

    def get_max_output_power(self) -> float:
        return self.get_max_input_power()

    def get_max_input_power(self) -> float:
        #The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        if self.capacity_power_curve is not None:
            soc_normalized = self.soc_init/self.capacity
            # Calculating the maximum power rate at which the battery can be charged or discharged
            idx = max(0, np.argmax(soc_normalized <= self.capacity_power_curve[0]) - 1)
            max_output_power = self.nominal_power*(
                self.capacity_power_curve[1][idx] 
                + (self.capacity_power_curve[1][idx+1] - self.capacity_power_curve[1][idx])*(soc_normalized - self.capacity_power_curve[0][idx])
                /(self.capacity_power_curve[0][idx+1] - self.capacity_power_curve[0][idx])
            )
        else:
            max_output_power = self.nominal_power
        
        return max_output_power

    def get_current_efficiency(self, energy: float) -> float:
        if self.power_efficiency_curve is not None:
            # Calculating the maximum power rate at which the battery can be charged or discharged
            energy_normalized = np.abs(energy)/self.nominal_power
            idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
            efficiency = self.power_efficiency_curve[1][idx]\
                + (energy_normalized - self.power_efficiency_curve[0][idx]
                )*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx]
                )/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
            efficiency = efficiency**self.efficiency_scaling
        else:
            efficiency = self.efficiency

        return efficiency

    def degrade(self) -> float:
        # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge
        capacity_degrade = self.capacity_loss_coef*self.capacity_history[0]*np.abs(self.energy_balance[-1])/(2*self.capacity)
        return capacity_degrade

    def reset(self):
        super().reset()
        self.__efficiency_history = self.efficiency_history[0:1]
        self.__capacity_history = self.capacity_history[0:1]