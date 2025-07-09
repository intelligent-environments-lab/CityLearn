import ast
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union
import numpy as np
import pandas as pd
from PySAM import Pvwattsv8
from citylearn.base import Environment, EpisodeTracker
from citylearn.data import DataSet, ZERO_DIVISION_PLACEHOLDER, EnergySimulation, WashingMachineSimulation
np.seterr(divide='ignore', invalid='ignore')

LOGGER = logging.getLogger()

class Device(Environment):
    r"""Base device class.

    Parameters
    ----------
    efficiency : Union[float, Tuple[float, float]], default: (0.8, 1.0)
        Technical efficiency. Must be set to > 0.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, efficiency: Union[float, Tuple[float, float]] = None, **kwargs):
        kwargs.pop("dynamics",None)
        kwargs.pop("occupant",None)
        super().__init__(**kwargs)
        self.efficiency = efficiency
        self._autosize_config = None
        self.time_step_ratio = self.time_step_ratio if self.time_step_ratio is not None else 1
       
    @property
    def efficiency(self) -> float:
        """Technical efficiency."""

        return self.__efficiency

    @property
    def autosize_config(self) -> Mapping[str, Union[str, float]]:
        """Reference for configuration parameters used during autosizing."""

        return self._autosize_config

    @efficiency.setter
    def efficiency(self, efficiency: Union[float, Tuple[float, float]]):
        efficiency = self._get_property_value(efficiency, (0.8, 1.0))
        assert efficiency > 0, 'efficiency must be > 0.'
        self.__efficiency = efficiency

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'efficiency': self.efficiency,
            'autosize_config': self.autosize_config,
        }
    
    def _get_property_value(self, value: Union[float, None, Tuple[float, float]], default_value: Union[float, Tuple[float, float]]):
        """Returns `value` if it is a float or a number in the uniform distribution whose limits are defined by `value`. If `value`
        is `None`, the defalut value is used. Ideal and primarily used for stochastically setting device parameters."""

        if value is None or math.isnan(value):
            if isinstance(default_value, tuple):
                value = self.numpy_random_state.uniform(*default_value)

            else:
                value = default_value

        else:
            if isinstance(value, tuple):
                value = self.numpy_random_state.uniform(*value)
            
            else:
                pass

        return value

class ElectricDevice(Device):
    r"""Base electric device class.

    Parameters
    ----------
    nominal_power : float, default: 0.0
        Electric device nominal power >= 0.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, nominal_power: float = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.nominal_power = nominal_power

    @property
    def nominal_power(self) -> float:
        r"""Nominal power."""

        return self.__nominal_power
    
    @nominal_power.setter
    def nominal_power(self, nominal_power: float):
        nominal_power = 0.0 if nominal_power is None else nominal_power
        assert nominal_power >= 0, 'nominal_power must be >= 0.'
        self.__nominal_power = nominal_power

    @property
    def electricity_consumption(self) -> np.ndarray:
        r"""Electricity consumption time series [kWh]."""
        return self.__electricity_consumption * self.time_step_ratio

    @property
    def available_nominal_power(self) -> float:
        r"""Difference between `nominal_power` and `electricity_consumption` at current `time_step`."""

        return None if self.nominal_power is None else self.nominal_power - self.electricity_consumption[self.time_step]


    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'nominal_power': self.nominal_power,
        }

    def update_electricity_consumption(self, electricity_consumption: float, enforce_polarity: bool = None):
        r"""Updates `electricity_consumption` at current `time_step`.
        
        Parameters
        ----------
        electricity_consumption: float
            Value to add to current `time_step` `electricity_consumption`. Must be >= 0.
        enforce_polarity: bool, default: True
            Whether to allow only positive `electricity_consumption` values. Some electric
            devices like :py:class:`citylearn.energy_model.Battery` may be bi-directional and
            allow electricity discharge thus, cause negative electricity consumption.
        """

        enforce_polarity = True if enforce_polarity is None else enforce_polarity
        assert not enforce_polarity or electricity_consumption >= 0.0,\
            f'electricity_consumption must be >= 0 but value: {electricity_consumption} was provided.'
        self.__electricity_consumption[self.time_step] += electricity_consumption

    def reset(self):
        r"""Reset `ElectricDevice` to initial state and set `electricity_consumption` at `time_step` 0 to = 0.0."""

        super().reset()
        self.__electricity_consumption = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')

class HeatPump(ElectricDevice):
    r"""Base heat pump class.

    Parameters
    ----------
    nominal_power: float, default: 0.0
        Maximum amount of electric power that the heat pump can consume from the power grid (given by the nominal power of the compressor).
    efficiency : Union[float, Tuple[float, float]], default: (0.2, 0.3)
        Technical efficiency.
    target_heating_temperature : Union[float, Tuple[float, float]], default: (45.0, 50.0)
        Target heating supply dry bulb temperature in [C].
    target_cooling_temperature : Union[float, Tuple[float, float]], default: (7.0, 10.0)
        Target cooling supply dry bulb temperature in [C].

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, nominal_power: float = None, efficiency: float = None, target_heating_temperature: Union[float, Tuple[float, float]] = None, target_cooling_temperature: Union[float, Tuple[float, float]] = None, **kwargs: Any):
        super().__init__(nominal_power = nominal_power, efficiency = efficiency, **kwargs)
        self.target_heating_temperature = target_heating_temperature
        self.target_cooling_temperature = target_cooling_temperature

    @property
    def target_heating_temperature(self) -> float:
        r"""Target heating supply dry bulb temperature in [C]."""

        return self.__target_heating_temperature

    @property
    def target_cooling_temperature(self) -> float:
        r"""Target cooling supply dry bulb temperature in [C]."""

        return self.__target_cooling_temperature

    @target_heating_temperature.setter
    def target_heating_temperature(self, target_heating_temperature: Union[float, Tuple[float, float]]):
        target_heating_temperature = self._get_property_value(target_heating_temperature, (45.0, 50.0))
        self.__target_heating_temperature = target_heating_temperature

    @target_cooling_temperature.setter
    def target_cooling_temperature(self, target_cooling_temperature: Union[float, Tuple[float, float]]):
        target_cooling_temperature = self._get_property_value(target_cooling_temperature, (7.0, 10.0))
        self.__target_cooling_temperature = target_cooling_temperature

    @ElectricDevice.efficiency.setter
    def efficiency(self, efficiency: Union[float, Tuple[float, float]]):
        efficiency = self._get_property_value(efficiency, (0.2, 0.3))
        ElectricDevice.efficiency.fset(self, efficiency)

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'target_heating_temperature': self.target_heating_temperature,
            'target_cooling_temperature': self.target_cooling_temperature,
        }

    def get_cop(self, outdoor_dry_bulb_temperature: Union[float, Iterable[float]], heating: bool) -> Union[float, Iterable[float]]:
        r"""Return coefficient of performance.

        Calculate the Carnot cycle COP for heating or cooling mode. COP is set to 20 if < 0 or > 20.

        Parameters
        ----------
        outdoor_dry_bulb_temperature : Union[float, Iterable[float]]
            Outdoor dry bulb temperature in [C].
        heating : bool
            If `True` return the heating COP else return cooling COP.

        Returns
        -------
        cop : Union[float, Iterable[float]]
            COP as single value or time series depending on input parameter types.

        Notes
        -----
        heating_cop = (`t_target_heating` + 273.15)*`efficiency`/(`t_target_heating` - outdoor_dry_bulb_temperature)
        cooling_cop = (`t_target_cooling` + 273.15)*`efficiency`/(outdoor_dry_bulb_temperature - `t_target_cooling`)
        """

        c_to_k = lambda x: x + 273.15
        outdoor_dry_bulb_temperature = np.array(outdoor_dry_bulb_temperature)

        if heating:
            cop = self.efficiency*c_to_k(self.target_heating_temperature)/(self.target_heating_temperature - outdoor_dry_bulb_temperature)
        else:
            cop = self.efficiency*c_to_k(self.target_cooling_temperature)/(outdoor_dry_bulb_temperature - self.target_cooling_temperature)
        
        cop = np.array(cop)
        cop[cop < 0] = 20
        cop[cop > 20] = 20
        return cop

    def get_max_output_power(self, outdoor_dry_bulb_temperature: Union[float, Iterable[float]], heating: bool, max_electric_power: Union[float, Iterable[float]] = None) -> Union[float, Iterable[float]]:
        r"""Return maximum output power.

        Calculate maximum output power from heat pump given `cop`, `available_nominal_power` and `max_electric_power` limitations.

        Parameters
        ----------
        outdoor_dry_bulb_temperature : Union[float, Iterable[float]]
            Outdoor dry bulb temperature in [C].
        heating : bool
            If `True` use heating COP else use cooling COP.
        max_electric_power : Union[float, Iterable[float]], optional
            Maximum amount of electric power that the heat pump can consume from the power grid.

        Returns
        -------
        max_output_power : Union[float, Iterable[float]]
            Maximum output power as single value or time series depending on input parameter types.

        Notes
        -----
        max_output_power = min(max_electric_power, `available_nominal_power`)*cop
        """

        cop = self.get_cop(outdoor_dry_bulb_temperature, heating)

        if max_electric_power is None: 
            return self.available_nominal_power*cop  
        else:
            return np.min([max_electric_power, self.available_nominal_power], axis=0)*cop

    def get_input_power(self, output_power: Union[float, Iterable[float]], outdoor_dry_bulb_temperature: Union[float, Iterable[float]], heating: bool) -> Union[float, Iterable[float]]:
        r"""Return input power.

        Calculate power needed to meet `output_power` given `cop` limitations.

        Parameters
        ----------
        output_power : Union[float, Iterable[float]]
            Output power from heat pump
        outdoor_dry_bulb_temperature : Union[float, Iterable[float]]
            Outdoor dry bulb temperature in [C].
        heating : bool
            If `True` use heating COP else use cooling COP.

        Returns
        -------
        input_power : Union[float, Iterable[float]]
            Input power as single value or time series depending on input parameter types.

        Notes
        -----
        input_power = output_power/cop
        """

        return output_power/self.get_cop(outdoor_dry_bulb_temperature, heating)

    def autosize(self, outdoor_dry_bulb_temperature: Iterable[float], cooling_demand: Iterable[float] = None, heating_demand: Iterable[float] = None, safety_factor: Union[float, Tuple[float, float]] = None) -> float:
        r"""Autosize `nominal_power`.

        Set `nominal_power` to the minimum power needed to always meet `cooling_demand` + `heating_demand`.

        Parameters
        ----------
        outdoor_dry_bulb_temperature : Union[float, Iterable[float]]
            Outdoor dry bulb temperature in [C].
        cooling_demand : Union[float, Iterable[float]], optional
            Cooling demand in [kWh].
        heating_demand : Union[float, Iterable[float]], optional
            Heating demand in [kWh].
        safety_factor : Union[float, Tuple[float, float]], default: 1.0
            `nominal_power` is oversized by factor of `safety_factor`.

        Returns
        -------
        nominal_power : float
            Autosized nominal power

        Notes
        -----
        `nominal_power` = max((cooling_demand/cooling_cop) + (heating_demand/heating_cop))*safety_factor
        """
        
        safety_factor = self._get_property_value(safety_factor, 1.0)


        if cooling_demand is not None:
            cooling_demand = cooling_demand * self.time_step_ratio 
            cooling_nominal_power = np.array(cooling_demand)/self.get_cop(outdoor_dry_bulb_temperature, False)
        else:
            cooling_nominal_power = 0
        
        if heating_demand is not None:
            heating_demand = heating_demand * self.time_step_ratio 
            heating_nominal_power = np.array(heating_demand)/self.get_cop(outdoor_dry_bulb_temperature, True)
        else:
            heating_nominal_power = 0

        nominal_power = np.nanmax(cooling_nominal_power + heating_nominal_power)*safety_factor

        return nominal_power

class ElectricHeater(ElectricDevice):
    r"""Base electric heater class.

    Parameters
    ----------
    nominal_power : float, default: (0.9, 0.99)
        Maximum amount of electric power that the electric heater can consume from the power grid.
    efficiency : Union[float, Tuple[float, float]], default: 0.9
        Technical efficiency.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, nominal_power: float = None, efficiency: Union[float, Tuple[float, float]] = None, **kwargs: Any):
        super().__init__(nominal_power = nominal_power, efficiency = efficiency, **kwargs)

    @ElectricDevice.efficiency.setter
    def efficiency(self, efficiency: float):
        efficiency = self._get_property_value(efficiency, (0.9, 0.99))
        ElectricDevice.efficiency.fset(self, efficiency)

    def get_max_output_power(self, max_electric_power: Union[float, Iterable[float]] = None) -> Union[float, Iterable[float]]:
        r"""Return maximum output power.

        Calculate maximum output power from heat pump given `max_electric_power` limitations.

        Parameters
        ----------
        max_electric_power : Union[float, Iterable[float]], optional
            Maximum amount of electric power that the heat pump can consume from the power grid.

        Returns
        -------
        max_output_power : Union[float, Iterable[float]]
            Maximum output power as single value or time series depending on input parameter types.

        Notes
        -----
        max_output_power = min(max_electric_power, `available_nominal_power`)*`efficiency`
        """

        if max_electric_power is None:
            return self.available_nominal_power*self.efficiency
        else:
            return np.min([max_electric_power, self.available_nominal_power], axis=0)*self.efficiency

    def get_input_power(self, output_power: Union[float, Iterable[float]]) -> Union[float, Iterable[float]]:
        r"""Return input power.

        Calculate power demand to meet `output_power`.

        Parameters
        ----------
        output_power : Union[float, Iterable[float]] 
            Output power from heat pump

        Returns
        -------
        input_power : Union[float, Iterable[float]]
            Input power as single value or time series depending on input parameter types.

        Notes
        -----
        input_power = output_power/`efficiency`
        """

        return np.array(output_power)/self.efficiency

    def autosize(self, demand: Iterable[float], safety_factor: Union[float, Tuple[float, float]] = None) -> float:
        r"""Autosize `nominal_power`.

        Set `nominal_power` to the minimum power needed to always meet `demand`.

        Parameters
        ----------
        demand : Union[float, Iterable[float]], optional
            Heating demand in [kWh].
        safety_factor : Union[float, Tuple[float, float]], default: 1.0
            `nominal_power` is oversized by factor of `safety_factor`.

        Returns
        -------
        nominal_power : float
            Autosized nominal power

        Notes
        -----
        `nominal_power` = max(demand/`efficiency`)*safety_factor
        """
        demand = demand * self.time_step_ratio
        safety_factor = safety_factor = self._get_property_value(safety_factor, 1.0)
        nominal_power = np.nanmax(np.array(demand)/self.efficiency)*safety_factor

        return nominal_power

class PV(ElectricDevice):
    r"""Base photovoltaic array class.

    Parameters
    ----------
    nominal_power : float, default: 0.0
        PV array output power in [kW]. Must be >= 0.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, nominal_power: float = None, **kwargs: Any):
        super().__init__(nominal_power=nominal_power, **kwargs)

    def get_generation(self, inverter_ac_power_per_kw: Union[float, Iterable[float]]) -> Union[float, Iterable[float]]:
        r"""Get solar generation output.

        Parameters
        ----------
        inverter_ac_power_perk_w : Union[float, Iterable[float]]
            Inverter AC power output per kW of PV capacity in [W/kW].

        Returns
        -------
        generation : Union[float, Iterable[float]]
            Solar generation as single value or time series depending on input parameter types.

        Notes
        -----
        .. math::
            \textrm{generation} = \frac{\textrm{capacity} \times \textrm{inverter_ac_power_per_w}}{1000}
        """

        return self.nominal_power*np.array(inverter_ac_power_per_kw)/1000.0

    def autosize(self, demand: float, epw_filepath: Union[Path, str], use_sample_target: bool = None, zero_net_energy_proportion: Union[float, Tuple[float, float]] = None, roof_area: float = None, safety_factor: Union[float, Tuple[float, float]] = None, sizing_data: pd.DataFrame = None) -> Tuple[float, np.ndarray]:
        r"""Autosize `nominal_power` and `inverter_ac_power_per_kw`.

        Samples PV data from Tracking the Sun dataset to set PV system design parameters in System Adivosry Model's `PVWattsNone` model.
        The PV is sized to generate `zero_net_energy_proportion` of `annual_demand` limited by the `roof_area`. It is assumed that
        the building's roof is suitable for the installation tilt and azimuth in the sampled data.

        Parameters
        ----------
        demand : float
            Building annual demand in [kWh].
        epw_filepath : Union[Path, str]
            EnergyPlus weather file path used as input to :code:`PVWattsNone` model.
        use_sample_target : bool, default: False
            Whether to directly use the sizing in the sampled instance instead of sizing for `zero_net_energy_proportion`.
            Will still limit the size to the `roof_area`.
        zero_net_energy_proportion : Union[float, Tuple[float, float]], default: (0.7, 1.0)
            Proportion
        roof_area : float, optional
            Roof area where the PV is mounted in m^2. The default is to assume an infinite roof area.
        safety_factor : Union[float, Tuple[float, float]], default: 1.0
            The `nominal_power` is oversized by factor of `safety_factor`.
            It is only applied to the `zero_net_energy_proportion` estimate.
        sizing_data: pd.DataFrame, optional
            The sizing dataframe from which PV systems are sampled from. If initialized from
            py:class:`citylearn.citylearn.CityLearnEnv`, the data is parsed in when autosizing
            a building's PV. If the dataframe is not provided it is read in using
            :py:meth:`citylearn.data.DataSet.get_pv_sizing_data`.

        Returns
        -------
        nominal_power : float
            Autosized nominal power.
        inverter_ac_power_per_kw : np.ndarray
            SAM :code:`ac` output for :code:`PVWattsNone` model.

        Notes
        -----
        Data source: https://github.com/intelligent-environments-lab/CityLearn/tree/master/citylearn/data/misc/lbl-tracking_the_sun_res-pv.csv.
        """

        zero_net_energy_proportion = self._get_property_value(zero_net_energy_proportion, (0.7, 1.0))
        safety_factor = self._get_property_value(safety_factor, 1.0)
        roof_area = np.inf if roof_area is None else roof_area
        use_sample_target = False if use_sample_target is None else use_sample_target

        sizing_data = DataSet().get_pv_sizing_data() if sizing_data is None else sizing_data
        random_seed = self.random_seed
        tries = 3

        for i in range(tries):
            self._autosize_config = sizing_data.sample(1, random_state=random_seed + i).iloc[0].to_dict()
            model = Pvwattsv8.default('PVWattsNone')
            pv_nominal_power = self.autosize_config['nameplate_capacity_module_1']/1000.0
            model.SystemDesign.system_capacity = pv_nominal_power
            model.SystemDesign.dc_ac_ratio = self.autosize_config['inverter_loading_ratio']
            model.SystemDesign.tilt = self.autosize_config['tilt_1']
            model.SystemDesign.azimuth = self.autosize_config['azimuth_1']
            model.SystemDesign.bifaciality = self.autosize_config['bifacial_module_1']*0.65
            model.SolarResource.solar_resource_file = epw_filepath
        
            try:
                model.execute()
                break

            except Exception as e:
                LOGGER.debug(f'Failed to simulate PVWatts using config: {self._autosize_config}')

                if i == tries - 1:
                    raise e
                
                else:
                    pass
        
        inverter_ac_power_per_kw = np.array(model.Outputs.ac, dtype='float32')/pv_nominal_power

        if use_sample_target:
            target_nominal_power = self.autosize_config['PV_system_size_DC']
        
        else:
            zne_nominal_power = demand/sum(inverter_ac_power_per_kw/1000.0)
            limited_zne_nominal_power = zne_nominal_power*zero_net_energy_proportion
            target_nominal_power = math.floor(limited_zne_nominal_power*safety_factor/pv_nominal_power)*pv_nominal_power

        module_area = self.autosize_config['module_area']
        pv_area = pv_nominal_power*5.263 if module_area is None or math.isnan(module_area) else module_area
        # Fix bug: roof_area OverflowError: cannot convert float infinity to integer
        if np.isinf(roof_area):
            roof_limit_nominal_power = np.inf
        else:
            roof_limit_nominal_power = math.floor(roof_area / pv_area) * pv_nominal_power

        nominal_power = min(max(target_nominal_power, pv_nominal_power), roof_limit_nominal_power)
        self._autosize_config = {
            **self.autosize_config,
            'demand': demand,
            'epw_filepath': epw_filepath,
            'use_sample_target': use_sample_target,
            'zero_net_energy_proportion': zero_net_energy_proportion,
            'roof_area': roof_area,
            'safety_factor': safety_factor,
            'pv_area': pv_area,
            'nameplate_capacity_module_1': model.SystemDesign.system_capacity,
            'bifacial_module_1': model.SystemDesign.bifaciality,
            'target_nominal_power': target_nominal_power,
            'roof_limit_nominal_power': roof_limit_nominal_power,
            'nominal_power': nominal_power
        }
        
        return nominal_power, inverter_ac_power_per_kw

class StorageDevice(Device):
    r"""Base storage device class.

    Parameters
    ----------
    capacity : float, default: 0.0
        Maximum amount of energy the storage device can store in [kWh]. Must be >= 0.
    efficiency : Union[float, Tuple[float, float]], default: (0.90, 0.98)
        Technical efficiency.
    loss_coefficient : Union[float, Tuple[float, float]], default: (0.001, 0.009)
        Standby hourly losses. Must be between 0 and 1 (this value is often 0 or really close to 0).
    initial_soc : Union[float, Tuple[float, float]], default: 0.0
        State of charge when `time_step` = 0. Must be >= 0 and < `capacity`.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, capacity: float = None, efficiency: Union[float, Tuple[float, float]] = None, loss_coefficient: Union[float, Tuple[float, float]] = None, initial_soc: Union[float, Tuple[float, float]] = None, time_step_ratio:float = None, **kwargs: Any):
        self.random_seed = kwargs.get('random_seed', None)
        self.capacity = capacity
        self.loss_coefficient = loss_coefficient
        self.initial_soc = initial_soc
        self.time_step_ratio=time_step_ratio
        super().__init__(efficiency = efficiency, **kwargs)

    @property
    def capacity(self) -> float:
        r"""Maximum amount of energy the storage device can store in [kWh]."""

        return self.__capacity
    
    @property
    def time_step_ratio(self) -> float:
        r"""Maximum amount of energy the storage device can store in [kWh]."""

        return self.__time_step_ratio

    @property
    def loss_coefficient(self) -> float:
        r"""Standby hourly losses."""

        return self.__loss_coefficient * self.time_step_ratio
    
    @property
    def initial_soc(self) -> float:
        r"""State of charge when `time_step` = 0 in [kWh]."""

        return self.__initial_soc

    @property
    def soc(self) -> np.ndarray:
        r"""State of charge time series between [0, 1] in [:math:`\frac{\textrm{capacity}_{\textrm{charged}}}{\textrm{capacity}}`]."""

        return self.__soc

    @property
    def energy_init(self) -> float:
        r"""Latest energy level after accounting for standby hourly lossses in [kWh]."""
        if self.time_step == 0:
            return max(0.0, self.__soc[self.time_step]*self.capacity*(1 - self.loss_coefficient))
        return max(0.0, self.__soc[self.time_step - 1]*self.capacity*(1 - self.loss_coefficient))

    @property
    def energy_balance(self) -> np.ndarray:
        r"""Charged/discharged energy time series in [kWh]."""

        return self.__energy_balance * self.time_step_ratio
        
    @property
    def round_trip_efficiency(self) -> float:
        """Efficiency square root."""

        return self.efficiency**0.5

    @capacity.setter
    def capacity(self, capacity: float):
        capacity = 0.0 if capacity is None else capacity
        assert capacity >= 0, 'capacity must be >= 0.'
        self.__capacity = capacity

    @Device.efficiency.setter
    def efficiency(self, efficiency: float):
        efficiency = self._get_property_value(efficiency, (0.9, 0.98))
        Device.efficiency.fset(self, efficiency)

    @loss_coefficient.setter
    def loss_coefficient(self, loss_coefficient: Union[float, Tuple[float, float]]):
        loss_coefficient = self._get_property_value(loss_coefficient, (0.001, 0.009))
        assert 0 <= loss_coefficient <= 1, 'loss_coefficient must be >= 0 and <= 1.'
        self.__loss_coefficient = loss_coefficient

    @initial_soc.setter
    def initial_soc(self, initial_soc: Union[float, Tuple[float, float]]):
        initial_soc = self._get_property_value(initial_soc, 0.0)
        assert 0.0 <= initial_soc <= 1.0, 'initial_soc must be >= 0.0 and <= 1.0.'
        self.__initial_soc = initial_soc

    @time_step_ratio.setter
    def time_step_ratio(self, time_step_ratio: float):
        time_step_ratio = self._get_property_value(time_step_ratio, 1.0)
  
        self.__time_step_ratio = time_step_ratio    


    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'capacity': self.capacity,
            'loss_coefficient': self.loss_coefficient,
            'initial_soc': self.initial_soc,
            'round_trip_efficiency': self.round_trip_efficiency
        }

    def charge(self, energy: float):
        """Charges or discharges storage with respect to specified energy while considering `capacity` and `soc_init` limitations and, energy losses to the environment quantified by `round_trip_efficiency`.

        Parameters
        ----------
        energy : float
            Energy to charge if (+) or discharge if (-) in [kWh].

        Notes
        -----
        If charging, soc = min(`soc_init` + energy*`round_trip_efficiency`, `capacity`)
        If discharging, soc = max(0, `soc_init` + energy/`round_trip_efficiency`)
        """
        energy = energy * self.time_step_ratio
        energy_init = self.energy_init
        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        energy_final = min(energy_init + energy*self.round_trip_efficiency, self.capacity) if energy >= 0\
            else max(0.0, energy_init + energy/self.round_trip_efficiency)

        self.__soc[self.time_step] = energy_final/max(self.capacity, ZERO_DIVISION_PLACEHOLDER)
        self.__energy_balance[self.time_step] = self.set_energy_balance(energy_final, energy_init)

    def force_set_soc(self, soc: float):
        self.__soc[self.time_step] = soc

    def set_energy_balance(self, energy: float, energy_init:float) -> float:
        r"""Calculate energy balance.

        Parameters
        ----------
        energy: float
            Energy equivalent of state-of-charge in [kWh].
        energy_init: float
            Latest energy level after accounting for standby hourly lossses in [kWh]

        Returns
        -------
        energy: float
            Charged/discharged energy since last time step in [kWh]

        The energy balance is a derived quantity and is the product or quotient of the difference between consecutive SOCs and `round_trip_efficiency`
        for discharge or charge events respectively thus, thus accounts for energy losses to environment during charging and discharge. It is the
        actual energy charged/discharged irrespective of what is determined in the step function after taking into account storage design limits 
        e.g. maximum power input/output, capacity.
        """
        energy = energy * self.time_step_ratio
        energy -= energy_init
        energy_balance = energy/self.round_trip_efficiency if energy >= 0 else energy*self.round_trip_efficiency
        return energy_balance

    def autosize(self, demand: Iterable[float], safety_factor: Union[float, Tuple[float, float]] = None) -> float:
        r"""Autosize `capacity`.

        Set `capacity` to the minimum capacity needed to always meet `demand`.

        Parameters
        ----------
        demand : Union[float, Iterable[float]], optional
            Heating demand in [kWh].
        safety_factor : Union[float, Tuple[float, float]], default: (1.0, 2.0)
            The `capacity` is oversized by factor of `safety_factor`.

        Returns
        -------
        capacity : float
            Autosized cpacity.

        Notes
        -----
        `capacity` = max(demand/`efficiency`)*safety_factor
        """
        demand = demand * self.time_step_ratio
        safety_factor = self._get_property_value(safety_factor, (1.0, 2.0))
        capacity = np.nanmax(demand)*safety_factor

        return capacity

    def reset(self):
        r"""Reset `StorageDevice` to initial state."""

        super().reset()
        self.__soc = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')
        self.__soc[0] = self.initial_soc
        self.__energy_balance = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')

class StorageTank(StorageDevice):
    r"""Base thermal energy storage class.

    Parameters
    ----------
    capacity : float, default: 0.0
        Maximum amount of energy the storage device can store in [kWh]. Must be >= 0.
    max_output_power : float, optional
        Maximum amount of power that the storage unit can output [kW].
    max_input_power : float, optional
        Maximum amount of power that the storage unit can use to charge [kW].
    
    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, capacity: float = None, max_output_power: float = None, max_input_power: float = None, **kwargs: Any):
        super().__init__(capacity = capacity, **kwargs)
        self.max_output_power = max_output_power
        self.max_input_power = max_input_power

    @property
    def max_output_power(self) -> float:
        r"""Maximum amount of power that the storage unit can output [kW]."""

        return self.__max_output_power

    @property
    def max_input_power(self) -> float:
        r"""Maximum amount of power that the storage unit can use to charge [kW]."""

        return self.__max_input_power

    @max_output_power.setter
    def max_output_power(self, max_output_power: float):
        assert max_output_power is None or max_output_power >= 0, '`max_output_power` must be >= 0.'
        self.__max_output_power = max_output_power

    @max_input_power.setter
    def max_input_power(self, max_input_power: float):
        assert max_input_power is None or max_input_power >= 0, '`max_input_power` must be >= 0.'
        self.__max_input_power = max_input_power

    def charge(self, energy: float):
        """Charges or discharges storage with respect to specified energy while considering `capacity` and `soc_init` limitations and, energy losses to the environment quantified by `efficiency`.

        Parameters
        ----------
        energy : float
            Energy to charge if (+) or discharge if (-) in [kWh].

        Notes
        -----
        If charging, soc = min(`soc_init` + energy*`efficiency`, `max_input_power`, `capacity`)
        If discharging, soc = max(0, `soc_init` + energy/`efficiency`, `max_output_power`)
        """
        energy = energy * self.time_step_ratio

        if energy >= 0:    
            energy = energy if self.max_input_power is None else np.nanmin([energy, self.max_input_power])
        else:
            energy = energy if self.max_output_power is None else np.nanmax([-self.max_output_power, energy])
        
        super().charge(energy)

class Battery(StorageDevice, ElectricDevice):
    r"""Base electricity storage class.

    Parameters
    ----------
    capacity : float, default: 0.0
        Maximum amount of energy the storage device can store in [kWh]. Must be >= 0.
    nominal_power: float
        Maximum amount of electric power that the battery can use to charge or discharge.
    capacity_loss_coefficient : Union[float, Tuple[float, float]], default: (1e-5, 1e-4)
        Battery degradation; storage capacity lost in each charge and discharge cycle (as a fraction of the total capacity).
    power_efficiency_curve: list, default: [[0, 0.83],[0.3, 0.83],[0.7, 0.9],[0.8, 0.9],[1, 0.85]]
        Charging/Discharging efficiency as a function of nominal power.
    capacity_power_curve: list, default: [[0.0, 1],[0.8, 1],[1.0, 0.2]]   
        Maximum power of the battery as a function of its current state of charge.
    depth_of_discharge: Union[float, Tuple[float, float]], default: 1.0
        Maximum fraction of the battery that can be discharged relative to the total battery capacity.

    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super classes.
    """
    
    def __init__(self, capacity: float = None, nominal_power: float = None, capacity_loss_coefficient: Union[float, Tuple[float, float]] = None, power_efficiency_curve: List[List[float]] = None, capacity_power_curve: List[List[float]] = None, depth_of_discharge: Union[float, Tuple[float, float]] = None, time_step_ratio: float = None, **kwargs: Any):
        self._efficiency_history = []
        self._capacity_history = []
        self.random_seed = kwargs.get('random_seed', None)
        self.depth_of_discharge = depth_of_discharge
        super().__init__(capacity=capacity, nominal_power=nominal_power, time_step_ratio = time_step_ratio, **kwargs)
        self._capacity_history = [self.capacity]
        self.capacity_loss_coefficient = capacity_loss_coefficient
        self.power_efficiency_curve = power_efficiency_curve
        self.capacity_power_curve = capacity_power_curve
        self.time_step_ratio=time_step_ratio

    @StorageDevice.efficiency.getter
    def efficiency(self) -> float:
        """Current time step technical efficiency."""

        return self.efficiency_history[-1]
    
    @property
    def degraded_capacity(self) -> float:
        r"""Maximum amount of energy the storage device can store after degradation in [kWh]."""

        return self.capacity_history[-1]

    @property
    def capacity_loss_coefficient(self) -> float:
        """Battery degradation; storage capacity lost in each charge and discharge cycle (as a fraction of the total capacity)."""

        return self.__capacity_loss_coefficient

    @property
    def power_efficiency_curve(self) -> np.ndarray:
        """Charging/Discharging efficiency as a function of the nomianl power."""

        return self.__power_efficiency_curve

    @property
    def capacity_power_curve(self) -> np.ndarray:
        """Maximum power of the battery as a function of its current state of charge."""

        return self.__capacity_power_curve
    
    @property
    def time_step_ratio(self) -> float:
        """Maximum power of the battery as a function of its current state of charge."""

        return self.__time_step_ratio
    
    @property
    def depth_of_discharge(self) -> float:
        """Maximum fraction of the battery that can be discharged relative to the total battery capacity."""

        return self.__depth_of_discharge

    @property
    def efficiency_history(self) -> List[float]:
        """Time series of technical efficiency."""

        return self._efficiency_history

    @property
    def capacity_history(self) -> List[float]:
        """Time series of maximum amount of energy the storage device can store in [kWh]."""

        return self._capacity_history
    
    @StorageDevice.capacity.setter
    def capacity(self, capacity: Union[float, Tuple[float, float]]):
        StorageDevice.capacity.fset(self, capacity)
        self._capacity_history = [super().capacity]

    @efficiency.setter
    def efficiency(self, efficiency: Union[float, Tuple[float, float]]):
        StorageDevice.efficiency.fset(self, efficiency)
        self._efficiency_history.append(super().efficiency)

    @capacity_loss_coefficient.setter
    def capacity_loss_coefficient(self, capacity_loss_coefficient: Union[float, Tuple[float, float]]):
        capacity_loss_coefficient = self._get_property_value(capacity_loss_coefficient, (1e-5, 1e-4))
        self.__capacity_loss_coefficient = capacity_loss_coefficient

    @power_efficiency_curve.setter
    def power_efficiency_curve(self, power_efficiency_curve: List[List[float]]):
        if power_efficiency_curve is None:
            power_efficiency_curve = [
                [0, self.numpy_random_state.uniform(self.efficiency*0.85, self.efficiency*0.90)],
                [self.numpy_random_state.uniform(0.25, 0.35), self.numpy_random_state.uniform(self.efficiency*0.90, self.efficiency*0.95)],
                [self.numpy_random_state.uniform(0.65, 0.75), self.numpy_random_state.uniform(self.efficiency*0.98, self.efficiency*1.0)],
                [self.numpy_random_state.uniform(0.75, 0.85), self.efficiency],
                [1, self.numpy_random_state.uniform(self.efficiency*0.95, self.efficiency*0.98)]
            ]
        else:
            pass

        self.__power_efficiency_curve = np.array(power_efficiency_curve).T

    @capacity_power_curve.setter
    def capacity_power_curve(self, capacity_power_curve: List[List[float]]):
        if capacity_power_curve is None:
            capacity_power_curve = [
                [0.0, self.numpy_random_state.uniform(0.95, 1.0)],
                [self.numpy_random_state.uniform(0.75, 0.85), self.numpy_random_state.uniform(0.90, 0.95)],
                [1.0, self.numpy_random_state.uniform(0.20, 0.30)]
            ]
        else:
            pass

        self.__capacity_power_curve = np.array(capacity_power_curve).T

    @StorageDevice.initial_soc.setter
    def initial_soc(self, initial_soc: float):
        initial_soc = 1.0 - self.depth_of_discharge if initial_soc is None else initial_soc
        StorageDevice.initial_soc.fset(self, initial_soc)

    @depth_of_discharge.setter
    def depth_of_discharge(self, depth_of_discharge: float):
        self.__depth_of_discharge = self._get_property_value(depth_of_discharge, 1.0)

    @time_step_ratio.setter
    def time_step_ratio(self, time_step_ratio: float):
        self.__time_step_ratio = self._get_property_value(time_step_ratio, 1.0)     

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'depth_of_discharge': self.depth_of_discharge,
            'capacity_loss_coefficient': self.capacity_loss_coefficient,
            'power_efficiency_curve': self.power_efficiency_curve,
            'capacity_power_curve': self.capacity_power_curve,
        }

    def charge(self, energy: float):
        """Charges or discharges storage with respect to specified energy while considering `capacity` degradation and `soc_init` 
        limitations, losses to the environment quantified by `efficiency`, `power_efficiency_curve` and `capacity_power_curve`.

        Parameters
        ----------
        energy : float
            Energy to charge if (+) or discharge if (-) in [kWh].
        """
        energy = energy * self.time_step_ratio # Normalise energy with the time_step_ratio
        action_energy = energy

        if energy >= 0:
            energy_wrt_degrade = self.degraded_capacity - self.energy_init
            max_input_power = self.get_max_input_power()
            energy = min(max_input_power, self.available_nominal_power, energy_wrt_degrade, energy)
            self.efficiency = self.get_current_efficiency(min(action_energy, max_input_power))

        else:
            soc_limit_wrt_dod = 1.0 - self.depth_of_discharge
            soc_init = self.soc[self.time_step - 1] if self.time_step > 0 else self.soc[self.time_step]
            soc_difference = soc_init - soc_limit_wrt_dod
            energy_limit_wrt_dod = max(soc_difference * self.capacity * self.round_trip_efficiency, 0.0) * -1
            max_output_power = self.get_max_output_power()
            energy = max(-max_output_power, energy_limit_wrt_dod, energy)
            self.efficiency = self.get_current_efficiency(min(abs(action_energy), max_output_power))

        super().charge(energy)
        degraded_capacity = max(self.degraded_capacity - self.degrade(), 0.0)
        self._capacity_history.append(degraded_capacity)
        self.update_electricity_consumption(self.energy_balance[self.time_step], enforce_polarity=False)

    def get_max_output_power(self) -> float:
        r"""Get maximum output power while considering `capacity_power_curve` limitations if defined otherwise, returns `nominal_power`.

        Returns
        -------
        max_output_power : float
            Maximum amount of power that the storage unit can output [kW].
        """

        return self.get_max_input_power()

    def get_max_input_power(self) -> float:
        r"""Get maximum input power while considering `capacity_power_curve` limitations.

        Returns
        -------
        max_input_power : float
            Maximum amount of power that the storage unit can use to charge [kW].
        """

        #The initial SOC is the previous SOC minus the energy losses
        soc = self.energy_init/max(self.capacity, ZERO_DIVISION_PLACEHOLDER)

        # Calculating the maximum power rate at which the battery can be charged or discharged
        idx = max(0, np.argmax(soc <= self.capacity_power_curve[0]) - 1)
        max_output_power = self.nominal_power*(
            self.capacity_power_curve[1][idx] 
            + (self.capacity_power_curve[1][idx+1] - self.capacity_power_curve[1][idx])*(soc - self.capacity_power_curve[0][idx])
            /(self.capacity_power_curve[0][idx+1] - self.capacity_power_curve[0][idx])
        )
        
        return max_output_power

    def get_current_efficiency(self, energy: float) -> float:
        r"""Get technical efficiency while considering `power_efficiency_curve` limitations.

        Returns
        -------
        efficiency : float
            Technical efficiency.
        """

        # Calculating the maximum power rate at which the battery can be charged or discharged
        energy_normalized = np.abs(energy)/max(self.nominal_power, ZERO_DIVISION_PLACEHOLDER)
        idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
        efficiency = self.power_efficiency_curve[1][idx]\
            + (energy_normalized - self.power_efficiency_curve[0][idx]
            )*(self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx]
            )/(self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])

        return efficiency

    def force_set_soc(self, soc: float):
        """
        Forcefully set the battery's state-of-charge (SOC) for the current time step,
        bypassing restrictions such as efficiency losses, power limits, and degradation.

        This is used for reconnections of the EV to the platform.

        Parameters
        ----------
        soc : float
            Desired state-of-charge as a fraction (between 0 and 1). Values outside this range are not accepted.
        """
        # Ensure soc is between 0 and 1
        if soc < 0 or soc > 1:
            raise AttributeError("Soc must be between 0 and 1. Check your dataset")
        # Directly update the internal SOC array.
        # Note: __soc is defined in the StorageDevice class, so we access it via name mangling.
        super().force_set_soc(soc)

    def degrade(self) -> float:
        r"""Get amount of capacity degradation.

        Returns
        -------
        capacity : float
            Maximum amount of energy the storage device can store in [kWh].
        """

        # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge
        capacity_degrade = self.capacity_loss_coefficient*self.capacity*np.abs(self.energy_balance[self.time_step])/(2*max(self.degraded_capacity, ZERO_DIVISION_PLACEHOLDER))
        return capacity_degrade * self.time_step_ratio # Normalize with time_step_ratio (seconds_per_timestep/schema_time_delta)    
    
    def autosize(
        self, demand: float, duration: Union[float, Tuple[float, float]] = None, parallel: bool = None, safety_factor: Union[float, Tuple[float, float]] = None,
        sizing_data: pd.DataFrame = None
    ) -> Tuple[float, float, float, float, float, float]:
        r"""Randomly selects a battery from the internally defined real world manufacturer model and autosizes its parameters.

        The total capacity and nominal power are autosized to meet the hourly demand for a specified duration. It is assumed that
        there is no limit on the number of batteries that can be connected in series or parallel for any of the battery models.

        Parameters
        ----------
        demand : float
            Hourly, building demand to be met for duration.
        duration : Union[float, Tuple[float, float]], default : (1.5, 3.5)
            Number of hours the sized battery should be able to meet demand.
        parallel : bool, default : False
            Whether to assume multiple batteries are connected in parallel so
            that the maximum nominal power is the product of the unit count and
            the nominal_power of one battery i.e., increasing number of battery
            units also increases nominal power.
        safety_factor : Union[float, Tuple[float, float]], default: 1.0
            The `target capacity is oversized by factor of `safety_factor`.

        Returns
        -------
        capacity : float
            Selected battery's autosized capacity to meet demand for duration.
        nominal_power : float
            Selected battery's autosized nominal power to meet demand for duration.
        depth_of_discharge : float
            Selected battery depth-of-discharge.
        efficiency : float
            Selected battery efficiency.
        loss_coefficient : float
            Selected battery loss coefficient.
        capacity_loss_coefficient : float
            Selected battery capacity loss coefficient.
        sizing_data: pd.DataFrame, optional
            The sizing dataframe from which batteries systems are sampled from. If initialized from
            py:class:`citylearn.citylearn.CityLearnEnv`, the data is parsed in when autosizing
            a building's battery. If the dataframe is not provided it is read in using
            :py:meth:`citylearn.data.DataSet.get_battery_sizing_data`.

        Notes
        -----
        Data source: https://github.com/intelligent-environments-lab/CityLearn/tree/master/citylearn/data/misc/battery_choices.yaml.
        """

        demand = demand * self.time_step_ratio
        duration = self._get_property_value(duration, (1.5, 3.5))
        safety_factor = self._get_property_value(safety_factor, 1.0)
        parallel = False if parallel is None else parallel

        sizing_data = DataSet().get_battery_sizing_data() if sizing_data is None else sizing_data
        choices = sizing_data[sizing_data['nominal_power']<=demand].copy()

        if choices.shape[0] == 0:
            choices = sizing_data.sort_values('nominal_power').iloc[0:1].copy()
        
        else:
            pass
        
        choices = choices.to_dict('index')
        choice = self.numpy_random_state.choice(list(choices.keys()))
        target_capacity = demand*duration*safety_factor
        unit_count = max(1, math.floor(target_capacity/choices[choice]['capacity']))
        
        capacity = choices[choice]['capacity']*unit_count
        nominal_power = choices[choice]['nominal_power']*max(1.0, unit_count*int(parallel))
        depth_of_discharge = choices[choice]['depth_of_discharge']
        efficiency = choices[choice]['efficiency']
        loss_coefficient = choices[choice]['loss_coefficient']
        capacity_loss_coefficient = choices[choice]['capacity_loss_coefficient']
        
        self._autosize_config = {
            'model': choice,
            'demand': demand,
            'duration': duration,
            'safety_factor': safety_factor,
            'unit_count': unit_count,
            **choices[choice],
        }

        return capacity, nominal_power, depth_of_discharge, efficiency, loss_coefficient, capacity_loss_coefficient

    def as_dict(self) -> dict:
        """
        Return a dictionary representation of the current state for use in rendering or logging.
        """
        return {
            'Battery Soc-%': self.soc[self.time_step],
            'Battery (Dis)Charge-kWh': self.energy_balance[self.time_step]
        }

    def reset(self):
        r"""Reset `Battery` to initial state."""

        super().reset()
        self._efficiency_history = self._efficiency_history[0:1]
        self._capacity_history = self._capacity_history[0:1]

class WashingMachine(ElectricDevice):
    """Represents a smart washing machine controlled via time-varying load profiles (kWh over time) instead of predefined fixed cycles."""

    def __init__(
        self,
        washing_machine_simulation: WashingMachineSimulation = None,
        name: str = None,
        **kwargs
    ):  
        """Initialize the washing machine with optional simulation data and a unique name."""
        self.washing_machine_simulation = washing_machine_simulation
        self.name = name
        self.__initiated = False
        super().__init__(**kwargs)

    @property
    def washing_machine_simulation(self) -> WashingMachineSimulation:
        """Returns the associated washing machine simulation containing time-based load profiles."""
        return self.__washing_machine_simulation

    @washing_machine_simulation.setter
    def washing_machine_simulation(self, washing_machine_simulation: WashingMachineSimulation):
        """Sets the simulation object for this washing machine."""
        self.__washing_machine_simulation = washing_machine_simulation    

    @property
    def name(self) -> str:
        """Returns the unique identifier or name of the washing machine."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """Sets the unique name of the washing machine."""
        self.__name = name        

    @property
    def initiated(self) -> bool:
        """Indicates whether a washing cycle has been initiated in the current time step."""
        return self.__initiated

    @property
    def past_action_values(self) -> np.ndarray:
        """Returns the history of control actions issued to this washing machine."""
        return self.__past_action_values

    def next_time_step(self):
        """Advance the simulation by one time step and update internal state and buffers accordingly."""
        super().next_time_step()

        if self.__past_action_values is None:
            self.__past_action_values = np.zeros(
                self.episode_tracker.episode_time_steps, dtype='float32'
            )
        if self._ElectricDevice__electricity_consumption is None:
            self._ElectricDevice__electricity_consumption = np.zeros(
                self.episode_tracker.episode_time_steps, dtype='float32'
            )

        # Reset cycle initiation if the configured cycle boundaries change between steps
        if self.time_step > 0:
            prev_start = self.washing_machine_simulation.wm_start_time_step[self.time_step - 1]
            curr_start = self.washing_machine_simulation.wm_start_time_step[self.time_step]
            prev_end = self.washing_machine_simulation.wm_end_time_step[self.time_step - 1]
            curr_end = self.washing_machine_simulation.wm_end_time_step[self.time_step]
            if (prev_start != curr_start or prev_end != curr_end) and self.initiated:
                self.__initiated = False

    def start_cycle(self, action_value: float):
        """Trigger a washing cycle if conditions are met and apply the associated load profile to power consumption."""
        self.__past_action_values[self.time_step] = action_value

        start_time_step = self.washing_machine_simulation.wm_start_time_step[self.time_step]
        end__time_step = self.washing_machine_simulation.wm_end_time_step[self.time_step]

        if not self.initiated and action_value > 0 and start_time_step != -1 and end__time_step != -1 and start_time_step <= self.time_step <= end__time_step:
            load_profile = self.washing_machine_simulation.load_profile[self.time_step]
            if len(load_profile) == 0:
                print("No load profile available at this step.")
                return

            self.__initiated = True

            # Apply load values from the profile to the internal electricity usage
            for offset, load in enumerate(load_profile):
                step = self.time_step + offset
                if step < self.episode_tracker.episode_time_steps:
                    self.update_electricity_consumption(load, enforce_polarity=False)

    def observations(self) -> Mapping[str, float]:
        """Return the current observation dictionary including simulation inputs and machine state."""
        unwanted_keys = []  # Add any keys you want to exclude

        observations = {
            **{
                k.lstrip('_'): v[self.time_step]
                for k, v in vars(self.washing_machine_simulation).items()
                if isinstance(v, np.ndarray) and k.lstrip('_') not in unwanted_keys
            },
            'washing_machine_initiated': float(self.initiated),
            'washing_machine_action': self.past_action_values[self.time_step] if self.past_action_values is not None else 0.0
        }

        return observations    

    def reset(self):
        """Reset the internal state of the washing machine at the beginning of a new episode."""
        super().reset()
        self.__initiated = False
        self.__past_action_values = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32') 
        self._ElectricDevice__electricity_consumption = np.zeros(self.episode_tracker.episode_time_steps, dtype='float32')

    def __str__(self) -> str:
        """Return a human-readable string representation of the washing machine's current state."""
        return str(self.as_dict())

    def as_dict(self) -> dict:
        """Return the current state of the washing machine as a dictionary."""
        return {
            'name': self.name,
            'initiated': self.initiated,
            **self.observations()
        }

    def render_simulation_end_data(self) -> dict:
        """Generate structured simulation output data for all time steps."""
        num_steps = self.episode_tracker.episode_time_steps
        simulation_attrs = {
            key: value
            for key, value in vars(self.washing_machine_simulation).items()
            if isinstance(value, np.ndarray)
        }

        time_steps = []
        for i in range(num_steps):
            step_data = {
                "time_step": i,
                "simulation": {},
                "status": {
                    "initiated": self.initiated if i == self.time_step else None,
                    "action_value": self.past_action_values[i] if self.past_action_values is not None else None
                }
            }

            for key, array in simulation_attrs.items():
                value = array[i]
                if isinstance(value, np.generic):
                    value = value.item()
                step_data["simulation"][key] = value

            time_steps.append(step_data)

        return {
            "simulation_name": self.name if self.name else "WashingMachineSimulation",
            "data": time_steps
        }