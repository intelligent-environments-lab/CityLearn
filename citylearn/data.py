import os
from pathlib import Path
import shutil
from typing import Iterable, List, Union
import numpy as np

from citylearn.utilities import read_json

class DataSet:
    """CityLearn input data set and schema class."""
    
    __ROOT_DIRECTORY = os.path.join(os.path.dirname(__file__),'data')

    @staticmethod
    def get_names() -> List[str]:
        """Returns list of internally stored CityLearn datasets that are `schema` 
        names and can be used to initialize `citylearn.citylearn.CityLearnEnv`.
        
        Returns
        -------
        names: List[str]
            schema names 
        """
        
        return sorted([
            d for d in os.listdir(DataSet.__ROOT_DIRECTORY) 
            if os.path.isdir(os.path.join(DataSet.__ROOT_DIRECTORY,d))
        ])

    @staticmethod
    def copy(name: str, destination_directory: Union[Path, str] = None):
        """Copies an internally stored CityLearn dataset to a location of choice.

        Parameters
        ----------
        destination_directory: Union[Path, str], optional
            Target directory to copy data set files to. Copies to current directory if not specifed.
        """
        
        source_directory = os.path.join(DataSet.__ROOT_DIRECTORY,name)
        destination_directory = '' if destination_directory is None else destination_directory
        destination_directory = os.path.join(destination_directory,name)
        os.makedirs(destination_directory,exist_ok=True)

        for f in os.listdir(source_directory):
            if f.endswith('.csv') or f.endswith('.json'):
                source_filepath = os.path.join(source_directory,f)
                destination_filepath = os.path.join(destination_directory,f)
                shutil.copy(source_filepath,destination_filepath)
            else:
                continue

    @staticmethod
    def get_schema(name: str) -> dict:
        """Returns a data set's schema.

        Parameters
        ----------
        name: str
            Name of data set.

        Returns
        -------
        schema: dict
            Data set schema.
        """
        
        root_directory = os.path.join(DataSet.__ROOT_DIRECTORY,name)
        filepath = os.path.join(root_directory,'schema.json')
        schema = read_json(filepath)
        schema['root_directory'] = root_directory
        
        return schema

class EnergySimulation:
    """`Building` `energy_simulation` data class.

    Parameters
    ----------
    month : np.array
        Month time series value ranging from 1 - 12.
    hour : np.array
        Hour time series value ranging from 1 - 24.
    day_type : np.array
        Numeric day of week time series ranging from 1 - 8 where 1 - 7 is Monday - Sunday and 8 is reserved for special days e.g. holiday.
    daylight_savings_status : np.array
        Daylight saving status time series signal of 0 or 1 indicating inactive  or active daylight saving respectively.
    indoor_dry_bulb_temperature : np.array
        Average building dry bulb temperature time series in [C].
    average_unmet_cooling_setpoint_difference : np.array
        Average difference between `indoor_dry_bulb_temperature` and cooling temperature setpoints time series in [C].
    indoor_relative_humidity : np.array
        Average building relative humidity time series in [%].
    non_shiftable_load : np.array
        Total building non-shiftable plug and equipment loads time series in [kWh].
    dhw_demand : np.array
        Total building domestic hot water demand time series in [kWh].
    cooling_demand : np.array
        Total building space cooling demand time series in [kWh].
    heating_demand : np.array
        Total building space heating demand time series in [kWh].
    solar_generation : np.array
        Inverter output per 1 kW of PV system time series in [W/kW].
    occupant_count: np.array
        Building occupant count time series in [people].
    indoor_dry_bulb_temperature_set_point: np.array
        Average building dry bulb temperature set point time series in [C].
    hvac_mode: np.array, default: 1
        Heat pump and auxilliary electric heating availability. If 0, both HVAC devices are unavailable (off), if 1,
        the heat pump is available for space cooling and if 2, the heat pump and auxilliary electric heating are available
        for space heating only. The default is to set the mode to cooling at all times. The HVAC devices are always available
        for cooling and heating storage charging irrespective of the hvac mode.
    """

    def __init__(
        self, month: Iterable[int], hour: Iterable[int], day_type: Iterable[int],
        daylight_savings_status: Iterable[int], indoor_dry_bulb_temperature: Iterable[float], average_unmet_cooling_setpoint_difference: Iterable[float], indoor_relative_humidity: Iterable[float], 
        non_shiftable_load: Iterable[float], dhw_demand: Iterable[float], cooling_demand: Iterable[float], heating_demand: Iterable[float], solar_generation: Iterable[float], 
        occupant_count: Iterable[int] = None, indoor_dry_bulb_temperature_set_point: Iterable[int] = None, hvac_mode: Iterable[int] = None
    ):
        self.month = np.array(month, dtype = int)
        self.hour = np.array(hour, dtype = int)
        self.day_type = np.array(day_type, dtype = int)
        self.daylight_savings_status = np.array(daylight_savings_status, dtype = int)
        self.indoor_dry_bulb_temperature = np.array(indoor_dry_bulb_temperature, dtype = float)
        self.average_unmet_cooling_setpoint_difference = np.array(average_unmet_cooling_setpoint_difference, dtype = float)
        self.indoor_relative_humidity = np.array(indoor_relative_humidity, dtype = float)
        self.non_shiftable_load = np.array(non_shiftable_load, dtype = float)
        self.dhw_demand = np.array(dhw_demand, dtype = float)
        
        # set space demands and check there is not cooling and heating demand at same time step
        self.cooling_demand = np.array(cooling_demand, dtype = float)
        self.heating_demand = np.array(heating_demand, dtype = float)
        assert (self.cooling_demand*self.heating_demand).sum() == 0, 'Cooling and heating in the same time step is not allowed.'

        self.solar_generation = np.array(solar_generation, dtype = float)

        # optional
        self.occupant_count = np.zeros(len(solar_generation), dtype=float) if occupant_count is None else np.array(occupant_count, dtype=float)
        self.indoor_dry_bulb_temperature_set_point = np.zeros(len(solar_generation), dtype=float) if indoor_dry_bulb_temperature_set_point is None else np.array(indoor_dry_bulb_temperature_set_point, dtype=float)

        if hvac_mode is None:
            self.hvac_mode = np.zeros(len(solar_generation), dtype=float)*1 
        
        else:
            unique = list(set(hvac_mode))

            for i in range(3):
                try:
                    unique.remove(i)
                except ValueError:
                    pass

            assert len(unique) == 0, f'Invalid hvac_mode values were found: {unique}. Valid values are 0, 1 and 2 to indicate off, cooling mode and heating mode.'
            self.hvac_mode = np.array(hvac_mode, dtype=int)

class Weather:
    """`Building` `weather` data class.

    Parameters
    ----------
    outdoor_dry_bulb_temperature : np.array
        Outdoor dry bulb temperature time series in [C].
    outdoor_relative_humidity : np.array
        Outdoor relative humidity time series in [%].
    diffuse_solar_irradiance : np.array
        Diffuse solar irradiance time series in [W/m^2].
    direct_solar_irradiance : np.array
        Direct solar irradiance time series in [W/m^2].
    outdoor_dry_bulb_temperature_predicted_6h : np.array
        Outdoor dry bulb temperature 6 hours ahead prediction time series in [C].
    outdoor_dry_bulb_temperature_predicted_12h : np.array
        Outdoor dry bulb temperature 12 hours ahead prediction time series in [C].
    outdoor_dry_bulb_temperature_predicted_24h : np.array
        Outdoor dry bulb temperature 24 hours ahead prediction time series in [C].
    outdoor_relative_humidity_predicted_6h : np.array
        Outdoor relative humidity 6 hours ahead prediction time series in [%].
    outdoor_relative_humidity_predicted_12h : np.array
        Outdoor relative humidity 12 hours ahead prediction time series in [%].
    outdoor_relative_humidity_predicted_24h : np.array
        Outdoor relative humidity 24 hours ahead prediction time series in [%].
    diffuse_solar_irradiance_predicted_6h : np.array
        Diffuse solar irradiance 6 hours ahead prediction time series in [W/m^2].
    diffuse_solar_irradiance_predicted_12h : np.array
        Diffuse solar irradiance 12 hours ahead prediction time series in [W/m^2].
    diffuse_solar_irradiance_predicted_24h : np.array
        Diffuse solar irradiance 24 hours ahead prediction time series in [W/m^2].
    direct_solar_irradiance_predicted_6h : np.array
        Direct solar irradiance 6 hours ahead prediction time series in [W/m^2].
    direct_solar_irradiance_predicted_12h : np.array
        Direct solar irradiance 12 hours ahead prediction time series in [W/m^2].
    direct_solar_irradiance_predicted_24h : np.array
        Direct solar irradiance 24 hours ahead prediction time series in [W/m^2].
    """

    def __init__(
        self, outdoor_dry_bulb_temperature: Iterable[float], outdoor_relative_humidity: Iterable[float], diffuse_solar_irradiance: Iterable[float], direct_solar_irradiance: Iterable[float], 
        outdoor_dry_bulb_temperature_predicted_6h: Iterable[float], outdoor_dry_bulb_temperature_predicted_12h: Iterable[float], outdoor_dry_bulb_temperature_predicted_24h: Iterable[float],
        outdoor_relative_humidity_predicted_6h: Iterable[float], outdoor_relative_humidity_predicted_12h: Iterable[float], outdoor_relative_humidity_predicted_24h: Iterable[float],
        diffuse_solar_irradiance_predicted_6h: Iterable[float], diffuse_solar_irradiance_predicted_12h: Iterable[float], diffuse_solar_irradiance_predicted_24h: Iterable[float],
        direct_solar_irradiance_predicted_6h: Iterable[float], direct_solar_irradiance_predicted_12h: Iterable[float], direct_solar_irradiance_predicted_24h: Iterable[float],
    ):
        self.outdoor_dry_bulb_temperature = np.array(outdoor_dry_bulb_temperature, dtype = float)
        self.outdoor_relative_humidity = np.array(outdoor_relative_humidity, dtype = float)
        self.diffuse_solar_irradiance = np.array(diffuse_solar_irradiance, dtype = float)
        self.direct_solar_irradiance = np.array(direct_solar_irradiance, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_6h = np.array(outdoor_dry_bulb_temperature_predicted_6h, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_12h = np.array(outdoor_dry_bulb_temperature_predicted_12h, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_24h = np.array(outdoor_dry_bulb_temperature_predicted_24h, dtype = float)
        self.outdoor_relative_humidity_predicted_6h = np.array(outdoor_relative_humidity_predicted_6h, dtype = float)
        self.outdoor_relative_humidity_predicted_12h = np.array(outdoor_relative_humidity_predicted_12h, dtype = float)
        self.outdoor_relative_humidity_predicted_24h = np.array(outdoor_relative_humidity_predicted_24h, dtype = float)
        self.diffuse_solar_irradiance_predicted_6h = np.array(diffuse_solar_irradiance_predicted_6h, dtype = float)
        self.diffuse_solar_irradiance_predicted_12h = np.array(diffuse_solar_irradiance_predicted_12h, dtype = float)
        self.diffuse_solar_irradiance_predicted_24h = np.array(diffuse_solar_irradiance_predicted_24h, dtype = float)
        self.direct_solar_irradiance_predicted_6h = np.array(direct_solar_irradiance_predicted_6h, dtype = float)
        self.direct_solar_irradiance_predicted_12h = np.array(direct_solar_irradiance_predicted_12h, dtype = float)
        self.direct_solar_irradiance_predicted_24h = np.array(direct_solar_irradiance_predicted_24h, dtype = float)

class Pricing:
    """`Building` `pricing` data class.

    Parameters
    ----------
    electricity_pricing : np.array
        Electricity pricing time series in [$/kWh].
    electricity_pricing_predicted_6h : np.array
        Electricity pricing 6 hours ahead prediction time series in [$/kWh].
    electricity_pricing_predicted_12h : np.array
        Electricity pricing 12 hours ahead prediction time series in [$/kWh].
    electricity_pricing_predicted_24h : np.array
        Electricity pricing 24 hours ahead prediction time series in [$/kWh].
    """

    def __init__(
        self, electricity_pricing: Iterable[float], electricity_pricing_predicted_6h: Iterable[float], 
        electricity_pricing_predicted_12h: Iterable[float], electricity_pricing_predicted_24h: Iterable[float]
    ):
        self.electricity_pricing = np.array(electricity_pricing, dtype = float)
        self.electricity_pricing_predicted_6h = np.array(electricity_pricing_predicted_6h, dtype = float)
        self.electricity_pricing_predicted_12h = np.array(electricity_pricing_predicted_12h, dtype = float)
        self.electricity_pricing_predicted_24h = np.array(electricity_pricing_predicted_24h, dtype = float)

class CarbonIntensity:
    """`Building` `carbon_intensity` data class.

    Parameters
    ----------
    carbon_intensity : np.array
        Grid carbon emission rate time series in [kg_co2/kWh].
    """

    def __init__(self, carbon_intensity: Iterable[float]):
        self.carbon_intensity = np.array(carbon_intensity, dtype = float)