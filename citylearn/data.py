from typing import Iterable
import numpy as np

class EnergySimulation:
    def __init__(
        self, month: Iterable[int], hour: Iterable[int], day_type: Iterable[int],
        daylight_savings_status: Iterable[int], indoor_dry_bulb_temperature: Iterable[float], average_unmet_cooling_setpoint_difference: Iterable[float], indoor_relative_humidity: Iterable[float], 
        non_shiftable_load: Iterable[float], dhw_demand: Iterable[float], cooling_demand: Iterable[float], heating_demand: Iterable[float],
        solar_generation: Iterable[float]
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
        self.cooling_demand = np.array(cooling_demand, dtype = float)
        self.heating_demand = np.array(heating_demand, dtype = float)
        self.solar_generation = np.array(solar_generation, dtype = float)

class Weather:
    def __init__(
        self, outdoor_dry_bulb_temperature: Iterable[float], outdoor_relative_humidity: Iterable[float], diffuse_solar_radiation: Iterable[float], direct_solar_radiation: Iterable[float], 
        outdoor_dry_bulb_temperature_predicted_6h: Iterable[float], outdoor_dry_bulb_temperature_predicted_12h: Iterable[float], outdoor_dry_bulb_temperature_predicted_24h: Iterable[float],
        outdoor_relative_humidity_predicted_6h: Iterable[float], outdoor_relative_humidity_predicted_12h: Iterable[float], outdoor_relative_humidity_predicted_24h: Iterable[float],
        diffuse_solar_radiation_predicted_6h: Iterable[float], diffuse_solar_radiation_predicted_12h: Iterable[float], diffuse_solar_radiation_predicted_24h: Iterable[float],
        direct_solar_radiation_predicted_6h: Iterable[float], direct_solar_radiation_predicted_12h: Iterable[float], direct_solar_radiation_predicted_24h: Iterable[float],
    ):
        self.outdoor_dry_bulb_temperature = np.array(outdoor_dry_bulb_temperature, dtype = float)
        self.outdoor_relative_humidity = np.array(outdoor_relative_humidity, dtype = float)
        self.diffuse_solar_radiation = np.array(diffuse_solar_radiation, dtype = float)
        self.direct_solar_radiation = np.array(direct_solar_radiation, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_6h = np.array(outdoor_dry_bulb_temperature_predicted_6h, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_12h = np.array(outdoor_dry_bulb_temperature_predicted_12h, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_24h = np.array(outdoor_dry_bulb_temperature_predicted_24h, dtype = float)
        self.outdoor_relative_humidity_predicted_6h = np.array(outdoor_relative_humidity_predicted_6h, dtype = float)
        self.outdoor_relative_humidity_predicted_12h = np.array(outdoor_relative_humidity_predicted_12h, dtype = float)
        self.outdoor_relative_humidity_predicted_24h = np.array(outdoor_relative_humidity_predicted_24h, dtype = float)
        self.diffuse_solar_radiation_predicted_6h = np.array(diffuse_solar_radiation_predicted_6h, dtype = float)
        self.diffuse_solar_radiation_predicted_12h = np.array(diffuse_solar_radiation_predicted_12h, dtype = float)
        self.diffuse_solar_radiation_predicted_24h = np.array(diffuse_solar_radiation_predicted_24h, dtype = float)
        self.direct_solar_radiation_predicted_6h = np.array(direct_solar_radiation_predicted_6h, dtype = float)
        self.direct_solar_radiation_predicted_12h = np.array(direct_solar_radiation_predicted_12h, dtype = float)
        self.direct_solar_radiation_predicted_24h = np.array(direct_solar_radiation_predicted_24h, dtype = float)

class CarbonIntensity:
    def __init__(self, carbon_intensity: Iterable[float]):
        self.carbon_intensity = np.array(carbon_intensity, dtype = float)