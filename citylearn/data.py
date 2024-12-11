import logging
import os
from pathlib import Path
from platformdirs import user_cache_dir
import shutil
from typing import Any, Iterable, Mapping, List, Union
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from citylearn.__init__ import __version__
from citylearn.utilities import join_url, read_json, read_yaml, write_json

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO)

TOLERANCE = 0.0001
ZERO_DIVISION_PLACEHOLDER = 0.000001
MISC_DIRECTORY = os.path.join(os.path.dirname(__file__), 'misc')
QUERIES_DIRECTORY = os.path.join(MISC_DIRECTORY, 'queries')
SETTINGS_FILEPATH = os.path.join(MISC_DIRECTORY, 'settings.yaml')

def get_settings():
    directory = os.path.join(os.path.join(os.path.dirname(__file__), 'misc'))
    filepath = os.path.join(directory, 'settings.yaml')
    settings = read_yaml(filepath)

    return settings

class DataSet:
    """CityLearn input data set and schema class."""

    GITHUB_ACCOUNT = 'intelligent-environments-lab'
    REPOSITORY_NAME = 'CityLearn'
    REPOSITORY_TAG = f'v{__version__}'
    REPOSITORY_DATA_PATH = join_url('data')
    REPOSITORY_DATA_DATASETS_PATH = join_url(REPOSITORY_DATA_PATH, 'datasets')
    REPOSITORY_DATA_MISC_PATH = join_url(REPOSITORY_DATA_PATH, 'misc')
    GITHUB_API_CONTENT_URL = join_url('https://api.github.com/repos/', GITHUB_ACCOUNT, REPOSITORY_NAME, 'contents')
    DEFAULT_CACHE_DIRECTORY = os.path.join(user_cache_dir('citylearn'), REPOSITORY_TAG)
    BATTERY_CHOICES_FILENAME = 'battery_choices.yaml'
    PV_CHOICES_FILENAME = 'lbl-tracking_the_sun-res-pv.csv'

    def __init__(self, github_account: str = None, repository: str = None, tag: str = None, datasets_path: str = None, misc_path: str = None, logging_level: int = None):
        self.github_account = github_account
        self.repository = repository
        self.tag = tag
        self.datasets_path = datasets_path
        self.misc_path = misc_path
        self.logging_level = logging_level

    @property
    def github_account(self) -> str:
        return  self.__github_account
    
    @property
    def repository(self) -> str:
        return self.__repository
    
    @property
    def tag(self) -> str:
        return self.__tag
    
    @property
    def datasets_path(self) -> str:
        return self.__datasets_path
    
    @property
    def misc_path(self) -> str:
        return self.__misc_path
    
    @property
    def cache_directory(self) -> Union[Path, str]:
        directory = user_cache_dir(
            appname=self.repository.lower(),
            appauthor=self.github_account,
            version=self.tag,
        )
        os.makedirs(directory, exist_ok=True)
        
        return directory
    
    @property
    def logging_level(self) -> int:
        return self.__logging_level
    
    @github_account.setter
    def github_account(self, value: str):
        self.__github_account = self.GITHUB_ACCOUNT if value is None else value

    @repository.setter
    def repository(self, value: str):
        self.__repository = self.REPOSITORY_NAME if value is None else value

    @tag.setter
    def tag(self, value: str):
        self.__tag = self.REPOSITORY_TAG if value is None else value

    @datasets_path.setter
    def datasets_path(self, value: str):
        self.__datasets_path = self.REPOSITORY_DATA_DATASETS_PATH if value is None else value

    @misc_path.setter
    def misc_path(self, value: str):
        self.__misc_path = self.REPOSITORY_DATA_MISC_PATH if value is None else value

    @logging_level.setter
    def logging_level(self, value: int):
        self.__logging_level = 20 if value is None else value
        LOGGER.setLevel(self.logging_level)

    def get_schema(self, name: str) -> dict:
        schema_filepath = self.get_dataset(name)
        schema = read_json(schema_filepath)
        schema['root_directory'] = os.path.split(Path(schema_filepath).absolute())[0]

        return schema

    def get_dataset(self, name: str, directory: Union[Path, str] = None) -> str:
        datasets_directory = os.path.join(self.cache_directory, 'datasets')
        root_directory = os.path.join(datasets_directory, name)
        schema_filepath = os.path.join(root_directory, 'schema.json')
        path = join_url(self.datasets_path, name)

        # check that dataset does not already exist using the schema as a proxy
        if not os.path.isfile(schema_filepath):
            LOGGER.info(f'The {name} dataset DNE in cache. Will download from '
                f'{self.github_account}/{self.repository}/tree/{self.tag} GitHub repository and write to {datasets_directory}. '
                    f'Next time DataSet.get_dataset(\'{name}\') is called, it will read '
                        'from cache unless DataSet.clear_cache is run first.')
            contents = self.get_github_contents(path)

            if os.path.isdir(root_directory):
                shutil.rmtree(root_directory)
            
            else:
                pass

            for c in contents:
                if c['type'] == 'file':
                    relative_directory_content = c['path'].split(f'{name}/')[-1].split('/')[:-1]
                    content_directory = os.path.join(root_directory, *relative_directory_content)
                    filepath = os.path.join(content_directory, c['name'])
                    os.makedirs(content_directory, exist_ok=True)
                    response = self.get_requests_session().get(c['download_url'])

                    with open(filepath, 'wb') as f:
                        f.write(response.content)

                else:
                    pass

        else:
            pass

        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            shutil.copytree(root_directory, directory, dirs_exist_ok=True)
            schema_filepath = os.path.join(directory, name, 'schema.json')
        
        else:
            pass
    
        return schema_filepath

    def get_dataset_names(self) -> List[str]:
        contents = self.get_github_contents(self.datasets_path)
        filepath = os.path.join(self.cache_directory, 'dataset_names.json')

        if os.path.isfile(filepath):
            contents = read_json(filepath)
        
        else:
            LOGGER.info(f'The dataset names DNE in cache. Will download from '
                f'{self.github_account}/{self.repository}/tree/{self.tag} GitHub repository and write to {filepath}. '
                    'Next time DataSet.get_dataset_names is called, it will read '
                        'from cache unless DataSet.clear_cache is run first.')
            contents = [
                r['name'] for r in contents 
                    if r.get('type') == 'dir' 
                        and r.get('path').replace(r['name'], '').strip('/') == self.datasets_path
            ]
            write_json(filepath, contents)
            
        contents = sorted(contents)

        return contents
    
    def get_pv_sizing_data(self) -> pd.DataFrame:
        """Reads and returns NREL's Tracking The Sun dataset that has been prefilered for completeness.
        
        Returns
        -------
        data: pd.DataFrame
        """

        misc_directory = os.path.join(self.cache_directory, 'misc')
        os.makedirs(misc_directory, exist_ok=True)
        filepath = os.path.join(misc_directory, self.PV_CHOICES_FILENAME)
        path = join_url(self.misc_path)

        # check that file DNE
        if not os.path.isfile(filepath):
            LOGGER.info(f'The PV sizing data DNE in cache. Will download from '
                f'{self.github_account}/{self.repository}/tree/{self.tag} GitHub repository and write to {misc_directory}. '
                    'Next time DataSet.get_pv_sizing_data is called, it will read '
                        'from cache unless DataSet.clear_cache is run first.')
            contents = self.get_github_contents(path)
            url = [f['download_url'] for f in contents if f['name'] == self.PV_CHOICES_FILENAME][0]
            response = self.get_requests_session().get(url)

            with open(filepath, 'wb') as f:
                f.write(response.content)

        else:
            pass

        data = pd.read_csv(filepath, low_memory=False)
        
        return data
    
    def get_battery_sizing_data(self) -> Mapping[str, Union[float, str]]:
        """Reads and returns internally defined real world manufacturer models.
        
        Returns
        -------
        data: Mapping[str, Union[float, str]]
        """

        misc_directory = os.path.join(self.cache_directory, 'misc')
        os.makedirs(misc_directory, exist_ok=True)
        filepath = os.path.join(misc_directory, self.BATTERY_CHOICES_FILENAME)
        path = join_url(self.misc_path)

        # check that file DNE
        if not os.path.isfile(filepath):
            LOGGER.info(f'The battery sizing data DNE in cache. Will download from '
                f'{self.github_account}/{self.repository}/tree/{self.tag} GitHub repository and write to {misc_directory}. '
                    'Next time DataSet.get_battery_sizing_data is called, it will read '
                        'from cache unless DataSet.clear_cache is run first.')
            contents = self.get_github_contents(path)
            url = [f['download_url'] for f in contents if f['name'] == self.BATTERY_CHOICES_FILENAME][0]
            response = self.get_requests_session().get(url)

            with open(filepath, 'wb') as f:
                f.write(response.content)

        else:
            pass

        data = read_yaml(filepath)
        data = pd.DataFrame([{'model': k, **v['attributes']} for k, v in data.items()])
        data = data.set_index('model')

        return data
    
    def clear_cache(self):
        if os.path.isdir(self.cache_directory):
            shutil.rmtree(self.cache_directory)
        
        else:
            pass

    def get_github_contents(self, path: str = None) -> List[Mapping[str, Any]]:
        url = self.GITHUB_API_CONTENT_URL if path is None else join_url(self.GITHUB_API_CONTENT_URL, path) 
        params = dict(ref=self.tag)
        contents = self.get_requests_session().get(url, params=params)

        if contents.status_code == 200:
            contents = contents.json()

        else:
            raise Exception(f'Unable to get response from GitHub API for endpoint: {url}.'\
                f'\rReturned status code: {contents.status_code};\rContent: {contents.content}')

        return contents
    
    @staticmethod
    def get_requests_session(**kwargs) -> requests.Session:
        session = requests.Session()
        kwargs = {
            'total': 5,
            'backoff_factor': 1,
            'status_forcelist': [400, 502, 503, 504],
            **kwargs
        }
        retries = Retry(**kwargs)
        session.mount('http://', HTTPAdapter(max_retries=retries))
        
        return session
    
class TimeSeriesData:
    """Generic time series data class.
    
    
    Parameters
    ----------
    variable: np.array, optional
        A generic time series variable.
    start_time_step: int, optional
        Time step to start reading variables.
    end_time_step: int, optional
         Time step to end reading variables.
    """

    def __init__(self, variable: Iterable = None, start_time_step: int = None, end_time_step: int = None):
        self.variable = variable if variable is None else np.array(variable)
        self.start_time_step = start_time_step
        self.end_time_step = end_time_step

    def __getattr__(self, name: str, start_time_step: int = None, end_time_step: int = None):
        """Returns values of the named variable within the specified time steps and
        is useful for selecting episode-specific observation."""
        
        # not the most elegant solution tbh
        try:
            variable = self.__dict__[f'_{name}']
        except KeyError:
            raise AttributeError(f'_{name}')
        
        if isinstance(variable, Iterable):
            start_time_step = self.start_time_step if start_time_step is None else start_time_step
            start_index = 0 if start_time_step is None else start_time_step
            end_time_step = self.end_time_step if end_time_step is None else end_time_step
            end_index = len(variable) if end_time_step is None else end_time_step + 1
            return variable[start_index:end_index]
        
        else:
            return variable
        
    def __setattr__(self, name: str, value: Any):
        """Sets named variable.
        
        Variables are named with a single underscore prefix.
        """

        self.__dict__[f'_{name}'] = value

class EnergySimulation(TimeSeriesData):
    """`Building` `energy_simulation` data class.

    Parameters
    ----------
    month : np.array
        Month time series value ranging from 1 - 12.
    hour : np.array
        Hour time series value ranging from 1 - 24.
    day_type : np.array
        Numeric day of week time series ranging from 1 - 8 where 1 - 7 is Monday - Sunday and 8 is reserved for special days e.g. holiday.
    indoor_dry_bulb_temperature : np.array
        Average building dry bulb temperature time series in [C].
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
    daylight_savings_status : np.array, optional
        Daylight saving status time series signal of 0 or 1 indicating inactive  or active daylight saving respectively.
    average_unmet_cooling_setpoint_difference : np.array, optional
        Average difference between `indoor_dry_bulb_temperature` and cooling temperature setpoints time series in [C].
    indoor_relative_humidity : np.array, optional
        Average building relative humidity time series in [%].
    occupant_count: np.array, optional
        Building occupant count time series in [people].
    indoor_dry_bulb_temperature_cooling_set_point: np.array
        Average building dry bulb temperature cooling set point time series in [C].
    indoor_dry_bulb_temperature_heating_set_point: np.array
        Average building dry bulb temperature heating set point time series in [C].
    hvac_mode: np.array, default: 1
        Cooling and heating device availability. If 0, both HVAC devices are unavailable (off), if 1,
        the cooling device is available for space cooling and if 2, the heating device is available
        for space heating only. Automatic (auto) mode is 3 and allows for either cooling or heating 
        depending on the control action. The default is to set the mode to cooling at all times. 
        The HVAC devices are always available for cooling and heating storage charging irrespective 
        of the hvac mode.
    power_outage np.array, default: 0
        Signal for power outage. If 0, there is no outage and building can draw energy from grid. 
        If 1, there is a power outage and building can only use its energy resources to meet loads.
    comfort_band np.array, default: 2
        Occupant comfort band above the `indoor_dry_bulb_temperature_cooling_set_point` and below the `indoor_dry_bulb_temperature_heating_set_point` [C]. The value is added
        to and subtracted from the set point to set the upper and lower bounds of comfort bound.
    start_time_step: int, optional
        Time step to start reading variables.
    end_time_step: int, optional
        Time step to end reading variables.
    """

    DEFUALT_COMFORT_BAND = 2.0

    def __init__(
        self, month: Iterable[int], hour: Iterable[int], day_type: Iterable[int],
         indoor_dry_bulb_temperature: Iterable[float], 
        non_shiftable_load: Iterable[float], dhw_demand: Iterable[float], cooling_demand: Iterable[float], heating_demand: Iterable[float], solar_generation: Iterable[float], 
        daylight_savings_status: Iterable[int] = None, average_unmet_cooling_setpoint_difference: Iterable[float] = None, indoor_relative_humidity: Iterable[float] = None, occupant_count: Iterable[int] = None, indoor_dry_bulb_temperature_cooling_set_point: Iterable[int] = None, indoor_dry_bulb_temperature_heating_set_point: Iterable[int] = None, hvac_mode: Iterable[int] = None, power_outage: Iterable[int] = None, comfort_band: Iterable[float] = None, start_time_step: int = None, end_time_step: int = None
    ):
        super().__init__(start_time_step=start_time_step, end_time_step=end_time_step)
        self.month = np.array(month, dtype='int32')
        self.hour = np.array(hour, dtype='int32')
        self.day_type = np.array(day_type, dtype='int32')
        self.indoor_dry_bulb_temperature = np.array(indoor_dry_bulb_temperature, dtype='float32')
        self.non_shiftable_load = np.array(non_shiftable_load, dtype = 'float32')
        self.dhw_demand = np.array(dhw_demand, dtype = 'float32')
        
        # set space demands and check there is not cooling and heating demand at same time step
        self.cooling_demand = np.array(cooling_demand, dtype = 'float32')
        self.heating_demand = np.array(heating_demand, dtype = 'float32')
        assert (self.cooling_demand*self.heating_demand).sum() == 0, 'Cooling and heating in the same time step is not allowed.'

        self.solar_generation = np.array(solar_generation, dtype = 'float32')

        # optional
        self.daylight_savings_status = np.zeros(len(solar_generation), dtype='int32') if daylight_savings_status is None else np.array(daylight_savings_status, dtype='int32')
        self.average_unmet_cooling_setpoint_difference = np.zeros(len(solar_generation), dtype='float32') if average_unmet_cooling_setpoint_difference is None else np.array(average_unmet_cooling_setpoint_difference, dtype='float32')
        self.indoor_relative_humidity = np.zeros(len(solar_generation), dtype='float32') if indoor_relative_humidity is None else np.array(indoor_relative_humidity, dtype = 'float32')
        self.occupant_count = np.zeros(len(solar_generation), dtype='float32') if occupant_count is None else np.array(occupant_count, dtype='float32')
        self.indoor_dry_bulb_temperature_cooling_set_point = np.zeros(len(solar_generation), dtype='float32') if indoor_dry_bulb_temperature_cooling_set_point is None else np.array(indoor_dry_bulb_temperature_cooling_set_point, dtype='float32')
        self.indoor_dry_bulb_temperature_heating_set_point = np.zeros(len(solar_generation), dtype='float32') if indoor_dry_bulb_temperature_heating_set_point is None else np.array(indoor_dry_bulb_temperature_heating_set_point, dtype='float32')
        self.power_outage = np.zeros(len(solar_generation), dtype='float32') if power_outage is None else np.array(power_outage, dtype='float32')
        self.comfort_band = np.zeros(len(solar_generation), dtype='float32') + self.DEFUALT_COMFORT_BAND if comfort_band is None else np.array(comfort_band, dtype='float32')

        # set controlled variable defaults
        self.indoor_dry_bulb_temperature_without_control = self.indoor_dry_bulb_temperature.copy()
        self.cooling_demand_without_control = self.cooling_demand.copy()
        self.heating_demand_without_control = self.heating_demand.copy()
        self.dhw_demand_without_control = self.dhw_demand.copy()
        self.non_shiftable_load_without_control = self.non_shiftable_load.copy()
        self.indoor_relative_humidity_without_control = self.indoor_relative_humidity.copy()
        self.indoor_dry_bulb_temperature_cooling_set_point_without_control = self.indoor_dry_bulb_temperature_cooling_set_point.copy()
        self.indoor_dry_bulb_temperature_heating_set_point_without_control = self.indoor_dry_bulb_temperature_heating_set_point.copy()

        if hvac_mode is None:
            hvac_mode = np.zeros(len(solar_generation), dtype='int32') + 1 
        
        else:
            unique = list(set(hvac_mode))

            for i in range(4):
                try:
                    unique.remove(i)
                except ValueError:
                    pass

            assert len(unique) == 0, f'Invalid hvac_mode values were found: {unique}. '\
                'Valid values are 0, 1, 2, 3 to indicate off, cooling mode, heating mode, and automatic mode.'
            
        self.hvac_mode = np.array(hvac_mode, dtype='int32')
    
class LogisticRegressionOccupantParameters(TimeSeriesData):
    def __init__(self, a_increase: Iterable[float], b_increase: Iterable[float], a_decrease: Iterable[float], b_decrease: Iterable[float], start_time_step: int = None, end_time_step: int = None):
        super().__init__(start_time_step=start_time_step, end_time_step=end_time_step)
        self.a_increase = np.array(a_increase, dtype='float32')
        self.b_increase = np.array(b_increase, dtype='float32')
        self.a_decrease = np.array(a_decrease, dtype='float32')
        self.b_decrease = np.array(b_decrease, dtype='float32')
        self.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta = np.zeros(len(self.a_increase), dtype='float32')
        self.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_without_control = np.zeros(len(self.a_increase), dtype='float32')

class Weather(TimeSeriesData):
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
    outdoor_dry_bulb_temperature_predicted_1 : np.array
        Outdoor dry bulb temperature `n` hours ahead prediction time series in [C]. `n` can be any number of hours and is typically 6 hours in existing datasets.
    outdoor_dry_bulb_temperature_predicted_2 : np.array
        Outdoor dry bulb temperature `n` hours ahead prediction time series in [C]. `n` can be any number of hours and is typically 12 hours in existing datasets.
    outdoor_dry_bulb_temperature_predicted_3 : np.array
        Outdoor dry bulb temperature `n` hours ahead prediction time series in [C]. `n` can be any number of hours and is typically 24 hours in existing datasets.
    outdoor_relative_humidity_predicted_1 : np.array
        Outdoor relative humidity `n` hours ahead prediction time series in [%]. `n` can be any number of hours and is typically 6 hours in existing datasets.
    outdoor_relative_humidity_predicted_2 : np.array
        Outdoor relative humidity `n` hours ahead prediction time series in [%]. `n` can be any number of hours and is typically 12 hours in existing datasets.
    outdoor_relative_humidity_predicted_3 : np.array
        Outdoor relative humidity `n` hours ahead prediction time series in [%]. `n` can be any number of hours and is typically 24 hours in existing datasets.
    diffuse_solar_irradiance_predicted_1 : np.array
        Diffuse solar irradiance `n` hours ahead prediction time series in [W/m^2]. `n` can be any number of hours and is typically 6 hours in existing datasets.
    diffuse_solar_irradiance_predicted_2 : np.array
        Diffuse solar irradiance `n` hours ahead prediction time series in [W/m^2]. `n` can be any number of hours and is typically 12 hours in existing datasets.
    diffuse_solar_irradiance_predicted_3 : np.array
        Diffuse solar irradiance `n` hours ahead prediction time series in [W/m^2]. `n` can be any number of hours and is typically 24 hours in existing datasets.
    direct_solar_irradiance_predicted_1 : np.array
        Direct solar irradiance `n` hours ahead prediction time series in [W/m^2]. `n` can be any number of hours and is typically 6 hours in existing datasets.
    direct_solar_irradiance_predicted_2 : np.array
        Direct solar irradiance `n` hours ahead prediction time series in [W/m^2]. `n` can be any number of hours and is typically 12 hours in existing datasets.
    direct_solar_irradiance_predicted_3 : np.array
        Direct solar irradiance `n` hours ahead prediction time series in [W/m^2]. `n` can be any number of hours and is typically 24 hours in existing datasets.
    start_time_step: int, optional
        Time step to start reading variables.
    end_time_step: int, optional
         Time step to end reading variables.
    """

    def __init__(
        self, outdoor_dry_bulb_temperature: Iterable[float], outdoor_relative_humidity: Iterable[float], diffuse_solar_irradiance: Iterable[float], direct_solar_irradiance: Iterable[float], 
        outdoor_dry_bulb_temperature_predicted_1: Iterable[float], outdoor_dry_bulb_temperature_predicted_2: Iterable[float], outdoor_dry_bulb_temperature_predicted_3: Iterable[float],
        outdoor_relative_humidity_predicted_1: Iterable[float], outdoor_relative_humidity_predicted_2: Iterable[float], outdoor_relative_humidity_predicted_3: Iterable[float],
        diffuse_solar_irradiance_predicted_1: Iterable[float], diffuse_solar_irradiance_predicted_2: Iterable[float], diffuse_solar_irradiance_predicted_3: Iterable[float],
        direct_solar_irradiance_predicted_1: Iterable[float], direct_solar_irradiance_predicted_2: Iterable[float], direct_solar_irradiance_predicted_3: Iterable[float], start_time_step: int = None, end_time_step: int = None
    ):
        super().__init__(start_time_step=start_time_step, end_time_step=end_time_step)
        self.outdoor_dry_bulb_temperature = np.array(outdoor_dry_bulb_temperature, dtype='float32')
        self.outdoor_relative_humidity = np.array(outdoor_relative_humidity, dtype='float32')
        self.diffuse_solar_irradiance = np.array(diffuse_solar_irradiance, dtype='float32')
        self.direct_solar_irradiance = np.array(direct_solar_irradiance, dtype='float32')
        self.outdoor_dry_bulb_temperature_predicted_1 = np.array(outdoor_dry_bulb_temperature_predicted_1, dtype='float32')
        self.outdoor_dry_bulb_temperature_predicted_2 = np.array(outdoor_dry_bulb_temperature_predicted_2, dtype='float32')
        self.outdoor_dry_bulb_temperature_predicted_3 = np.array(outdoor_dry_bulb_temperature_predicted_3, dtype='float32')
        self.outdoor_relative_humidity_predicted_1 = np.array(outdoor_relative_humidity_predicted_1, dtype='float32')
        self.outdoor_relative_humidity_predicted_2 = np.array(outdoor_relative_humidity_predicted_2, dtype='float32')
        self.outdoor_relative_humidity_predicted_3 = np.array(outdoor_relative_humidity_predicted_3, dtype='float32')
        self.diffuse_solar_irradiance_predicted_1 = np.array(diffuse_solar_irradiance_predicted_1, dtype='float32')
        self.diffuse_solar_irradiance_predicted_2 = np.array(diffuse_solar_irradiance_predicted_2, dtype='float32')
        self.diffuse_solar_irradiance_predicted_3 = np.array(diffuse_solar_irradiance_predicted_3, dtype='float32')
        self.direct_solar_irradiance_predicted_1 = np.array(direct_solar_irradiance_predicted_1, dtype='float32')
        self.direct_solar_irradiance_predicted_2 = np.array(direct_solar_irradiance_predicted_2, dtype='float32')
        self.direct_solar_irradiance_predicted_3 = np.array(direct_solar_irradiance_predicted_3, dtype='float32')

class Pricing(TimeSeriesData):
    """`Building` `pricing` data class.

    Parameters
    ----------
    electricity_pricing : np.array
        Electricity pricing time series in [$/kWh].
    electricity_pricing_predicted_1 : np.array
        Electricity pricing `n` hours ahead prediction time series in [$/kWh]. `n` can be any number of hours and is typically 1 or 6 hours in existing datasets.
    electricity_pricing_predicted_2 : np.array
        Electricity pricing `n` hours ahead prediction time series in [$/kWh]. `n` can be any number of hours and is typically 2 or 12 hours in existing datasets.
    electricity_pricing_predicted_3 : np.array
        Electricity pricing `n` hours ahead prediction time series in [$/kWh]. `n` can be any number of hours and is typically 3 or 24 hours in existing datasets.
    start_time_step: int, optional
        Time step to start reading variables.
    end_time_step: int, optional
         Time step to end reading variables.
    """

    def __init__(
        self, electricity_pricing: Iterable[float], electricity_pricing_predicted_1: Iterable[float], electricity_pricing_predicted_2: Iterable[float], 
        electricity_pricing_predicted_3: Iterable[float], start_time_step: int = None, end_time_step: int = None
    ):
        super().__init__(start_time_step=start_time_step, end_time_step=end_time_step)
        self.electricity_pricing = np.array(electricity_pricing, dtype='float32')
        self.electricity_pricing_predicted_1 = np.array(electricity_pricing_predicted_1, dtype='float32')
        self.electricity_pricing_predicted_2 = np.array(electricity_pricing_predicted_2, dtype='float32')
        self.electricity_pricing_predicted_3 = np.array(electricity_pricing_predicted_3, dtype='float32')

class CarbonIntensity(TimeSeriesData):
    """`Building` `carbon_intensity` data class.

    Parameters
    ----------
    carbon_intensity : np.array
        Grid carbon emission rate time series in [kg_co2/kWh].
    start_time_step: int, optional
        Time step to start reading variables.
    end_time_step: int, optional
         Time step to end reading variables.
    """

    def __init__(self, carbon_intensity: Iterable[float], start_time_step: int = None, end_time_step: int = None):
        super().__init__(start_time_step=start_time_step, end_time_step=end_time_step)
        self.carbon_intensity = np.array(carbon_intensity, dtype='float32')


class ElectricVehicleSimulation(TimeSeriesData):
    """`Electric_Vehicle` `electric_vehicle_simulation` data class.

    Month,Hour,Day Type,Location,Estimated Departure Time,Required Soc At Departure

    Attributes
    ----------
    electric_vehicle_charger_state : np.array
        State of the electric_vehicle indicating whether it is 'parked ready to charge' represented as 0, 'in transit', represented as 1.
    charger : np.array
        (available only for 'in transit' state) Charger where the electric_vehicle will plug in the next "parked ready to charge" state.
        It can be nan if no destination charger is specified or the charger id in the format "Charger_X_Y", where X is
        the number of the building and Y the number of the charger within that building.
    electric_vehicle_departure_time : np.array
        Number of time steps  expected until the vehicle departs (available only for 'parked ready to charge' state)
    electric_vehicle_required_soc_departure : np.array
        Estimated SOC percentage required for the electric_vehicle at departure time. (available only for 'parked ready to charge' state)
    electric_vehicle_estimated_arrival_time : np.array
        Number of time steps  expected until the vehicle arrives at the charger (available only for 'in transit' state)
    electric_vehicle_estimated_soc_arrival : np.array
        Estimated SOC percentage for the electric_vehicle at arrival time. (available only for 'in transit' state)

    """

    def __init__(
            self, state: Iterable[str],
            charger: Iterable[str], estimated_departure_time: Iterable[int], required_soc_departure: Iterable[float],
            estimated_arrival_time: Iterable[int], estimated_soc_arrival: Iterable[float], start_time_step: int = None, end_time_step: int = None
    ):
        r"""Initialize `ElectricVehicleSimulation`."""
        super().__init__(start_time_step=start_time_step, end_time_step=end_time_step)
        self.electric_vehicle_charger_state = np.array(state, dtype=int)
        self.charger = np.array(charger, dtype=str)
        # NaNs are considered and filled as -1, i.e., when they serve no value or no data is recorded from them
        default_value = -1
        self.electric_vehicle_departure_time = np.nan_to_num(np.array(estimated_departure_time, dtype=float),
                                                         nan=default_value).astype(int)
        self.electric_vehicle_required_soc_departure = np.nan_to_num(np.array(required_soc_departure, dtype=float), nan=default_value)
        self.electric_vehicle_estimated_arrival_time = np.nan_to_num(np.array(estimated_arrival_time, dtype=float),
                                                       nan=default_value).astype(int)
        self.electric_vehicle_estimated_soc_arrival = np.nan_to_num(np.array(estimated_soc_arrival, dtype=float), nan=default_value)

