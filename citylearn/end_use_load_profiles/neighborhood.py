from copy import deepcopy
import os
from enum import Enum, unique
from pathlib import Path
import random
import shutil
from typing import Any, List, Mapping, Tuple, Union
from doe_xstock.data import VersionDatasetType
from doe_xstock.end_use_load_profiles import EndUseLoadProfiles
from doe_xstock.simulate import EndUseLoadProfilesEnergyPlusSimulator
import numpy as np
import pandas as pd
from citylearn.base import Environment
from citylearn.end_use_load_profiles.clustering import MetadataClustering
from citylearn.end_use_load_profiles.simulate import EnergyPlusPartialLoadSimulator

@unique
class SampleMethod(Enum):
    RANDOM = 0
    METADATA_CLUSTER_FREQUENCY = 1

class Neighborhood:
    __BUILDING_TYPE_COLUMN = 'in.geometry_building_type_recs'
    __SINGLE_FAMILY_BUILDING_TYPE_NAME = 'Single-Family Detached'
    __COUNTY_COLUMN = 'in.resstock_county_id'

    def __init__(
        self, weather_data: str = None, year_of_publication: int = None, release: int = None, cache: bool = None,
        energyplus_output_directory: Union[Path, str] = None, random_seed: int = None
    ) -> None:
        self.__end_use_load_profiles = EndUseLoadProfiles(
            dataset_type=VersionDatasetType.RESSTOCK,
            weather_data=weather_data,
            year_of_publication=year_of_publication,
            release=release,
            cache=cache,
        )
        self.energyplus_output_directory = energyplus_output_directory
        self.random_seed = random_seed

    @property
    def end_use_load_profiles(self) -> EndUseLoadProfiles:
        return self.__end_use_load_profiles
    
    @property
    def energyplus_output_directory(self) -> Union[Path, str]:
        return self.__energyplus_output_directory
    
    @property
    def random_seed(self) -> int:
        return self.__random_seed

    @energyplus_output_directory.setter
    def energyplus_output_directory(self, value: Union[Path, str]):
        self.__energyplus_output_directory = 'energyplus_output' if value is None else value

    @random_seed.setter
    def random_seed(self, value: int):
        self.__random_seed = random.randint(*Environment.DEFAULT_RANDOM_SEED_RANGE) if value is None else value

    def build(
        self, idd_filepath: Union[Path, str], bldg_ids: List[int] = None, sample_buildings_kwargs: Mapping[str, Any] = None, 
        energyplus_simulation_kwargs: Mapping[str, Any] = None, train_lstm_kwargs: Mapping[str, Any] = None, schema_kwargs: Mapping[str, Any] = None
    ):
        sample_buildings_kwargs = {} if sample_buildings_kwargs is None else sample_buildings_kwargs
        energyplus_simulation_kwargs = {} if energyplus_simulation_kwargs is None else energyplus_simulation_kwargs
        train_lstm_kwargs = {} if train_lstm_kwargs is None else train_lstm_kwargs
        schema_kwargs = {} if schema_kwargs is None else schema_kwargs

        if bldg_ids is None:
            bldg_ids = self.sample_buildings(**sample_buildings_kwargs)
        
        else:
            pass
        
        simulators = self.simulate_energy_plus(idd_filepath, **energyplus_simulation_kwargs)
        lstm_training_data = self.get_lstm_training_data(simulators)

    def get_schema(self, simulators: Mapping[int, Mapping[str, Tuple[EndUseLoadProfilesEnergyPlusSimulator, EndUseLoadProfilesEnergyPlusSimulator, EnergyPlusPartialLoadSimulator]]], models: Mapping[str, Path]):
        raise NotImplementedError
    
    def train_lstm(data: Mapping[int, pd.DataFrame], **kwargs) -> Mapping[int, Mapping[str, Any]]:
        """
        TODO: Satvik & Pavani
        1. Install training repo using pip.
        2. Parse training data and custom kwargs for training to some function in the training package
           that trains and finds a best model for the building
        3. Train the building LSTM and return .pth, normalization limits, & error metrics
        """

        raise NotImplementedError
    
    def get_lstm_training_data(self, simulators: Mapping[int, Mapping[int, Tuple[EndUseLoadProfilesEnergyPlusSimulator, EndUseLoadProfilesEnergyPlusSimulator, EnergyPlusPartialLoadSimulator]]]) -> Mapping[int, pd.DataFrame]:
        data = {}

        for bldg_id, building_simulators in simulators.items():
            data_list = []

            for partial_simulator in building_simulators['partial']:
                query_filepath = os.path.join(EnergyPlusPartialLoadSimulator.QUERIES_DIRECTORY, 'select_lstm_training_data.sql')
                pdata = partial_simulator.get_output_database().query_table_from_file(query_filepath)
                pdata.insert(0, 'reference_name', partial_simulator.simulation_id.split('-')[-2])
                pdata.insert(0, 'reference', int(partial_simulator.simulation_id.split('-')[-1]))
                pdata.insert(0, 'bldg_id', bldg_id)
                data_list.append(pdata)
             
            data[bldg_id] = pd.concat(data_list, ignore_index=True)

        return data
    
    def simulate_energy_plus(
        self, bldg_ids: List[int], idd_filepath: Union[Path, str], simulation_ids: List[str] = None, models: List[Union[Path, str]] = None, 
        schedules: Union[Path, pd.DataFrame] = None, osm: bool = None,
        partial_loads_simulations: int = None, partial_loads_kwargs: Mapping[str, Any] = None, max_workers: int = None, **kwargs
    ) -> Mapping[int, Mapping[str, Tuple[EndUseLoadProfilesEnergyPlusSimulator, EndUseLoadProfilesEnergyPlusSimulator, EnergyPlusPartialLoadSimulator]]]:
        assert models is None or len(models) == len(bldg_ids), 'There must be as many models as bldg_ids.'
        assert schedules is None or len(schedules) == len(bldg_ids), 'There must be as many schedules as bldg_ids.'
        assert simulation_ids is None or len(simulation_ids) == len(bldg_ids), 'There must be as many simulation_ids as bldg_ids.'
        os.makedirs(self.energyplus_output_directory, exist_ok=True)
        partial_loads_simulations = 4 if partial_loads_simulations is None else partial_loads_simulations
        partial_loads_kwargs = {} if partial_loads_kwargs is None else partial_loads_kwargs
        simulators = {}

        kwargs = dict(
            idd_filepath=idd_filepath,
            osm=osm,
            **kwargs,
            number_of_time_steps_per_hour=kwargs.pop('number_of_time_steps_per_hour', 1)
        )

        if (
            kwargs.get('default_output_variables', None) is None or not kwargs['default_output_variables']
        ) and kwargs.get('output_variables', None) is None:
            kwargs['default_output_variables'] = True

        else:
            pass

        for i, bldg_id in enumerate(bldg_ids):
            simulation_id = f'{self.end_use_load_profiles.version}-{bldg_id}' if simulation_ids is None else simulation_ids[i]
            output_directory = os.path.join(self.energyplus_output_directory, simulation_id)

            if os.path.isdir(output_directory):
                shutil.rmtree(output_directory)
            
            else:
                pass
            
            os.makedirs(output_directory)

            _kwargs = dict(
                **kwargs,
                bldg_id=bldg_id,
                model=models[i] if models is not None else models,
                schedules=schedules[i] if schedules is not None else schedules,
            )

            # mechanical loads
            _simulation_id = f'{simulation_id}-mechanical'
            _output_directory = os.path.join(output_directory, _simulation_id)
            mechanical_simulator = self.end_use_load_profiles.simulate_building(
                ideal_loads=False,
                simulation_id=_simulation_id,
                output_directory=_output_directory,
                **_kwargs
            ).simulator

            # ideal loads
            _simulation_id = f'{simulation_id}-ideal'
            _output_directory = os.path.join(output_directory, _simulation_id)
            ideal_simulator = self.end_use_load_profiles.simulate_building(
                ideal_loads=True,
                simulation_id=_simulation_id,
                output_directory=_output_directory,
                **_kwargs
            ).simulator

            # partial loads
            partial_loads_simulators: List[EnergyPlusPartialLoadSimulator] = []
            kwargs_multiplier_minimum = partial_loads_kwargs.pop('multiplier_minimum', None)
            kwargs_multiplier_maximum = partial_loads_kwargs.pop('multiplier_maximum', None)
            kwargs_multiplier_probability = partial_loads_kwargs.pop('multiplier_probability', None)

            for i in range(partial_loads_simulations + 2):
                multiplier_minimum = kwargs_multiplier_minimum
                multiplier_maximum = kwargs_multiplier_maximum
                multiplier_probability = kwargs_multiplier_probability

                if i == 0:
                    reference_name = 'ideal'
                    multiplier_probability = 0
                
                elif i == 1:
                    reference_name = 'free_float'
                    multiplier_minimum = 0
                    multiplier_maximum = 0

                else:
                    reference_name = 'undercool_overcool_underheat_overheat'

                _simulation_id = f'{simulation_id}-partial-{reference_name}-{i}'
                _output_directory = os.path.join(output_directory, _simulation_id)
                _ = partial_loads_kwargs.pop('simulation_id', None)
                _ = partial_loads_kwargs.pop('output_directory', None)
                _ = partial_loads_kwargs.pop('random_seed', None)
                partial_loads_simulators.append(EnergyPlusPartialLoadSimulator(
                    ideal_loads_simulator=deepcopy(ideal_simulator),
                    multiplier_minimum=multiplier_minimum,
                    multiplier_maximum=multiplier_maximum,
                    multiplier_probability=multiplier_probability,
                    simulation_id=_simulation_id,
                    output_directory=_output_directory,
                    random_seed=int(bldg_id) + i,
                    **partial_loads_kwargs
                ))
            
            EnergyPlusPartialLoadSimulator.multi_simulate(partial_loads_simulators, max_workers=max_workers)

            simulators[bldg_id] = {
                'mechanical': mechanical_simulator,
                'ideal': ideal_simulator,
                'partial': partial_loads_simulators
            }
        
        return simulators

    def sample_buildings(
        self, filters: Mapping[str, List[Any]] = None, sample_method: SampleMethod = None, sample_count: int = None, duplicate_to_count: bool = None, 
        single_county: bool = None, single_family_detached: bool = None, **kwargs
    ):
        sample_method = SampleMethod.RANDOM if sample_method is None else sample_method
        valid_sample_method = [v for v in SampleMethod]
        assert sample_method in valid_sample_method, f'Valid sample methods are {valid_sample_method}.'
        sample_count = 100 if sample_count is None else sample_count
        duplicate_to_count = True if duplicate_to_count is None else duplicate_to_count
        single_county = True if single_county is None else single_county
        single_family_detached = True if single_family_detached is None else single_family_detached
        
        nprs = np.random.RandomState(self.random_seed)
        sample_metadata = None

        if single_family_detached:
            filters = {} if filters is None else filters
            filters[self.__BUILDING_TYPE_COLUMN] =  [self.__SINGLE_FAMILY_BUILDING_TYPE_NAME]
        
        else:
            pass
        
        metadata = self.end_use_load_profiles.metadata.metadata.get(filters)

        if single_county:
            county = nprs.choice(metadata[self.__COUNTY_COLUMN].unique(), 1)[0]
            metadata = metadata[metadata[self.__COUNTY_COLUMN]==county].copy()

        else:
            pass

        if sample_method == SampleMethod.RANDOM:
            metadata['label'] = 0

        elif sample_method == SampleMethod.METADATA_CLUSTER_FREQUENCY:
            mc = MetadataClustering(self.end_use_load_profiles, metadata.index.tolist(), random_seed=self.random_seed, **kwargs)
            optimal_clusters, scores, labels = mc.cluster()
            sample_metadata = {
                'optimal_clusters': optimal_clusters,
                'scores': scores,
                'labels': labels
            }

            metadata = metadata.merge(labels[labels['clusters']==optimal_clusters].set_index('bldg_id'), left_index=True, right_index=True)

        metadata['label_count'] = metadata.groupby('label')['label'].transform('count')
        metadata = metadata.sample(sample_count, weights='label_count', replace=duplicate_to_count, random_state=self.random_seed)
        bldg_ids = metadata.index.tolist()
        labels = metadata['label'].tolist()

        return bldg_ids, labels, sample_metadata