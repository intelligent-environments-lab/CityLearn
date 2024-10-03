import os
from pathlib import Path
from typing import Any, List, Mapping, Union
from doe_xstock.simulate import EndUseLoadProfilesEnergyPlusSimulator
from eppy.modeleditor import IDF
import numpy as np
import pandas as pd

class EndUseLoadProfilesEnergyPlusPartialLoadSimulator(EndUseLoadProfilesEnergyPlusSimulator):
    QUERIES_DIRECTORY = os.path.join(os.path.join(*Path(os.path.dirname(os.path.abspath(__file__))).parts[0:-1]), 'misc', 'queries')

    def __init__(
        self, ideal_loads_simulator: EndUseLoadProfilesEnergyPlusSimulator, allow_multi_load_time_step: bool = None, multiplier_minimum: float = None,
        multiplier_maximum: float = None, multiplier_probability: float = None, output_variables: List[str] = None, output_meters: List[str] = None, 
        simulation_id: str = None, output_directory: Union[Path, str] = None, random_seed: int = None,
    ):
        self.ideal_loads_simulator = ideal_loads_simulator
        super().__init__(
            self.ideal_loads_simulator.version, 
            self.ideal_loads_simulator.idd_filepath, 
            self.ideal_loads_simulator.idf, 
            self.ideal_loads_simulator.epw, 
            schedules_filepath=self.ideal_loads_simulator.schedules_filepath, 
            number_of_time_steps_per_hour=self.ideal_loads_simulator.number_of_time_steps_per_hour, 
            output_variables=output_variables if output_variables is not None else self.ideal_loads_simulator.output_variables, 
            output_meters=output_meters if output_meters is not None else self.ideal_loads_simulator.output_meters, 
            osm=False, 
            ideal_loads=False, 
            edit_ems=False, 
            simulation_id=simulation_id,
            output_directory=output_directory
        )
        self.allow_multi_load_time_step = allow_multi_load_time_step
        self.multiplier_minimum = multiplier_minimum
        self.multiplier_maximum = multiplier_maximum
        self.multiplier_probability = multiplier_probability
        self.random_seed = random_seed
    
    @property
    def ideal_loads_simulator(self) -> EndUseLoadProfilesEnergyPlusSimulator:
        return self.__ideal_loads_simulator
    
    @property
    def allow_multi_load_time_step(self) -> bool:
        return self.__allow_multi_load_time_step
    
    @property
    def multiplier_minimum(self) -> float:
        return self.__multiplier_minimum
    
    @property
    def multiplier_maximum(self) -> float:
        return self.__multiplier_maximum
    
    @property
    def multiplier_probability(self) -> float:
        return self.__multiplier_probability
    
    @property
    def random_seed(self) -> int:
        return self.__random_seed
    
    @ideal_loads_simulator.setter
    def ideal_loads_simulator(self, value: EndUseLoadProfilesEnergyPlusSimulator):
        self.__ideal_loads_simulator = value

    @allow_multi_load_time_step.setter
    def allow_multi_load_time_step(self, value: bool):
        self.__allow_multi_load_time_step = False if value is None else value

    @multiplier_minimum.setter
    def multiplier_minimum(self, value: float):
        self.__multiplier_minimum = 0.3 if value is None else value

    @multiplier_maximum.setter
    def multiplier_maximum(self, value: float):
        self.__multiplier_maximum = 1.7 if value is None else value

    @multiplier_probability.setter
    def multiplier_probability(self, value: float):
        value = 0.6 if value is None else value
        assert 0 <= value <= 1.0, 'multiplier_probability must be > 0 and <= 1.'
        self.__multiplier_probability = value

    @random_seed.setter
    def random_seed(self, value: int):
        self.__random_seed = 0 if value is None else value

    @classmethod
    def multi_simulate(cls, simulators: list, max_workers=None):
        super().multi_simulate(simulators, max_workers=max_workers)

        for s in simulators:
            s.insert_zone_metadata(s)

    def simulate(self, **run_kwargs):
        super().simulate(**run_kwargs)
        self.insert_zone_metadata(self)

    def preprocess_idf_for_simulation(self) -> IDF:
        idf = super().preprocess_idf_for_simulation()
        idf = self.remove_ideal_loads_air_system(idf)
        idf = self.add_other_equipment(idf)

        return idf
    
    def add_other_equipment(self, idf: IDF) -> IDF:
        # schedule type limit object
        schedule_type_limit_name = 'other equipment hvac power'
        obj = idf.newidfobject('ScheduleTypeLimits')
        obj.Name = schedule_type_limit_name
        obj.Lower_Limit_Value = ''
        obj.Upper_Limit_Value = ''
        obj.Numeric_Type = 'Continuous'
        obj.Unit_Type = 'Dimensionless'

        # generate stochastic thermal load
        ideal_loads = self.get_ideal_loads()
        timesteps = ideal_loads['timestep'].max()
        multipliers = self.get_multipliers(ideal_loads.shape[0])
        ideal_loads['cooling_load'] *= -multipliers
        ideal_loads['heating_load'] *= multipliers
        loads_filepath = os.path.join(self.output_directory, f'{self.simulation_id}-partial_load.csv')
        ideal_loads = ideal_loads.sort_values(['zone_name', 'timestep'])
        loads = ['cooling_load', 'heating_load']
        ideal_loads[loads].to_csv(loads_filepath, index=False)
        
        for i, (zone_name, _) in enumerate(ideal_loads.groupby('zone_name')):
            for j, load in enumerate(loads):
                # put schedule obj
                obj = idf.newidfobject('Schedule:File')
                schedule_object_name = f'{zone_name} partial {load}'
                obj.Name = schedule_object_name
                obj.Schedule_Type_Limits_Name = schedule_type_limit_name
                obj.File_Name = loads_filepath
                obj.Column_Number = j + 1
                obj.Rows_to_Skip_at_Top = 1 + i*timesteps
                obj.Number_of_Hours_of_Data = 8760

                if self.number_of_time_steps_per_hour is None:
                    timestep_obj = self.ideal_loads_simulator.get_idf_object().idfobjects['Timestep'][0]
                    number_of_time_steps_per_hour = timestep_obj.Number_of_Timesteps_per_Hour

                else:
                    number_of_time_steps_per_hour = self.number_of_time_steps_per_hour

                obj.Minutes_per_Item = int(60/number_of_time_steps_per_hour)

                # put other equipment
                obj = idf.newidfobject('OtherEquipment')
                obj.Name = f'{zone_name} partial {load}'
                obj.Fuel_Type = 'None'
                obj.Zone_or_ZoneList_or_Space_or_SpaceList_Name = zone_name
                obj.Schedule_Name = schedule_object_name
                obj.Design_Level_Calculation_Method = 'EquipmentLevel'
                obj.Design_Level = 1.0 # already in Watts from query
                obj.Fraction_Latent = 0.0
                obj.Fraction_Radiant = 0.0
                obj.Fraction_Lost = 0.0
                obj.EndUse_Subcategory = f'partial {load}'

        return idf
    
    def get_multipliers(self, size: int) -> np.ndarray:
        nprs = np.random.RandomState(self.random_seed)
        data = nprs.uniform(self.multiplier_minimum, self.multiplier_maximum, size)
        data[nprs.random(size) > self.multiplier_probability] = 1.0
        
        return data
    
    def get_ideal_loads(self):
        self.insert_zone_metadata(self.ideal_loads_simulator)
        zones = self.get_zone_conditioning_metadata()
        cooled_zone_names = [f'\'{k}\'' for k, v in zones.items() if v['is_cooled']==1]
        heated_zone_names = [f'\'{k}\'' for k, v in zones.items() if v['is_heated']==1]
        cooled_zone_names = ', '.join(cooled_zone_names)
        heated_zone_names = ', '.join(heated_zone_names)

        query_filepath = os.path.join(self.QUERIES_DIRECTORY, 'select_ideal_loads.sql')
        data = self.ideal_loads_simulator.get_output_database().query_table_from_file(query_filepath, replace={
            '<cooled_zone_names>': cooled_zone_names, 
            '<heated_zone_names>': heated_zone_names, 
        })
        data = data.pivot(index=['timestep', 'zone_index', 'zone_name'], columns='load', values='value')
        data = data.fillna(0.0)
        data = data.reset_index()

        if not self.allow_multi_load_time_step:
            data.loc[data['cooling_load'] > data['heating_load'], 'heating_load'] = 0.0
            data.loc[data['heating_load'] > data['cooling_load'], 'cooling_load'] = 0.0
        
        else:
            pass

        return data
    
    def insert_zone_metadata(self, simulator: EndUseLoadProfilesEnergyPlusSimulator):
        zones = self.get_zone_conditioning_metadata()
        data = pd.DataFrame([v for _, v in zones.items()])
        query_filepath = os.path.join(self.QUERIES_DIRECTORY, 'create_zone_metadata.sql')
        simulator.get_output_database().execute_sql_from_file(query_filepath)
        simulator.get_output_database().insert('zone_metadata', data.columns.tolist(), data.values)

    def get_zone_conditioning_metadata(self) -> Mapping[str, Mapping[str, Any]]:
        query_filepath = os.path.join(self.QUERIES_DIRECTORY, 'select_zone_conditioning_metadata.sql')
        data = self.ideal_loads_simulator.get_output_database().query_table_from_file(query_filepath)
        zones = {z['zone_name']: z for z in data.to_dict('records')}

        return zones

    def remove_ideal_loads_air_system(self, idf: IDF) -> IDF:
        idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem'] = []
        obj_names = [
            'ZoneControl:Thermostat', 'ZoneControl:Humidistat', 'ZoneControl:Thermostat:ThermalComfort', 'ThermostatSetpoint:DualSetpoint', 
            'ZoneControl:Thermostat:OperativeTemperature', 'ZoneControl:Thermostat:TemperatureAndHumidity', 'ZoneControl:Thermostat:StagedDualSetpoint'
        ]

        for name in obj_names:
            idf.idfobjects[name] = []

        # turn off sizing
        for obj in idf.idfobjects['SimulationControl']:
            obj.Do_Zone_Sizing_Calculation = 'No'
            obj.Do_System_Sizing_Calculation = 'No'
            obj.Do_Plant_Sizing_Calculation = 'No'
            obj.Run_Simulation_for_Sizing_Periods = 'No'

        return idf