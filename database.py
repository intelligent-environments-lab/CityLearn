from datetime import datetime
import math
import os
import pytz
import sqlite3
import numpy as np
import pandas as pd
from utilities import read_json

class SQLiteDatabase:
    def __init__(self,filepath):
        self.filepath = filepath
        self.__register_adapter()
    
    @property
    def filepath(self):
        return self.__filepath
    
    @filepath.setter
    def filepath(self,filepath):
        self.__filepath = filepath

    def __get_connection(self):
        return sqlite3.connect(self.filepath)

    def __validate_query(self,query):
        query = query.replace(',)',')')
        return query

    def get_table(self,table_name):
        query = f"SELECT * FROM {table_name}"
        return self.query_table(self.__validate_query(query))

    def query_table(self,query):
        try:
            connection = self.__get_connection()
            df = pd.read_sql(self.__validate_query(query),connection)
            connection.commit()
        finally:
            connection.close()

        return df

    def get_schema(self):
        try:
            connection = self.__get_connection()
            query = "SELECT * FROM sqlite_master WHERE type IN ('table', 'view')"
            schema = pd.read_sql(self.__validate_query(query),connection)['sql'].tolist()
        finally:
            connection.close()
        
        schema = '\n\n'.join(schema)
        return schema

    def vacuum(self):
        try:
            connection = self.__get_connection()
            connection.execute('VACUUM')
            connection.commit()
        finally:
            connection.close()

    def drop(self,name,is_view=False):    
        try:
            connection = self.__get_connection()
            query = f"DROP {'VIEW' if is_view else 'TABLE'} IF EXISTS {name}"
            connection.execute(self.__validate_query(query))
            connection.commit()
        finally:
            connection.close()

    def execute_sql_from_file(self,filepath):
        with open(filepath,'r') as f:
            queries = f.read()
        
        try:
            connection = self.__get_connection()

            for query in queries.split(';'):
                connection.execute(self.__validate_query(query))
                connection.commit()
        
        finally:
            connection.close()

    def insert_file(self,filepath,table_name,**kwargs):
        df = self.read_file(filepath)
        kwargs['values'] = df.to_records(index=False)
        kwargs['fields'] = kwargs.get('fields',list(df.columns))
        kwargs['table_name'] = table_name
        kwargs['on_conflict_fields'] = kwargs.get('on_conflict_fields',None)
        kwargs['ignore_on_conflict'] = kwargs.get('ignore_on_conflict',False)
        self.insert(**kwargs)

    def read_file(self,filepath):
        reader = {
            'csv':pd.read_csv,
            'pkl':pd.read_pickle,
            'parquet':pd.read_parquet,
        }
        extension = filepath.split('.')[-1]
        method = reader.get(extension,None)

        if method is not None:
            df = method(filepath)
        else:
            raise TypeError(f'Unsupported file extension: .{extension}. Supported file extensions are {list(reader.keys())}')
        
        return df

    def insert(self,table_name,fields,values,on_conflict_fields=None,ignore_on_conflict=False):
        values = [
            [
                None if isinstance(values[i][j],(int,float)) and math.isnan(values[i][j])\
                    else values[i][j] for j in range(len(values[i]))
            ] for i in range(len(values))
        ]
        fields_placeholder = ', '.join([f'\"{field}\"' for field in fields])
        values_placeholder = ', '.join(['?' for _ in fields])
        query = f"""
        INSERT INTO {table_name} ({fields_placeholder}) VALUES ({values_placeholder})
        """

        if on_conflict_fields:
            on_conflict_update_fields = [f'\"{field}\"' for field in fields if field not in on_conflict_fields]
            on_conflict_fields_placeholder = ', '.join([f'\"{field}\"' for field in on_conflict_fields])
            on_conflict_placeholder = f'({", ".join(on_conflict_update_fields)}) = '\
                f'({", ".join(["EXCLUDED." + field for field in on_conflict_update_fields])})'

            if ignore_on_conflict or len(set(fields+on_conflict_fields)) == len(on_conflict_fields):
                query = query.replace('INSERT','INSERT OR IGNORE')
            else:
                query += f"ON CONFLICT ({on_conflict_fields_placeholder}) DO UPDATE SET {on_conflict_placeholder}"
        
        else:
            pass
        
        try:
            connection = self.__get_connection()
            query = self.__validate_query(query)

            try:
                connection.executemany(query,values)
            except:
                print(query)
                print(values)
                assert False
            connection.commit()
        finally:
            connection.close()

    def __register_adapter(self):
        sqlite3.register_adapter(np.int64,lambda x: int(x))
        sqlite3.register_adapter(np.int32,lambda x: int(x))
        sqlite3.register_adapter(np.datetime64,lambda x: np.datetime_as_string(x,unit='s').replace('T',' '))

class CityLearnDatabase(SQLiteDatabase):
    def __init__(self,filepath,env,agent,overwrite=False,apply_changes=False):
        super().__init__(filepath)
        self.env = env
        self.agent = agent
        self.__build(overwrite,apply_changes)
        self.__MINIMUM_ROW_ID = 1

    @property
    def env(self):
        return self.__env

    @property
    def agent(self):
        return self.__agent

    @env.setter
    def env(self,env):
        self.__env = env

    @agent.setter
    def agent(self,agent):
        self.__agent = agent

    def __build(self,overwrite,apply_changes):
        schema_filepath = 'schema.sql'
        
        if os.path.isfile(self.filepath):
            if overwrite:
                os.remove(self.filepath)
            elif not apply_changes:
                return
            else:
                pass
        else:
            pass

        self.execute_sql_from_file(schema_filepath)

    def initialize(self,**kwargs):
        # simulation
        start_timestamp = datetime.utcnow().replace(tzinfo=pytz.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        self.__simulation_id = self.__MINIMUM_ROW_ID
        record = {
            'id':self.__simulation_id,
            'start_timestamp':start_timestamp,
            'end_timestamp':None,
            'last_timestep_timestamp':None,
            'success':False
        }
        self.insert('simulation',list(record.keys()),[list(record.values())])

        # environment
        self.__environment_id = self.__MINIMUM_ROW_ID
        record = {
            'id':self.__environment_id,
            'simulation_period_start':self.env.simulation_period[0],
            'simulation_period_end':self.env.simulation_period[1],
            'central_agent':self.env.central_agent
        }
        self.insert('environment',list(record.keys()),[list(record.values())])

        # agent
        columns = self.get_table('agent').columns.tolist()

        if not self.env.central_agent:
            agent_count = len(self.env.buildings)
        else:
            agent_count = 1
        
        for i in range(self.__MINIMUM_ROW_ID,agent_count + self.__MINIMUM_ROW_ID):
            record = {
                key:value if not isinstance(value,list) else str(value)
                for key,value in kwargs.items() if key in columns
            }
            record['name'] = kwargs['agent_name']
            record['reward_style'] = kwargs['reward_style']
            record['deterministic_period_start'] = kwargs['deterministic_period_start']
            record['id'] = i
            self.insert('agent',list(record.keys()),[list(record.values())])

        building_attributes = read_json(os.path.join(self.env.data_path,self.env.building_attributes))

        for i, (building_name, building) in enumerate(self.env.buildings.items()):
            # building
            building_id = i + 1
            record = {
                'id':building_id,
                'name':building_name,
                'environment_id':self.__environment_id,
                'agent_id':i+self.__MINIMUM_ROW_ID if agent_count > 1 else self.__MINIMUM_ROW_ID,
                'type':building_attributes[building_name]['Building_Type'],
                'solar_power_installed':building_attributes[building_name]['Solar_Power_Installed(kW)'],
            }
            self.insert('building',list(record.keys()),[list(record.values())])

            # state space
            record = self.env.buildings_states_actions[building_name]['states']
            record['building_id'] = building_id
            self.insert('state_space',list(record.keys()),[list(record.values())])

            # action space
            record = self.env.buildings_states_actions[building_name]['actions']
            record['building_id'] = building_id
            self.insert('action_space',list(record.keys()),[list(record.values())])

            # cooling device
            record = {
                'id':building_id,
                'building_id':building_id,
                'nominal_power':building.cooling_device.nominal_power,
                'eta_tech':building.cooling_device.eta_tech,
                't_target_cooling':building.cooling_device.t_target_cooling,
                't_target_heating':building.cooling_device.t_target_heating,
            }
            self.insert('cooling_device',list(record.keys()),[list(record.values())])

            # dhw heating device
            record = {
                'id':building_id,
                'building_id':building_id,
                'nominal_power':building.dhw_heating_device.nominal_power,
                'efficiency':building.dhw_heating_device.efficiency,
            }
            self.insert('dhw_heating_device',list(record.keys()),[list(record.values())])

            # cooling storage
            record = {
                'id':building_id,
                'building_id':building_id,
                'capacity':building.cooling_storage.capacity,
                'max_power_output':building.cooling_storage.max_power_output,
                'max_power_charging':building.cooling_storage.max_power_charging,
                'efficiency':building.cooling_storage.efficiency,
                'loss_coef':building.cooling_storage.loss_coef,
            }
            self.insert('cooling_storage',list(record.keys()),[list(record.values())])

            # dhw storage
            record = {
                'id':building_id,
                'building_id':building_id,
                'capacity':building.dhw_storage.capacity,
                'max_power_output':building.dhw_storage.max_power_output,
                'max_power_charging':building.dhw_storage.max_power_charging,
                'efficiency':building.dhw_storage.efficiency,
                'loss_coef':building.dhw_storage.loss_coef,
            }
            self.insert('dhw_storage',list(record.keys()),[list(record.values())])

            # electrical storage
            record = {
                'id':building_id,
                'building_id':building_id,
                'capacity':building.electrical_storage.capacity,
                'nominal_power':building.electrical_storage.nominal_power,
                'capacity_loss_coef':building.electrical_storage.capacity_loss_coef,
                'power_efficiency_curve':str(building.electrical_storage.power_efficiency_curve.tolist())\
                    if building.electrical_storage.power_efficiency_curve is not None else None,
                'capacity_power_curve':str(building.electrical_storage.capacity_power_curve.tolist())\
                    if building.electrical_storage.capacity_power_curve is not None else None,
                'efficiency':building.electrical_storage.efficiency,
                'loss_coef':building.electrical_storage.loss_coef,
            }
            self.insert('electrical_storage',list(record.keys()),[list(record.values())])

    def timestep_update(self,timestep,maximum_timestep,episode,env):
        ix = timestep
        timestep = int(timestep + (episode*maximum_timestep))

        # update environment table
        record = {
            'timestep':timestep,
            'carbon_emissions':env.carbon_emissions[ix],
            'net_electric_consumption':env.net_electric_consumption[ix],
            'electric_consumption_electric_storage':env.electric_consumption_electric_storage[ix],
            'electric_consumption_dhw_storage':env.electric_consumption_dhw_storage[ix],
            'electric_consumption_cooling_storage':env.electric_consumption_cooling_storage[ix],
            'electric_consumption_dhw':env.electric_consumption_dhw[ix],
            'electric_consumption_cooling':env.electric_consumption_cooling[ix],
            'electric_consumption_appliances':env.electric_consumption_appliances[ix],
            'electric_generation':env.electric_generation[ix]
        }
        self.insert(
            'environment_timeseries',
            list(record.keys()),
            [list(record.values())],
            on_conflict_fields=['timestamp']
        )
        
        # update building table
        for building in env.buildings:
            record = {
                'timestep':timestep,
                'building_id':building.buildingId,
                'cooling_demand_building':building.cooling_demand_building[ix],
                'dhw_demand_building':building.dhw_demand_building[ix],
                'electric_consumption_appliances':building.electric_consumption_appliances[ix],
                'electric_generation':building.electric_generation[ix],
                'electric_consumption_cooling':building.electric_consumption_cooling[ix],
                'electric_consumption_cooling_storage':building.electric_consumption_cooling_storage[ix],
                'electric_consumption_dhw':building.electric_consumption_dhw[ix],
                'electric_consumption_dhw_storage':building.electric_consumption_dhw_storage[ix],
                'net_electric_consumption':building.net_electric_consumption[ix],
                'cooling_device_to_building':building.cooling_device_to_building[ix],
                'cooling_storage_to_building':building.cooling_storage_to_building[ix],
                'cooling_device_to_storage':building.cooling_device_to_storage[ix],
                'cooling_storage_soc':building.cooling_storage_soc[ix],
                'dhw_heating_device_to_building':building.dhw_heating_device_to_building[ix],
                'dhw_storage_to_building':building.dhw_storage_to_building[ix],
                'dhw_heating_device_to_storage':building.dhw_heating_device_to_storage[ix],
                'dhw_storage_soc':building.dhw_storage_soc[ix],
                'electrical_storage_electric_consumption':building.electrical_storage_electric_consumption[ix],
                'electrical_storage_soc':building.electrical_storage_soc[ix]
            }
            self.insert(
                'building_timeseries',
                list(record.keys()),
                [list(record.values())],
                on_conflict_fields=['timestamp','building_id']
            )

            # update cooling device timeseries
            record = {
                'timestep':timestep,
                'building_id':building.buildingId,
                'cop_heating':building.cooling_device.cop_heating[ix],
                'cop_cooling':building.cooling_device.cop_heating[ix],
                'electrical_consumption_cooling':building.cooling_device.electrical_consumption_cooling[ix],
                'electrical_consumption_heating':building.cooling_device.electrical_consumption_heating[ix],
                'heat_supply':building.cooling_device.heat_supply[ix],
                'cooling_supply':building.cooling_device.cooling_supply[ix],
            }
            self.insert(
                'cooling_device_timeseries',
                list(record.keys()),
                [list(record.values())],
                on_conflict_fields=['timestep','building_id']
            )

            # update dhw heating device timeseries
            record = {
                'timestep':timestep,
                'building_id':building.buildingId,
                'electrical_consumption_heating':building.dhw_heating_device.electrical_consumption_heating[ix],
                'heat_supply':building.dhw_heating_device.heat_supply[ix]
            }
            self.insert(
                'dhw_heating_device_timeseries',
                list(record.keys()),
                [list(record.values())],
                on_conflict_fields=['timestep','building_id']
            )


