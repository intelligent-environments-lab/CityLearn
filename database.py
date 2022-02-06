from copy import deepcopy
from datetime import datetime
import math
import os
import pytz
import shutil
import sqlite3
import numpy as np
import pandas as pd
from energy_models import HeatPump

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

    def query(self,query,foreign_key_constraint=True,**kwargs):
        query = self.__validate_query(query)
        responses = []
        connection = self.__get_connection()
        
        try:
            queries = query.split(';')
            queries = [f'PRAGMA foreign_keys = {"ON" if foreign_key_constraint else "OFF"}'] + queries
            
            for q in queries:
                try:
                    cursor = connection.execute(q,**kwargs)
                except Exception as e:
                    print(q)
                    print(e)
                    assert False
                response = cursor.fetchall()
                responses.append(response)
                connection.commit()
        finally:
            connection.close()

        return responses

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
            except Exception as e:
                print(e)
                print(query)
                assert False
                
            connection.commit()
        finally:
            connection.close()

    def __register_adapter(self):
        sqlite3.register_adapter(np.int64,lambda x: int(x))
        sqlite3.register_adapter(np.int32,lambda x: int(x))
        sqlite3.register_adapter(np.float32,lambda x: float(x))
        sqlite3.register_adapter(np.float64,lambda x: float(x))
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

    @classmethod
    def concatenate(cls,filepath,source_filepaths):
        database = CityLearnDatabase(filepath,None,None,overwrite=True)
        table_names = database.query_table("""
        SELECT 
            name 
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """)['name'].tolist()
        id_update_table_names = [
            'simulation','environment','agent','building','cooling_device','dhw_heating_device',
            'cooling_storage','dhw_storage','electrical_storage','timestep',
        ]
        temp_filepath = 'temp.db'
        
        for i, source_filepath in enumerate(source_filepaths):
            try:
                source_database = SQLiteDatabase(temp_filepath)
                shutil.copy(source_filepath,temp_filepath)
                j = 0

                for table_name in table_names:
                    print(f'\rDatabase: {i+1}/{len(source_filepaths)}, Table: {j+1}/{len(table_names)}',end='')

                    if table_name in id_update_table_names:
                        max_id = database.query_table(f"SELECT MAX(id) AS id FROM {table_name}").iloc[0]['id'] if i > 0 else 0
                        source_database.query(f"UPDATE {table_name} SET id = id + {max_id}")
                    else:
                        pass

                    data = source_database.get_table(table_name)
                    database.insert(table_name,data.columns.tolist(),data.values)
                    j += 1
        
            finally:
                os.remove(temp_filepath)

    def initialize(self,**kwargs):
        # simulation
        start_timestamp = datetime.utcnow().replace(tzinfo=pytz.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        self.__simulation_id = self.__MINIMUM_ROW_ID
        simulation_name = kwargs.get('simulation_name',None)
        record = {
            'id':self.__simulation_id,
            'name': simulation_name if simulation_name is not None else str(self.__simulation_id),
            'start_timestamp':start_timestamp,
            'end_timestamp':None,
            'successful':False
        }
        self.insert('simulation',list(record.keys()),[list(record.values())])

        # environment
        self.__environment_id = self.__MINIMUM_ROW_ID
        record = {
            'id':self.__environment_id,
            'simulation_id':self.__simulation_id,
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
            record['basic_rbc'] = kwargs['basic_rbc']
            record['id'] = i
            self.insert('agent',list(record.keys()),[list(record.values())])

        for i, (building_name, building) in enumerate(self.env.buildings.items()):
            # building
            building = deepcopy(building)
            building_id = i + 1
            record = {
                'id':building_id,
                'name':building_name,
                'environment_id':self.__environment_id,
                'agent_id':i+self.__MINIMUM_ROW_ID if agent_count > 1 else self.__MINIMUM_ROW_ID,
                'type':building.building_type,
                'solar_power_installed':building.solar_power_capacity,
            }
            self.insert('building',list(record.keys()),[list(record.values())])

            # state space
            record = deepcopy(self.env.buildings_states_actions[building_name]['states'])
            record['building_id'] = building_id
            self.insert('state_space',list(record.keys()),[list(record.values())])

            # action space
            record = deepcopy(self.env.buildings_states_actions[building_name]['actions'])
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

    def timestep_update(self,**kwargs):
        start_timestep = kwargs['start_timestep']
        alt_start_timestep = max([0,start_timestep-1])
        end_timestep = kwargs['end_timestep']
        episode = kwargs['episode']
        timesteps = list(range(start_timestep,end_timestep))
        timestep_ids = [episode*self.env.simulation_period[1] + i + 1 for i in range(start_timestep,end_timestep)]
        alt_timestep_ids = timestep_ids[1:] if start_timestep == 0 else timestep_ids
        building_sim_results = list(self.env.buildings.values())[0].sim_results
        buildings = self.get_table('building').set_index('name').to_dict(orient='index')
        actions = self.get_table('action_space').set_index('building_id').to_dict(orient='index')

        # timestep
        record = pd.DataFrame({
            'id':timestep_ids,
            'timestep':timesteps,
            'month':building_sim_results['month'][start_timestep:end_timestep],
            'hour':building_sim_results['hour'][start_timestep:end_timestep],
            'day_type':building_sim_results['day'][start_timestep:end_timestep],
            'daylight_savings_status':building_sim_results['daylight_savings_status'][start_timestep:end_timestep],
        })
        record['episode'] = episode
        record['environment_id'] = self.__environment_id
        self.insert('timestep',record.columns.tolist(),record.values)

        # weather
        record = pd.DataFrame({
            'timestep_id':timestep_ids,
            'outdoor_drybulb_temperature':building_sim_results['t_out'][start_timestep:end_timestep],
            'outdoor_relative_humidity':building_sim_results['rh_out'][start_timestep:end_timestep],
            'diffuse_solar_radiation':building_sim_results['diffuse_solar_rad'][start_timestep:end_timestep],
            'direct_solar_radiation':building_sim_results['direct_solar_rad'][start_timestep:end_timestep],            
        })
        self.insert('weather_timeseries',record.columns.tolist(),record.values)

        # environment
        record = pd.DataFrame({
            'timestep_id':alt_timestep_ids,
            'carbon_emissions':self.env.carbon_emissions[alt_start_timestep:max(timesteps)],
            'net_electric_consumption':self.env.net_electric_consumption[alt_start_timestep:max(timesteps)],
            'electric_consumption_electric_storage':self.env.electric_consumption_electric_storage[alt_start_timestep:max(timesteps)],
            'electric_consumption_dhw_storage':self.env.electric_consumption_dhw_storage[alt_start_timestep:max(timesteps)],
            'electric_consumption_cooling_storage':self.env.electric_consumption_cooling_storage[alt_start_timestep:max(timesteps)],
            'electric_consumption_dhw':self.env.electric_consumption_dhw[alt_start_timestep:max(timesteps)],
            'electric_consumption_cooling':self.env.electric_consumption_cooling[alt_start_timestep:max(timesteps)],
            'electric_consumption_appliances':self.env.electric_consumption_appliances[alt_start_timestep:max(timesteps)],
            'electric_generation':self.env.electric_generation[alt_start_timestep:max(timesteps)]
        })
        record['environment_id'] = self.__MINIMUM_ROW_ID
        self.insert('environment_timeseries',record.columns.tolist(),record.values)

        for i, (building_name, building) in enumerate(self.env.buildings.items()):
            # building
            building = deepcopy(building)
            building_id = buildings[building_name]['id']
            record = pd.DataFrame({
                'timestep_id':timestep_ids,
                'indoor_temperature':building.sim_results['t_in'][start_timestep:end_timestep],
                'average_unmet_cooling_setpoint_difference':building.sim_results['avg_unmet_setpoint'][start_timestep:end_timestep],
                'indoor_relative_humidity':building.sim_results['rh_in'][start_timestep:end_timestep],
                'cooling_demand_building':building.sim_results['cooling_demand'][start_timestep:end_timestep],
                'dhw_demand_building':building.sim_results['dhw_demand'][start_timestep:end_timestep],
                'electric_consumption_appliances':building.sim_results['non_shiftable_load'][start_timestep:end_timestep],
                'electric_generation':building.sim_results['solar_gen'][start_timestep:end_timestep],
            })
            record = pd.merge(
                record,
                pd.DataFrame({
                    'timestep_id':alt_timestep_ids,
                    'electric_consumption_cooling':building.electric_consumption_cooling[alt_start_timestep:max(timesteps)],
                    'electric_consumption_cooling_storage':building.electric_consumption_cooling_storage[alt_start_timestep:max(timesteps)],
                    'electric_consumption_dhw':building.electric_consumption_dhw[alt_start_timestep:max(timesteps)],
                    'electric_consumption_dhw_storage':building.electric_consumption_dhw_storage[alt_start_timestep:max(timesteps)],
                    'cooling_device_to_building':building.cooling_device_to_building[alt_start_timestep:max(timesteps)],
                    'cooling_storage_to_building':building.cooling_storage_to_building[alt_start_timestep:max(timesteps)],
                    'cooling_device_to_storage':building.cooling_device_to_storage[alt_start_timestep:max(timesteps)],
                    'dhw_heating_device_to_building':building.dhw_heating_device_to_building[alt_start_timestep:max(timesteps)],
                    'dhw_storage_to_building':building.dhw_storage_to_building[alt_start_timestep:max(timesteps)],
                    'dhw_heating_device_to_storage':building.dhw_heating_device_to_storage[alt_start_timestep:max(timesteps)],
                    'electrical_storage_electric_consumption':building.electrical_storage_electric_consumption[alt_start_timestep:max(timesteps)]
            }),on='timestep_id',how='left')
            record['building_id'] = building_id
            self.insert('building_timeseries',record.columns.tolist(),record.values)

            # cooling device
            record = pd.DataFrame({
                'timestep_id':timestep_ids,
                'cop_cooling':building.cooling_device.cop_cooling[start_timestep:end_timestep],
                'cop_heating':building.cooling_device.cop_heating[start_timestep:end_timestep] if isinstance(building.dhw_heating_device,HeatPump) else None,
            })
            record = pd.merge(
                record,
                pd.DataFrame({
                    'timestep_id':alt_timestep_ids,
                    'electrical_consumption_cooling':building.cooling_device.electrical_consumption_cooling[alt_start_timestep:max(timesteps)],
                    'electrical_consumption_heating':building.cooling_device.electrical_consumption_heating[alt_start_timestep:max(timesteps)] if isinstance(building.dhw_heating_device,HeatPump) else None,
                    'cooling_supply':building.cooling_device.cooling_supply[alt_start_timestep:max(timesteps)],
                    'heat_supply':building.cooling_device.heat_supply[alt_start_timestep:max(timesteps)] if isinstance(building.dhw_heating_device,HeatPump) else None,
            }),on='timestep_id',how='left')
            record['cooling_device_id'] = building_id
            self.insert('cooling_device_timeseries',record.columns.tolist(),record.values)

            # dhw heating device
            record = pd.DataFrame({
                'timestep_id':alt_timestep_ids,
                'electrical_consumption_heating':building.dhw_heating_device.electrical_consumption_heating[alt_start_timestep:max(timesteps)],
                'heat_supply':building.dhw_heating_device.heat_supply[alt_start_timestep:max(timesteps)],
            })
            record['dhw_heating_device_id'] = building_id
            self.insert('dhw_heating_device_timeseries',record.columns.tolist(),record.values)

            # cooling storage
            record = pd.DataFrame({
                'timestep_id':alt_timestep_ids,
                'soc':building.cooling_storage.soc[alt_start_timestep:max(timesteps)],
                'energy_balance':building.cooling_storage.energy_balance[alt_start_timestep:max(timesteps)],
            })
            record['cooling_storage_id'] = building_id
            self.insert('cooling_storage_timeseries',record.columns.tolist(),record.values)

            # dhw storage
            record = pd.DataFrame({
                'timestep_id':alt_timestep_ids,
                'soc':building.dhw_storage.soc[alt_start_timestep:max(timesteps)],
                'energy_balance':building.dhw_storage.energy_balance[alt_start_timestep:max(timesteps)],
            })
            record['dhw_storage_id'] = building_id
            self.insert('dhw_storage_timeseries',record.columns.tolist(),record.values)

            # electrical storage
            record = pd.DataFrame({
                'timestep_id':alt_timestep_ids,
                'soc':building.electrical_storage.soc[alt_start_timestep:max(timesteps)],
                'energy_balance':building.electrical_storage.energy_balance[alt_start_timestep:max(timesteps)],
            })
            record['electrical_storage_id'] = building_id
            self.insert('electrical_storage_timeseries',record.columns.tolist(),record.values)

            # action timeseries
            record = {
                'timestep_id':timestep_ids,
                'cooling_storage':[],
                'dhw_storage':[],
                'electrical_storage':[]
            }
            for j in timesteps:
                if actions[building_id]['cooling_storage']:
                    record['cooling_storage'].append(kwargs['action'][j][i][0])
                    kwargs['action'][j][i] = kwargs['action'][j][i][1:]
                else:
                    record['cooling_storage'].append(None)
                if actions[building_id]['dhw_storage']:
                    record['dhw_storage'].append(kwargs['action'][j][i][0])
                    kwargs['action'][j][i] = kwargs['action'][j][i][1:]
                else:
                    record['dhw_storage'].append(None)
                if actions[building_id]['electrical_storage']:
                    record['electrical_storage'].append(kwargs['action'][j][i][0])
                    kwargs['action'][j][i] = kwargs['action'][j][i][1:]
                else:
                    record['electrical_storage'].append(None)
            
            record = pd.DataFrame(record)
            record['building_id'] = building_id
            self.insert('action_timeseries',record.columns.tolist(),record.values)

            # reward
            record = pd.DataFrame({
                'timestep_id':timestep_ids,
                'value':[kwargs['reward'][j][i] for j in timesteps]
            })
            record['building_id'] = building_id
            self.insert('reward_timeseries',record.columns.tolist(),record.values)

    def end_simulation(self,**kwargs):
        end_timestamp = datetime.utcnow().replace(tzinfo=pytz.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        query = f"""
        UPDATE simulation
        SET
            end_timestamp = '{end_timestamp}',
            successful = {int(kwargs['successful'])}
        WHERE
            id = {self.__MINIMUM_ROW_ID}
        """
        _ = self.query(query)