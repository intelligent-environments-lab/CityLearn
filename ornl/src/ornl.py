import os
import sqlite3
import random
from eppy import modeleditor
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs
import matplotlib.pyplot as plt
import pandas as pd
from load import BuildingAmericaDomesticHotWater
from utilities import make_directory, read_json, write_json

class ORNLIDF:
    __DEFAULT_OUTPUT_DIRECTORY = '../data/simulation_output/'

    def __init__(self,idf_filepath,epw_filepath,ddy_filepath,idd_filepath=None,id=None,occupancy=None,output_directory=None,random_state=None):
        self.idd_filepath = idd_filepath
        self.epw_filepath = epw_filepath
        self.idf_filepath = idf_filepath
        self.ddy_filepath = ddy_filepath
        self.id = id
        self.occupancy = occupancy
        self.output_directory = output_directory
        self.random_state = random_state

    @property
    def idf_filepath(self):
        return self.__idf_filepath

    @property
    def idf(self):
        return self.__idf

    @property
    def idd_filepath(self):
        return self.__idd_filepath
    
    @property
    def epw_filepath(self):
        return self.__epw_filepath
    
    @property
    def ddy_filepath(self):
        return self.__ddy_filepath

    @property
    def id(self):
        return self.__id

    @property
    def occupancy(self):
        return self.__occupancy

    @property
    def output_directory(self):
        return self.__output_directory
        
    @property
    def random_state(self):
        return self.__random_state

    @idf_filepath.setter
    def idf_filepath(self,idf_filepath):
        self.__idf_filepath = idf_filepath
        self.__idf = IDF(idf_filepath,self.epw_filepath)

    @idf.setter
    def idf(self,idf):
        self.__idf = idf
        
    @idd_filepath.setter
    def idd_filepath(self,idd_filepath):
        if idd_filepath is None:
            self.__idd_filepath = self.settings()['idd_filepath']
        else:
            self.__idd_filepath = idd_filepath
        
        IDF.setiddname(self.idd_filepath)

    @epw_filepath.setter
    def epw_filepath(self,epw_filepath):
        self.__epw_filepath = epw_filepath

    @ddy_filepath.setter
    def ddy_filepath(self,ddy_filepath):
        self.__ddy_filepath = ddy_filepath

    @id.setter
    def id(self,id):
        if id is None:
            self.__id = self.idf_filepath.split('/')[-1].split('.')[0]
        else:
            self.__id = id

    @occupancy.setter
    def occupancy(self,occupancy):
        if occupancy is not None:
            if occupancy >= 0:
                self.__occupancy = occupancy
            else:
                raise ValueError('occupancy must be >= 0.')
        else:
            self.__occupancy = self.estimate_occupancy()

    @output_directory.setter
    def output_directory(self,output_directory):
        if output_directory is None:
            self.__output_directory = os.path.join(ORNLIDF.__DEFAULT_OUTPUT_DIRECTORY,self.id)
        else:
            self.__output_directory = output_directory

    @random_state.setter
    def random_state(self,random_state):
        if random_state is not None:
            if random_state >= 0:
                self.__random_state = random_state
            else:
                raise ValueError('random_state must be >= 0.')
        else:
            self.random_state = random.randint(0,100000000)

    @staticmethod
    def settings():
        return read_json('.misc/settings.json')

    def simulate(self,**kwargs):    
        make_directory(self.output_directory)

    def eplaunch_options(self,**kwargs):
        idf_version = self.idf.idfobjects['version'][0].Version_Identifier.split('.')
        idf_version.extend([0] * (3 - len(idf_version)))
        idf_version_str = '-'.join([str(item) for item in idf_version])

        options = {
            'ep_version':idf_version_str,
            'output_prefix':self.id,
            'output_suffix':'C',
            'output_directory':self.output_directory,
            'readvars':False,
            'expandobjects':True,
        }
        options = {**options,**kwargs}
        return options

    def estimate_occupancy(self):
        occupancy = 0

        for people_object in self.idf.idfobjects['People']:
            occupancy_density = people_object.People_per_Zone_Floor_Area # people/m^2
            zone_list_object = self.idf.getobject('ZONELIST',people_object.Zone_or_ZoneList_Name)
            
            for i in range(1,501):
                zone_name = zone_list_object[f'Zone_{i}_Name']
                
                try:
                    assert zone_name != ''
                except AssertionError:
                    break

                occupancy += modeleditor.zonearea_floor(self.idf,zone_name)*occupancy_density
        
        occupancy = round(occupancy,0)
        return occupancy

    def preprocess(self):
        raise NotImplementedError

class CityLearnIDF(ORNLIDF):
    def __init__(self,*args,timeseries=None,attributes=None,state_action_space=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.timeseries = timeseries
        self.attributes = attributes
        self.state_action_space = state_action_space

    @property
    def timeseries(self):
        return self.__timeseries

    @property
    def attributes(self):
        return self.__attributes

    @property
    def state_action_space(self):
        return self.__state_action_space

    @timeseries.setter
    def timeseries(self,timeseries):
        self.__timeseries = timeseries

    @attributes.setter
    def attributes(self,attributes):
        self.__attributes = attributes

    @state_action_space.setter
    def state_action_space(self,state_action_space):
        self.__state_action_space = state_action_space
    
    def simulate(self,**kwargs):
        super().simulate(**kwargs)
        self.preprocess()
        self.idf.run(**self.eplaunch_options(**kwargs))
        self.__update_timeseries()
        self.__update_attributes()
        self.__update_state_action_space()

    def __update_timeseries(self):
        with open('.misc/citylearn_simulation_timeseries.sql','r') as f:
            query = f.read()
            query = query.replace(',)',')')

        con = sqlite3.connect(os.path.join(self.output_directory,f'{self.id}.sql'))
        timeseries = pd.read_sql(query,con)
        con.close()
        # Parantheses in column names changed to square braces to match CityLearn format
        # SQLite3 ignores square braces in column names so parentheses used as temporary fix. 
        timeseries.columns = [c.replace('(','[').replace(')',']') for c in timeseries.columns]
        timeseries['DHW Heating [kWh]'] = BuildingAmericaDomesticHotWater().get_demand(self.occupancy,self.epw_filepath)
        self.timeseries = timeseries

    def __update_attributes(self):
        attributes = read_json('.misc/citylearn_templates/attributes.json')
        self.attributes = attributes

    def __update_state_action_space(self):
        state_action_space = read_json('.misc/citylearn_templates/state_action_space.json')
        self.state_action_space = state_action_space

    def __plot_timeseries(self):
        columns = ['Indoor Temperature [C]','Average Unmet Cooling Setpoint Difference [C]','Indoor Relative Humidity [%]',
            'Equipment Electric Power [kWh]','DHW Heating [kWh]','Heating Load [kWh]','Cooling Load [kWh]',
        ]
        fig, axs = plt.subplots(len(columns),1,figsize=(18,len(columns)*2.5))

        for ax, column in zip(fig.axes, columns):
            ax.plot(self.timeseries[column])
            ax.set_title(column)
            ax.margins(0)

        plt.tight_layout()
        return fig, axs

    def save(self):
        self.idf.savecopy(os.path.join(self.output_directory,f'{self.id}.idf'))
        self.timeseries.to_csv(os.path.join(self.output_directory,f'{self.id}_timeseries.csv'),index=False)
        write_json(os.path.join(self.output_directory,f'{self.id}_attributes.json'),self.attributes)
        write_json(os.path.join(self.output_directory,f'{self.id}_state_action_space.json'),self.state_action_space)
        fig, axs = self.__plot_timeseries()
        fig.suptitle(f'{self.id}',y=1.01)
        plt.savefig(os.path.join(self.output_directory,f'{self.id}_timeseries_plot.pdf'),transparent=True,bbox_inches='tight')

    def preprocess(self):
        idf = self.idf
        ddy_idf = IDF(self.ddy_filepath)
        building_type = idf.idfobjects['BUILDING'][0]['Name'].split(' created')[0][1:]
        supported_building_types = list(ORNLIDF.settings()['flowrate_per_person'].keys())

        if building_type not in supported_building_types:
            raise UnsupportedBuildingTypeError(f'[Error] Unsupported building type. Valid building types are {ORNLIDF.supported_building_types()}.')
        else:
            pass

        # *********** update site location object ***********
        idf.idfobjects['Site:Location'] = []

        for obj in ddy_idf.idfobjects['Site:Location']:
            idf.copyidfobject(obj)

        # *********** update design-day objects ***********
        idf.idfobjects['SizingPeriod:DesignDay'] = []
        design_day_suffixes = ['Ann Htg 99.6% Condns DB', 'Ann Clg .4% Condns DB=>MWB']

        for obj in ddy_idf.idfobjects['SizingPeriod:DesignDay']:
            for suffix in design_day_suffixes:
                if obj.Name.endswith(suffix):
                    idf.copyidfobject(obj)
                    break
                else:
                    continue

        # *********** update HVAC to ideal loads ***********
        # remove existing HVAC objects not needed
        objects = [
            'AIRTERMINAL:SINGLEDUCT:CONSTANTVOLUME:NOREHEAT',
            'ZONEHVAC:AIRDISTRIBUTIONUNIT',
            'ZONEHVAC:EQUIPMENTLIST',
            'FAN:CONSTANTVOLUME',
            'COIL:COOLING:DX:SINGLESPEED',
            'COIL:HEATING:FUEL',
            'COILSYSTEM:COOLING:DX',
            'CONTROLLER:OUTDOORAIR',
            'CONTROLLER:MECHANICALVENTILATION',
            'AIRLOOPHVAC:CONTROLLERLIST',
            'AIRLOOPHVAC',
            'AIRLOOPHVAC:OUTDOORAIRSYSTEM:EQUIPMENTLIST',
            'AIRLOOPHVAC:OUTDOORAIRSYSTEM',
            'OUTDOORAIR:MIXER',
            'AIRLOOPHVAC:ZONESPLITTER',
            'AIRLOOPHVAC:SUPPLYPATH',
            'AIRLOOPHVAC:ZONEMIXER',
            'AIRLOOPHVAC:RETURNPATH',
            'BRANCH',
            'BRANCHLIST',
            'ZONEHVAC:EQUIPMENTCONNECTIONS',
            'NODELIST',
            'OUTDOORAIR:NODELIST',
            'AVAILABILITYMANAGER:SCHEDULED',
            'AVAILABILITYMANAGER:NIGHTCYCLE',
            'AVAILABILITYMANAGERASSIGNMENTLIST',
            'SETPOINTMANAGER:SINGLEZONE:REHEAT',
            'SETPOINTMANAGER:MIXEDAIR',
            'CURVE:QUADRATIC',
            'CURVE:CUBIC',
            'CURVE:BIQUADRATIC'
        ]

        for obj in objects:
            idf.idfobjects[obj] = []

        # turn off zone, system and plant sizing calculation
        obj = idf.idfobjects['SIMULATIONCONTROL'][0]
        obj.Do_Zone_Sizing_Calculation = 'No'
        obj.Do_System_Sizing_Calculation = 'No'
        obj.Do_Plant_Sizing_Calculation = 'No'
        obj.Run_Simulation_for_Sizing_Periods = 'No'
        obj.Do_HVAC_Sizing_Simulation_for_Sizing_Periods = 'No'

        # add ideal load object & other relevant objects per zone
        idf.idfobjects['ZONEHVAC:IDEALLOADSAIRSYSTEM'] = []
        idf.idfobjects['ZONEVENTILATION:DESIGNFLOWRATE'] = []
        zones = idf.idfobjects['Zone']

        for zone in zones:
            zone_name = zone.Name
            # ideal load object
            obj = idf.newidfobject('ZONEHVAC:IDEALLOADSAIRSYSTEM')
            obj.Name = f'{zone_name} Ideal Loads Air System'
            obj.Zone_Supply_Air_Node_Name = f'{zone_name} Ideal Loads Supply Inlet'
            # ventilation object
            obj = idf.newidfobject('ZONEVENTILATION:DESIGNFLOWRATE')
            obj.Name = f'{zone_name}  Ventilation per Person'
            obj.Zone_or_ZoneList_Name = f'{zone_name}'
            obj.Schedule_Name = 'MidriseApartment Apartment Occ'
            obj.Design_Flow_Rate_Calculation_Method = 'Flow/Person'
            obj.Ventilation_Type = ''
            obj.Flow_Rate_per_Person = ORNLIDF.settings()['flowrate_per_person'][building_type]
            # equipment list
            obj = idf.newidfobject('ZONEHVAC:EQUIPMENTLIST')
            obj.Name = f'{zone_name}  Equipment List'
            obj.Zone_Equipment_1_Object_Type = 'ZoneHVAC:IdealLoadsAirSystem'
            obj.Zone_Equipment_1_Name = idf.idfobjects['ZONEHVAC:IDEALLOADSAIRSYSTEM'][-1].Name
            obj.Zone_Equipment_1_Cooling_Sequence = 1
            obj.Zone_Equipment_1_Heating_or_NoLoad_Sequence = 1
            # equipment connections
            obj = idf.newidfobject('ZONEHVAC:EQUIPMENTCONNECTIONS')
            obj.Zone_Name = zone_name
            obj.Zone_Conditioning_Equipment_List_Name = idf.idfobjects['ZONEHVAC:EQUIPMENTLIST'][-1].Name
            obj.Zone_Air_Inlet_Node_or_NodeList_Name = idf.idfobjects['ZONEHVAC:IDEALLOADSAIRSYSTEM'][-1].Zone_Supply_Air_Node_Name
            obj.Zone_Air_Node_Name = f'{zone_name} Zone Air Node'
            obj.Zone_Return_Air_Node_or_NodeList_Name = f'{zone_name} Return Outlet'

        # *********** update timestep ***********
        obj = idf.idfobjects['Timestep'][0]
        obj.Number_of_Timesteps_per_Hour = 1

        # *********** update output files ***********
        obj = idf.idfobjects['OutputControl:Files'][0]
        obj.Output_SQLite = 'Yes'
        obj.Output_CSV = 'No'
        obj.Output_Tabular = 'No'
        obj.Output_END = 'No'

        # *********** update output variables ***********
        output_variables = [
            'Zone Ideal Loads Zone Total Cooling Energy',
            'Zone Ideal Loads Zone Total Heating Energy',
            'Electric Equipment Electricity Energy',
            'Zone Air Temperature',
            'Zone Air Relative Humidity',
            'Zone Thermostat Cooling Setpoint Temperature',
        ]
        idf.idfobjects['Output:Variable'] = []

        for output_variable in output_variables:
            obj = idf.newidfobject('Output:Variable')
            obj.Variable_Name = output_variable

        self.idf = idf

class Error(Exception):
    """Base class for other exceptions"""

class UnsupportedBuildingTypeError(Error):
    """Raised when attempting to preprocess an unsupported building."""
    pass

if __name__ == '__main__':
    selected_idfs = read_json('../data/idf/selected.json')
    idf_filepaths = [[os.path.join(f'../data/idf/cities/{city}/{idf_id}.idf') for idf_id in idf_ids] for city, idf_ids in selected_idfs.items()]
    idf_filepaths = [filepath for sublist in idf_filepaths for filepath in sublist]
    idf_filepath = idf_filepaths[0]
    epw_filepath = '../data/weather/USA_TX_Austin-Camp.Mabry.722544_TMY3.epw'
    ddy_filepath = '../data/weather/USA_TX_Austin-Camp.Mabry.722544_TMY3.ddy'
    clidf = CityLearnIDF(idf_filepath,epw_filepath,ddy_filepath)
    clidf.simulate()
    clidf.save()