import os
import sqlite3
from eppy import modeleditor
from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs
import matplotlib.pyplot as plt
import pandas as pd
from load import BuildingAmericaDomesticHotWater
from utilities import read_json, make_directory

class ORNL:
    def __init__(self,epw_filepath,ddy_filepath,idd_filepath=None,flowrate_per_person=None):
        self.idd_filepath = idd_filepath
        self.epw_filepath = epw_filepath
        self.ddy_filepath = ddy_filepath
        self.flowrate_per_person = flowrate_per_person

    @property
    def idd_filepath(self):
        return self.__idd_filepath

    @property
    def flowrate_per_person(self):
        return self.__flowrate_per_person
    
    @property
    def epw_filepath(self):
        return self.__epw_filepath
    
    @property
    def ddy_filepath(self):
        return self.__ddy_filepath

    @staticmethod
    def supported_building_types():
        return list(ORNL.settings()['flowrate_per_person'].keys())

    @staticmethod
    def settings():
        return read_json('.misc/settings.json')

    @staticmethod
    def citylearn_simulation_timeseries_query():
        with open('.misc/citylearn_simulation_timeseries.sql','r') as f:
            query = f.read()
            query = query.replace(',)',')')
        
        return query

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
        self.__ddy_idf = IDF(self.ddy_filepath)

    @flowrate_per_person.setter
    def flowrate_per_person(self,flowrate_per_person):
        if flowrate_per_person is None:
            self.__idd_filepath = self.settings()['flowrate_per_person']
        else:
            self.__flowrate_per_person = flowrate_per_person

    def simulate(self,idf_filepaths,output_directory=None,cpu=2):
        idf_output_directories = []
        idf_output_filepaths = []
        people_counts = {}

        # ********** PREPROCESSING **********
        if output_directory is None:
            output_directory = '../data/simulation_output/'
        else:
            pass
        
        # create output directories
        for idf_filepath in idf_filepaths:
            idf_directory = '/'.join(idf_filepath.split('/')[-2:]).split('.')[0]
            idf_output_directories.append(os.path.join(output_directory,idf_directory))
            make_directory(idf_output_directories[-1])

        # preprocess and store idf
        for idf_filepath, idf_output_directory in zip(idf_filepaths, idf_output_directories):
            idf = self.preprocess(IDF(idf_filepath,self.epw_filepath))
            idf_output_filepaths.append(os.path.join(idf_output_directory,f'{idf_output_directory.split("/")[-1]}.idf'))
            people_counts[idf_output_filepaths[-1]] = self.estimate_people_count(idf)
            idf.saveas(idf_output_filepaths[-1])

        # ********** SIMULATION **********
        idfs = (IDF(idf_output_filepath,self.epw_filepath) for idf_output_filepath in idf_output_filepaths)
        runs = (
            (idf,self.__get_eplaunch_options(idf,output_directory=idf_output_directory)) 
            for idf, idf_output_directory in zip(idfs,idf_output_directories)
        )
        runIDFs(runs,cpu)

        # ********** WRITE OUTPUT **********
        # hot water demand
        badhw = BuildingAmericaDomesticHotWater()
        hot_water_demand = {
            people_count:badhw.get_demand(people_count,self.epw_filepath)
            for people_count in set(people_counts.values())
        }

        # store citylearn simulation output
        query = ORNL.citylearn_simulation_timeseries_query()
        
        for idf_output_filepath, idf_output_directory in zip(idf_output_filepaths, idf_output_directories):
            db_filepath = idf_output_filepath.replace('.idf','.sql')
            data = self.citylearn_simulation_timeseries(db_filepath,query=query)
            data['DHW Heating [kWh]'] = hot_water_demand[people_counts[idf_output_filepath]]
            data.to_csv(idf_output_filepath.replace('.idf','.csv'),index=False)
            fig, axs = self.plot_citylearn_simulation_timeseries(data)
            fig.suptitle('/'.join(idf_output_filepath.split('/')[-3:-1]),y=1.01)
            plt.savefig(idf_output_filepath.replace('.idf','.pdf'),transparent=True,bbox_inches='tight')

    def __get_eplaunch_options(self,idf,**kwargs):
        idf_version = idf.idfobjects['version'][0].Version_Identifier.split('.')
        idf_version.extend([0] * (3 - len(idf_version)))
        idf_version_str = '-'.join([str(item) for item in idf_version])
        filename = idf.idfname

        options = {
            'ep_version':idf_version_str,
            'output_prefix':os.path.basename(filename).split('.')[0],
            'output_suffix':'C',
            'output_directory':'',
            'readvars':False,
            'expandobjects':True,
        }
        options = {**options,**kwargs}
        return options

    def citylearn_simulation_timeseries(self,db_filepath,query=None):
        if query is None:
            query = ORNL.citylearn_simulation_timeseries_query()
        else:
            pass

        con = sqlite3.connect(db_filepath)
        data = pd.read_sql(query,con)
        con.close()
        # Parantheses in column names changed to square braces to match CityLearn format
        # SQLite3 ignores square braces in column names so parentheses used as temporary fix. 
        data.columns = [c.replace('(','[').replace(')',']') for c in data.columns]
        return data

    def plot_citylearn_simulation_timeseries(self,data):
        columns = [
            'Indoor Temperature [C]',
            'Average Unmet Cooling Setpoint Difference [C]',
            'Indoor Relative Humidity [%]',
            'Equipment Electric Power [kWh]',
            'DHW Heating [kWh]',
            'Heating Load [kWh]',
            'Cooling Load [kWh]',
        ]
        fig, axs = plt.subplots(len(columns),1,figsize=(18,len(columns)*2.5))

        for ax, column in zip(fig.axes, columns):
            ax.plot(data[column])
            ax.set_title(column)
            ax.margins(0)

        plt.tight_layout()
        return fig, axs

    def estimate_people_count(self,idf):
        people_count = 0

        for people_object in idf.idfobjects['People']:
            occupancy_density = people_object.People_per_Zone_Floor_Area # people/m^2
            zone_list_object = idf.getobject('ZONELIST',people_object.Zone_or_ZoneList_Name)
            
            for i in range(1,501):
                zone_name = zone_list_object[f'Zone_{i}_Name']
                
                try:
                    assert zone_name != ''
                except AssertionError:
                    break

                people_count += modeleditor.zonearea_floor(idf,zone_name)*occupancy_density

        return round(people_count,0)

    def preprocess(self,idf):
        building_type = idf.idfobjects['BUILDING'][0]['Name'].split(' created')[0][1:]

        try:
            assert building_type in self.supported_building_types()
        except AssertionError:
            raise UnsupportedBuildingTypeError

        # *********** update site location object ***********
        idf.idfobjects['Site:Location'] = []

        for obj in self.__ddy_idf.idfobjects['Site:Location']:
            idf.copyidfobject(obj)

        # *********** update design-day objects ***********
        idf.idfobjects['SizingPeriod:DesignDay'] = []
        design_day_suffixes = ['Ann Htg 99.6% Condns DB', 'Ann Clg .4% Condns DB=>MWB']

        for obj in self.__ddy_idf.idfobjects['SizingPeriod:DesignDay']:
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
            obj.Flow_Rate_per_Person = ORNL.settings()['flowrate_per_person'][building_type]
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

        return idf

class Error(Exception):
    """Base class for other exceptions"""

class UnsupportedBuildingTypeError(Error):
    """Raised when attempting to preprocess an unsupported building."""
    
    def __init__(self,message=f'[Error] Unsupported building type. Valid building types are {ORNL.supported_building_types()}.'):
        super().__init__(message)
        self.message = message

if __name__ == '__main__':
    selected_idfs = read_json('../data/idf/selected.json')
    idf_filepaths = [[os.path.join(f'../data/idf/cities/{city}/{idf_id}.idf') for idf_id in idf_ids] for city, idf_ids in selected_idfs.items()]
    idf_filepaths = [filepath for sublist in idf_filepaths for filepath in sublist]
    idf_filepaths = idf_filepaths[0:1]
    ornl = ORNL(
        '../data/weather/USA_TX_Austin-Camp.Mabry.722544_TMY3.epw',
        '../data/weather/USA_TX_Austin-Camp.Mabry.722544_TMY3.ddy'
    )
    ornl.simulate(idf_filepaths)