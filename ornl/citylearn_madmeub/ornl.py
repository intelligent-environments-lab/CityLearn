import os
import sqlite3
import random
from eppy import modeleditor
from eppy.modeleditor import IDF
import matplotlib.pyplot as plt
import pandas as pd
from citylearn_madmeub.load import BuildingAmericaDomesticHotWater, PVSizing
from citylearn_madmeub.utilities import make_directory, read_json, write_json

class ORNLIDF:
    def __init__(self,idf_filepath,climate_zone,building_type,epw_filepath=None,ddy_filepath=None,idd_filepath=None,id=None,occupancy=None,output_directory=None,random_state=None):
        self.climate_zone = climate_zone
        self.building_type = building_type
        self.idd_filepath = idd_filepath
        self.epw_filepath = epw_filepath
        self.ddy_filepath = ddy_filepath
        self.idf_filepath = idf_filepath
        self.id = id
        self.occupancy = occupancy
        self.output_directory = output_directory
        self.random_state = random_state

    @property
    def idf_filepath(self):
        return self.__idf_filepath

    @property
    def climate_zone(self):
        return self.__climate_zone

    @property
    def building_type(self):
        return self.__building_type

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

    @climate_zone.setter
    def climate_zone(self,climate_zone):
        self.__climate_zone = climate_zone

    @building_type.setter
    def building_type(self,building_type):
        supported_building_types = [5]
        
        if building_type in supported_building_types:
            self.__building_type = building_type
        else:
            raise UnsupportedBuildingTypeError(
                f'Unsupported building type. Valid building types are {supported_building_types}.'
            )

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
        if epw_filepath is None:
            epw_filepath = self.settings()['weather']['epw'][str(int(self.climate_zone))]
            self.__epw_filepath = os.path.join(os.path.dirname(__file__),epw_filepath)
        else:
            self.__epw_filepath = epw_filepath

    @ddy_filepath.setter
    def ddy_filepath(self,ddy_filepath):
        if ddy_filepath is None:
            ddy_filepath = self.settings()['weather']['ddy'][str(int(self.climate_zone))]
            self.__ddy_filepath = os.path.join(os.path.dirname(__file__),ddy_filepath)
        else:
            self.__ddy_filepath = ddy_filepath

    @id.setter
    def id(self,id):
        if id is None:
            self.__id = self.idf_filepath.split('/')[-1].split('.')[0]
        else:
            self.__id = id

    @occupancy.setter
    def occupancy(self,occupancy):
        if occupancy is None:
            self.__occupancy = self.estimate_occupancy()
        else:
            if occupancy >= 0:
                self.__occupancy = occupancy
            else:
                raise ValueError('occupancy must be >= 0.')
            
    @output_directory.setter
    def output_directory(self,output_directory):
        if output_directory is None:
            output_directory = os.path.join(self.settings()['root_output_directory'],self.id)
            self.__output_directory = os.path.join(os.path.dirname(__file__),output_directory)
        else:
            self.__output_directory = output_directory

    @random_state.setter
    def random_state(self,random_state):
        if random_state is None:
            self.random_state = random.randint(0,100000000)
        else:
            self.__random_state = random_state

    @staticmethod
    def settings():
        return read_json(os.path.join(os.path.dirname(__file__),'.misc/settings.json'))

    def simulate(self,**kwargs):    
        make_directory(self.output_directory)
        self.idf.run(**self.eplaunch_options())

    def eplaunch_options(self):
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
        super().simulate()
        self.__update_timeseries()
        self.__update_attributes(**kwargs)
        self.__update_state_action_space()

    def __update_timeseries(self):
        with open(os.path.join(os.path.dirname(__file__),'.misc/citylearn_simulation_timeseries.sql'),'r') as f:
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

    def __update_attributes(self,**kwargs):
        random.seed(self.random_state)
        # metadata
        attributes = read_json(os.path.join(os.path.dirname(__file__),'.misc/citylearn_templates/attributes.json'))
        attributes['File_Name'] = f'{self.id}.csv'
        attributes['File_Name_Solar_Profile'] = attributes['File_Name_Solar_Profile']
        attributes['Building_Type'] = self.building_type
        attributes['Climate_Zone'] = self.climate_zone
        # PV installation
        pv_sizing = PVSizing(self.__get_roof_area(),self.__get_demand(),climate_zone=self.climate_zone,pv=kwargs.get('pv',None))
        pv_count = min(pv_sizing.size(method='annual_demand'))
        install_pv = random.choice([True,False])
        attributes['Solar_Power_Installed(kW)'] = pv_count*pv_sizing.pv['rating'] if install_pv else 0
        # heat pump
        attributes['Heat_Pump']['nominal_power'] = attributes['Heat_Pump']['nominal_power']
        attributes['Heat_Pump']['technical_efficiency'] = random.uniform(0.2,0.3)
        attributes['Heat_Pump']['t_target_heating'] = random.randint(47,50)
        attributes['Heat_Pump']['t_target_cooling'] = random.randint(7,10)
        # electric water heater
        attributes['Electric_Water_Heater']['nominal_power'] = attributes['Heat_Pump']['nominal_power']
        attributes['Electric_Water_Heater']['efficiency'] = random.uniform(0.9,1.0)
        # chilled water tank
        attributes['Chilled_Water_Tank']['capacity'] = random.randint(1,2)
        attributes['Chilled_Water_Tank']['loss_coefficient'] = random.uniform(0.002,0.01)
        # domestic hot water tank
        attributes['DHW_Tank']['capacity'] = random.randint(0,2)
        attributes['DHW_Tank']['loss_coefficient'] = random.uniform(0.002,0.01)
        # battery
        attributes['Battery']['capacity'] = max(self.__get_demand()) # is this a good way to calculate it?
        attributes['Battery']['efficiency'] = attributes['Battery']['efficiency']
        attributes['Battery']['capacity_loss_coefficient'] = attributes['Battery']['capacity_loss_coefficient']
        attributes['Battery']['loss_coefficient'] = attributes['Battery']['loss_coefficient']
        attributes['Battery']['nominal_power'] = attributes['Battery']['capacity']/2
        attributes['Battery']['power_efficiency_curve'] = attributes['Battery']['power_efficiency_curve']
        attributes['Battery']['capacity_power_curve'] = attributes['Battery']['capacity_power_curve']

        self.attributes = attributes

    def __update_state_action_space(self):
        random.seed(self.random_state)
        state_action_space = read_json(os.path.join(os.path.dirname(__file__),'.misc/citylearn_templates/state_action_space.json'))
        # states
        state_action_space['states']['month'] = True
        state_action_space['states']['day'] = True
        state_action_space['states']['hour'] = True
        state_action_space['states']['daylight_savings_status'] = False
        state_action_space['states']['t_out'] = True
        state_action_space['states']['t_out_pred_6h'] = True
        state_action_space['states']['t_out_pred_12h'] = True
        state_action_space['states']['t_out_pred_24h'] = True
        state_action_space['states']['rh_out'] = True
        state_action_space['states']['rh_out_pred_6h'] = True
        state_action_space['states']['rh_out_pred_12h'] = True
        state_action_space['states']['rh_out_pred_24h'] = True
        state_action_space['states']['diffuse_solar_rad'] = True
        state_action_space['states']['diffuse_solar_rad_pred_6h'] = True
        state_action_space['states']['diffuse_solar_rad_pred_12h'] = True
        state_action_space['states']['diffuse_solar_rad_pred_24h'] = True
        state_action_space['states']['direct_solar_rad'] = True
        state_action_space['states']['direct_solar_rad_pred_6h'] = True
        state_action_space['states']['direct_solar_rad_pred_12h'] = True
        state_action_space['states']['direct_solar_rad_pred_24h'] = True
        state_action_space['states']['t_in'] = True
        state_action_space['states']['avg_unmet_setpoint'] = False
        state_action_space['states']['rh_in'] = True
        state_action_space['states']['non_shiftable_load'] = True
        state_action_space['states']['solar_gen'] = True if self.attributes['Solar_Power_Installed(kW)'] > 0 else False
        state_action_space['states']['cooling_storage_soc'] = True
        state_action_space['states']['dhw_storage_soc'] = True
        state_action_space['states']['electrical_storage_soc'] = True
        state_action_space['states']['net_electricity_consumption'] = True
        state_action_space['states']['carbon_intensity'] = True
        # actions
        state_action_space['actions']['cooling_storage'] = True if self.attributes['Chilled_Water_Tank']['capacity'] > 0 else False
        state_action_space['actions']['dhw_storage'] = True if self.attributes['DHW_Tank']['capacity'] > 0 else False
        state_action_space['actions']['electrical_storage'] = True if self.attributes['Battery']['capacity'] > 0 else False

        self.state_action_space = state_action_space

    def __get_roof_area(self):
        return sum([modeleditor.zonearea_roofceiling(self.idf,zone['Name'],) for zone in self.idf.idfobjects['ZONE']])

    def __get_demand(self):
        return self.timeseries[['Cooling Load [kWh]','DHW Heating [kWh]']].sum(axis=1).tolist()

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
            obj.Flow_Rate_per_Person = ORNLIDF.settings()['flowrate_per_person'][str(int(self.building_type))]
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
            'Zone Thermostat Heating Setpoint Temperature',
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