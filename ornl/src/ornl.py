from eppy.modeleditor import IDF
from utilities import read_json

class ORNL:
    def __init__(self,epw_filepath,ddy_filepath,idd_filepath=None):
        self.idd_filepath = idd_filepath
        self.epw_filepath = epw_filepath
        self.ddy_filepath = ddy_filepath

    @property
    def idd_filepath(self):
        return self.__idd_filepath
    
    @property
    def epw_filepath(self):
        return self.__epw_filepath
    
    @property
    def ddy_filepath(self):
        return self.__ddy_filepath

    @idd_filepath.setter
    def idd_filepath(self,idd_filepath):
        if idd_filepath is None:
            self.__idd_filepath = read_json('.misc/settings.json')['idd_filepath']
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

    def preprocess(self,idf):
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
            obj.Flow_Rate_per_Person = 0.0169901079552 # [m3/s-person], need to confirm the source of this value
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
        obj.Output_RDD = 'Yes'
        obj.Output_MDD = 'Yes'

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
