import json
import os
import shutil
import pandas as pd

source_directory = 'data'
destination_directory = 'data_reward_function_exploration'
building_attributes_update_filepath = 'building_attributes_update.json'
building_state_action_space_filepath = 'buildings_state_action_space.json'
carbon_intensity_filepath = 'data/Climate_Zone_5/carbon_intensity.csv'

# copy original data directory
if os.path.isdir(destination_directory):
    shutil.rmtree(destination_directory)
else:
    pass

destination = shutil.copytree(source_directory,destination_directory)

# adjust attributes and state action space
with open(building_attributes_update_filepath) as f:
        building_attributes_update = json.load(f)

for climate_zone, climate_zone_building_attributes_update in building_attributes_update.items():
    climate_zone_building_attributes_source_filepath = os.path.join(destination_directory,os.path.join(climate_zone,'building_attributes.json'))
    climate_zone_building_attributes_destination_filepath = climate_zone_building_attributes_source_filepath
    climate_zone_building_state_action_space_source_filepath = building_state_action_space_filepath
    climate_zone_building_state_action_space_destination_filepath = os.path.join(destination_directory,os.path.join(climate_zone,'buildings_state_action_space.json'))

    with open(climate_zone_building_attributes_source_filepath) as f:
        climate_zone_building_attributes = json.load(f)

    with open(climate_zone_building_state_action_space_source_filepath) as f:
        climate_zone_building_state_action_space = json.load(f)

    for building in climate_zone_building_attributes:
        climate_zone_building_attributes[building]['Solar_Power_Installed(kW)'] = climate_zone_building_attributes_update[building]['Solar_Power_Installed(kW)']
        climate_zone_building_attributes[building]['Chilled_Water_Tank']['capacity'] = climate_zone_building_attributes_update[building]['Chilled_Water_Tank']['capacity']
        climate_zone_building_attributes[building]['DHW_Tank']['capacity'] = climate_zone_building_attributes_update[building]['DHW_Tank']['capacity']
        climate_zone_building_attributes[building]['Battery']['capacity'] = 0
        climate_zone_building_state_action_space[building]['states']['solar_gen'] = True if climate_zone_building_attributes[building]['Solar_Power_Installed(kW)'] > 0 else False
        climate_zone_building_state_action_space[building]['states']['cooling_storage_soc'] = True if climate_zone_building_attributes[building]['Chilled_Water_Tank']['capacity'] > 0 else False
        climate_zone_building_state_action_space[building]['states']['dhw_storage_soc'] = True if climate_zone_building_attributes[building]['DHW_Tank']['capacity'] > 0 else False
        climate_zone_building_state_action_space[building]['states']['electrical_storage_soc'] = True if climate_zone_building_attributes[building]['Battery']['capacity'] > 0 else False
        climate_zone_building_state_action_space[building]['actions']['cooling_storage'] = climate_zone_building_state_action_space[building]['states']['cooling_storage_soc']
        climate_zone_building_state_action_space[building]['actions']['dhw_storage'] = climate_zone_building_state_action_space[building]['states']['dhw_storage_soc']
        climate_zone_building_state_action_space[building]['actions']['electrical_storage'] = climate_zone_building_state_action_space[building]['states']['electrical_storage_soc']

    with open(climate_zone_building_attributes_destination_filepath,'w') as f:
        json.dump(climate_zone_building_attributes,f,sort_keys=False,indent=2)

    with open(climate_zone_building_state_action_space_destination_filepath,'w') as f:
        json.dump(climate_zone_building_state_action_space,f,sort_keys=False,indent=2)

# set carbon intensity data
carbon_intensity = pd.read_csv(carbon_intensity_filepath)
carbon_intensity['Datetime'] = pd.to_datetime(carbon_intensity['Datetime'])
carbon_intensity['month'] = carbon_intensity['Datetime'].dt.month
carbon_intensity['day'] = carbon_intensity['Datetime'].dt.day
carbon_intensity['hour'] = carbon_intensity['Datetime'].dt.hour
carbon_intensity = carbon_intensity.groupby(['month','day','hour'])[['kg_CO2/kWh']].mean()
carbon_intensity = carbon_intensity.reset_index()
# remove leap year 2/29
ixs = carbon_intensity[(carbon_intensity['month']==2)&(carbon_intensity['day']==29)].index
carbon_intensity = carbon_intensity.drop(ixs)

for climate_zone in [1,2,3,4]:
    filepath = os.path.join(
        destination_directory,
        os.path.join(f'Climate_Zone_{climate_zone}','carbon_intensity.csv')
    )
    carbon_intensity.to_csv(filepath,index=False)