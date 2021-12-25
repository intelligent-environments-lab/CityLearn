import json
import os
import shutil
import pandas as pd

source_directory = 'data'
destination_directory = 'data_reward_function_exploration'
building_attributes_update_filepath = 'building_attributes_update.json'
carbon_intensity_filepath = 'data/Climate_Zone_5/carbon_intensity.csv'

# copy original data directory
if os.path.isdir(destination_directory):
    shutil.rmtree(destination_directory)
else:
    pass

destination = shutil.copytree(source_directory,destination_directory)

# adjust attributes
with open(building_attributes_update_filepath) as f:
        building_attributes_update = json.load(f)

for climate_zone, climate_zone_building_attributes_update in building_attributes_update.items():
    filepath = os.path.join(
        destination_directory,
        os.path.join(climate_zone,'building_attributes.json')
    )
    with open(filepath) as f:
        climate_zone_building_attributes = json.load(f)

    for (_, building_attributes), (_, building_attributes_update) in zip(climate_zone_building_attributes.items(), climate_zone_building_attributes_update.items()):
        building_attributes['Solar_Power_Installed(kW)'] = building_attributes_update['Solar_Power_Installed(kW)']
        building_attributes['Chilled_Water_Tank']['capacity'] = building_attributes_update['Chilled_Water_Tank']['capacity']
        building_attributes['DHW_Tank']['capacity'] = building_attributes_update['DHW_Tank']['capacity']
        building_attributes['Battery']['capacity'] = 0

    with open(filepath,'w') as f:
        json.dump(climate_zone_building_attributes,f,sort_keys=False,indent=2)

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