import os
import pandas as pd
import simplejson as json

data_directory = 'data/'

for climate_zone_directory in os.listdir(data_directory):
    if climate_zone_directory.startswith('Climate_Zone'):
        climate_zone_directory = os.path.join(data_directory,climate_zone_directory)
        
        # add heating load to simulation results
        for filename in os.listdir(climate_zone_directory):
            if filename.startswith('Building_'):
                filepath = os.path.join(climate_zone_directory,filename)
                data = pd.read_csv(filepath)
                # setting to zero for testing purposes
                data['Heating Load [kWh]'] = 0.0
                data.to_csv(filepath,index=False)

            else:
                continue
    
    else:
        continue

    # update building attributes
    filepath = os.path.join(climate_zone_directory,'building_attributes.json')

    with open(filepath) as f:
        data = json.load(f)
    
    # use same definition as chilled water tank for testing]
    for building, attributes in data.items():
        attributes_keys = list(attributes.keys())
        attributes_values = list(attributes.values())
        ix = attributes_keys.index('Chilled_Water_Tank')
        attributes_keys.insert(ix + 1,'Hot_Water_Tank')
        attributes_values.insert(ix + 1,attributes['Chilled_Water_Tank'])
        data[building] = {key:value for key, value in zip(attributes_keys,attributes_values)}

    with open(filepath,'w') as f:
        json.dump(data,f,ignore_nan=True,sort_keys=False,indent=2)

# update state action space
filepath = 'buildings_state_action_space.json'

with open(filepath) as f:
        data = json.load(f)

for building, state_action_space in data.items():
    # use same definition as cooling related states and actions for testing
    state_keys = list(state_action_space['states'].keys())
    state_values = list(state_action_space['states'].values())
    ix = state_keys.index('cooling_storage_soc')
    state_keys.insert(ix + 1,'heating_storage_soc')
    state_values.insert(ix + 1,state_action_space['states']['cooling_storage_soc'])
    
    action_keys = list(state_action_space['actions'].keys())
    action_values = list(state_action_space['actions'].values())
    ix = action_keys.index('cooling_storage')
    action_keys.insert(ix + 1,'heating_storage')
    action_values.insert(ix + 1,state_action_space['actions']['cooling_storage'])

    data[building]['states'] = {key:value for key, value in zip(state_keys,state_values)}
    data[building]['actions'] = {key:value for key, value in zip(action_keys,action_values)}

with open(filepath,'w') as f:
    json.dump(data,f,ignore_nan=True,sort_keys=False,indent=2)