from copy import deepcopy
import os
import pandas as pd
import simplejson as json

data_directory = 'data/'
cooling_climate_zones = list(range(1,4))

with open('buildings_state_action_space.json') as f:
    state_action_space = json.load(f) 

for climate_zone_directory in os.listdir(data_directory):
    if climate_zone_directory.startswith('Climate_Zone'):
        climate_zone = int(climate_zone_directory.split('_')[-1])
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
    
    # use same definition as chilled water tank for testing
    for building, attributes in data.items():
        attributes_keys = list(attributes.keys())
        attributes_values = list(attributes.values())
        
        if not 'Hot_Water_Tank' in attributes_keys:
            ix = attributes_keys.index('Chilled_Water_Tank')
            attributes_keys.insert(ix + 1,'Hot_Water_Tank')
            attributes_values.insert(ix + 1,attributes['Chilled_Water_Tank'])
        else:
            pass

        data[building] = {key:value for key, value in zip(attributes_keys,attributes_values)}
            
        if climate_zone in cooling_climate_zones:
            data[building]['Hot_Water_Tank']['capacity'] = 0
        else:
            data[building]['Chilled_Water_Tank']['capacity'] = 0

        with open(filepath,'w') as f:
            json.dump(data,f,ignore_nan=True,sort_keys=False,indent=2)

    # update state action space
    data = deepcopy(state_action_space)

    for building, space in data.items():
        # use same definition as cooling related states and actions for testing
        state_keys = list(space['states'].keys())
        state_values = list(space['states'].values())
        action_keys = list(space['actions'].keys())
        action_values = list(space['actions'].values())

        if not 'heating_storage_soc' in state_keys:
            ix = state_keys.index('cooling_storage_soc')
            state_keys.insert(ix + 1,'heating_storage_soc')
            state_values.insert(ix + 1,space['states']['cooling_storage_soc'])
        else:
            pass
        
        if not 'heating_storage' in action_keys:
            ix = action_keys.index('cooling_storage')
            action_keys.insert(ix + 1,'heating_storage')
            action_values.insert(ix + 1,space['actions']['cooling_storage'])
        else:
            pass

        data[building]['states'] = {key:value for key, value in zip(state_keys,state_values)}
        data[building]['actions'] = {key:value for key, value in zip(action_keys,action_values)}

        if climate_zone in cooling_climate_zones:
            data[building]['states']['cooling_storage_soc'] = True
            data[building]['states']['heating_storage_soc'] = False
            data[building]['actions']['cooling_storage'] = True
            data[building]['actions']['heating_storage'] = False

        else:
            data[building]['states']['cooling_storage_soc'] = False
            data[building]['states']['heating_storage_soc'] = True
            data[building]['actions']['cooling_storage'] = False
            data[building]['actions']['heating_storage'] = True

    filepath = os.path.join(climate_zone_directory,'buildings_state_action_space.json')

    with open(filepath,'w') as f:
        json.dump(data,f,ignore_nan=True,sort_keys=False,indent=2)