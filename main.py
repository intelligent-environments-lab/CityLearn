#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt

from agent_D4PG import train_params, play_params, RL_Agents
import numpy as np
from csv import DictWriter
import json
from pathlib import Path
import time

from citylearn import  CityLearn

# Extra packages for postprocessing
import matplotlib.dates as dates
import pandas as pd
    
# In[2]:

# Load environment
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
#building_ids = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
building_ids = ["Building_1"]
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = True)

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()


# In[ ]:

# Initialise agent
agents = RL_Agents(None, env, play_params.RANDOM_SEED, False, play_params.CKPT_DIR, play_params.CKPT_FILE)

# Play a saved ckpt of actor network in the environment

# Select many episodes for training. In the final run we will set this value to 1 (the buildings run for one year)
episodes = 1

k, c = 0, 0
cost, cum_reward = {}, {}

for e in range(episodes):
    cum_reward[e] = 0
    rewards = []
    state = env.reset()
    done = False
            
    while not done:
                
        if k%(8760)==0:
            print('hour: '+str(k)+' of '+str(8760*episodes))
        
        # Add batch dimension to single state input, and remove batch dimension from single action output
        action = agents.select_action(state)
        #print(action)
        state, reward, done, _ = env.step(action)
        cum_reward[e] += reward
        rewards.append(reward)
        k+=1
            
    cost[e] = env.cost()
            
    if c%1==0:
        print(cost[e])
    c+=1
                    
env.close() 


# In[ ]:
## POSTPROCESSING

building_number = 'Building_1'

# Convert output to dataframes for easy plotting
time_periods = pd.date_range('2017-01-01 T01:00', '2017-12-31 T23:00', freq='1H')
output = pd.DataFrame(index = time_periods)

output['Electricity demand without storage or generation (kW)'] = env.net_electric_consumption_no_pv_no_storage
output['Electricity demand with PV generation and without storage(kW)'] = env.net_electric_consumption_no_storage
output['Electricity demand with PV generation and using D4PG for storage(kW)'] = env.net_electric_consumption
# Cooling Storage
output['Cooling Demand (kWh)'] = env.buildings[building_number].cooling_demand_building
output['Energy Storage State of Charge - SOC (kWh)'] = env.buildings[building_number].cooling_storage_soc
output['Heat Pump Total Cooling Supply (kW)'] = env.buildings[building_number].cooling_device_to_building + env.buildings[building_number].cooling_device_to_storage
output['Controller Action - Increase or Decrease of SOC (kW)'] = [k[0]*env.buildings[building_number].cooling_storage.capacity for k in [j for j in np.array(agents.action_tracker)]]
# DHW
output['DHW Demand (kWh)'] = env.buildings[building_number].dhw_demand_building
output['Energy Balance of DHW Tank (kWh)'] = -env.buildings[building_number].dhw_storage.energy_balance
output['DHW Heater Total Heating Supply (kWh)'] = env.buildings[building_number].dhw_heating_device.heat_supply
output['Controller Action - Increase or Decrease of SOC (kW)'] = [k[1]*env.buildings[building_number].dhw_storage.capacity for k in [j for j in np.array(agents.action_tracker)]]
output['DHW Heater Electricity Consumption (kWh)'] = env.buildings[building_number].electric_consumption_dhw

output_filtered = output.loc['2017-07-01':'2017-07-05']

fig, ax = plt.subplots(nrows = 3, figsize=(20,12), sharex = True)
output_filtered['Electricity demand without storage or generation (kW)'].plot(ax = ax[0], color='blue', label='Electricity demand without storage or generation (kW)', x_compat=True)
output_filtered['Electricity demand with PV generation and without storage(kW)'].plot(ax = ax[0], color='orange', label='Electricity demand with PV generation and without storage(kW)')
output_filtered['Electricity demand with PV generation and using D4PG for storage(kW)'].plot(ax = ax[0], color = 'green', ls = '--', label='Electricity demand with PV generation and using D4PG for storage(kW)')
ax[0].set_title('(a) - Electricity Demand')
ax[0].set(ylabel="Demand [kW]")
ax[0].legend(loc="upper right")
ax[0].xaxis.set_major_locator(dates.DayLocator())
ax[0].xaxis.set_major_formatter(dates.DateFormatter('\n%d/%m'))
ax[0].xaxis.set_minor_locator(dates.HourLocator(interval=6))
ax[0].xaxis.set_minor_formatter(dates.DateFormatter('%H'))
output_filtered['Cooling Demand (kWh)'].plot(ax = ax[1], color='blue', label='Cooling Demand (kWh)', x_compat=True)
output_filtered['Energy Storage State of Charge - SOC (kWh)'].plot(ax = ax[1], color='orange', label='Energy Storage State of Charge - SOC (kWh)')
output_filtered['Heat Pump Total Cooling Supply (kW)'].plot(ax = ax[1], color = 'green', label='Heat Pump Total Cooling Supply (kW)')
output_filtered['Controller Action - Increase or Decrease of SOC (kW)'].plot(ax = ax[1], color = 'red', label='Controller Action - Increase or Decrease of SOC (kW)')
ax[1].set_title('(b) - Cooling Storage Utilisation')
ax[1].set(ylabel="Power [kW]")
ax[1].legend(loc="upper right")
output_filtered['DHW Demand (kWh)'].plot(ax = ax[2], color='blue', label='DHW Demand (kWh)', x_compat=True)
output_filtered['Energy Balance of DHW Tank (kWh)'].plot(ax = ax[2], color='orange', label='Energy Balance of DHW Tank (kWh)')
output_filtered['DHW Heater Total Heating Supply (kWh)'].plot(ax = ax[2], color = 'green', label='DHW Heater Total Heating Supply (kWh)')
output_filtered['Controller Action - Increase or Decrease of SOC (kW)'].plot(ax = ax[2], color = 'red', label='Controller Action - Increase or Decrease of SOC (kW)')
output_filtered['DHW Heater Electricity Consumption (kWh)'].plot(ax = ax[2], color = 'purple', ls = '--', label='DHW Heater Electricity Consumption (kWh)')
ax[2].set_title('(c) - DWH Storage Utilisation')
ax[2].set(ylabel="Power [kW]")
ax[2].legend(loc="upper right")
# Set minor grid lines
ax[0].xaxis.grid(False) # Just x
ax[0].yaxis.grid(False) # Just x
for j in range(3):
	for xmin in ax[j].xaxis.get_minorticklocs():
		ax[j].axvline(x=xmin, ls='-', color = 'lightgrey')
ax[0].tick_params(direction='out', length=6, width=2, colors='black', top=0, right=0)
plt.setp( ax[0].xaxis.get_minorticklabels(), rotation=0, ha="center" )
plt.setp( ax[0].xaxis.get_majorticklabels(), rotation=0, ha="center" )
# Export Figure
plt.savefig(r"test.jpg", bbox_inches='tight', dpi = 300)
plt.close()

# In[ ]:

# Export Controller Performance Results to csv

def append_dict_as_row(file_name, dict_of_elem, field_names):
	# Open file in append mode
	with open(file_name, 'a+', newline='') as write_obj:
		# Create a writer object from csv module
		dict_writer = DictWriter(write_obj, fieldnames=field_names)
		# Add dictionary as wor in the csv
		dict_writer.writerow(dict_of_elem)

# SET UP RESULTS TABLE
run_results = {}
    
## TABULATE KEY VALIDATION RESULTS---------------------------------------------
run_results['Time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
run_results['Climate'] = climate_zone
run_results['Building'] = building_ids
run_results['Building_Attributes'] = json.load(open(building_attributes))
if env.central_agent == True:
    run_results['Reward_Function'] = 'reward_function_sa'
else:
    run_results['Reward_Function'] = 'reward_function_ma'
run_results['Central_Agent'] = env.central_agent
run_results['Model'] = agents.ckpt
run_results['Train_Episodes'] = train_params.NUM_STEPS_TRAIN
run_results['Ramping'] = cost[0]['ramping']
run_results['1-Load_Factor'] = cost[0]['1-load_factor']
run_results['Average_Daily_Peak'] = cost[0]['average_daily_peak']
run_results['Peak_Demand'] = cost[0]['peak_demand']
run_results['Net_Electricity_Consumption'] = cost[0]['net_electricity_consumption']
run_results['Total'] = cost[0]['total']
run_results['Reward'] = np.sum(rewards)

print("Reward: %02f \n\n" % np.sum(rewards))
	
field_names = ['Time','Climate','Building','Building_Attributes','Reward_Function','Central_Agent','Model',
               'Train_Episodes','Ramping','1-Load_Factor','Average_Daily_Peak','Peak_Demand',
               'Net_Electricity_Consumption','Total','Reward']

append_dict_as_row('test_results.csv', run_results, field_names)

