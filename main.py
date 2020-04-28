#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt

from agent_D4PG import train_params, play_params, Agent
import numpy as np
from csv import DictWriter
import json
from pathlib import Path
import time

from citylearn import  CityLearn
    
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
agent = Agent(None, env, play_params.RANDOM_SEED, False, play_params.CKPT_DIR, play_params.CKPT_FILE)

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
        action = agent.sess.run(agent.actor_net.output, {agent.state_ph:np.expand_dims(state, 0)})[0]
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

# Plotting winter operation
interval = range(0,8759)
plt.figure(figsize=(16,5))
plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
plt.plot(env.net_electric_consumption_no_storage[interval])
plt.plot(env.net_electric_consumption[interval], '--')
plt.xlabel('time (hours)')
plt.ylabel('kW')
plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using D4PG for storage(kW)'])

# In[ ]:

# Plotting winter operation
interval = range(5000,5050)
plt.figure(figsize=(16,5))
plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
plt.plot(env.net_electric_consumption_no_storage[interval])
plt.plot(env.net_electric_consumption[interval], '--')
plt.xlabel('time (hours)')
plt.ylabel('kW')
plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using D4PG for storage(kW)'])

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
run_results['Model'] = agent.ckpt
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

