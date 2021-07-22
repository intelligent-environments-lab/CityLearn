#!/usr/bin/env python
# coding: utf-8

# In[15]:


# To run this example, move this file to the main directory of this repository
from citylearn import  CityLearn
from pathlib import Path
import numpy as np                                                                                                                                                                                      
import torch
import matplotlib.pyplot as plt
from agents.sac import SAC as Agent


# In[11]:


# Load environment
climate_zone = 5
params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':["Building_"+str(i) for i in [1,2]],
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': (0, 8760-1), 
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': False,
        'save_memory': False }

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
env = CityLearn(**params)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()


# In[13]:


params_agent = {'building_ids':["Building_"+str(i) for i in [1,2]],
                 'buildings_states_actions':'buildings_state_action_space.json', 
                 'building_info':building_info,
                 'observation_spaces':observations_spaces, 
                 'action_spaces':actions_spaces}

# Instantiating the control agent(s)
agents = Agent(**params_agent)

for i in range(12):
    state = env.reset()
    done = False

    action, coordination_vars = agents.select_action(state)    
    R = 0
    while not done:
        next_state, reward, done, _ = env.step(action)
        action_next, coordination_vars_next = agents.select_action(next_state)
        agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
        coordination_vars = coordination_vars_next
        state = next_state
        action = action_next
        R += sum(reward)
    print(f"episode {i+1} reward {R}")
    a = env.cost()
    #import pdb; pdb.set_trace()
    #print(env.cost())
    print(a)
    print()

"""

# In[16]:


sim_period = (0, 8760*4 - 1)
interval = range(sim_period[0], sim_period[1])
plt.figure(figsize=(16,5))
plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
plt.plot(env.net_electric_consumption_no_storage[interval])
plt.plot(env.net_electric_consumption[interval], '--')
plt.xlabel('time (hours)')
plt.ylabel('kW')
plt.legend(['Electricity demand without storage or generation (kW)', 
            'Electricity demand with PV generation and without storage(kW)', 
            'Electricity demand with PV generation and using SAC for storage control (kW)'])


# In[17]:


# Plotting summer operation in the last year
interval = range(8760*3 + 24*30*6, 8760*3 + 24*30*6 + 24*10)
plt.figure(figsize=(16,5))
plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
plt.plot(env.net_electric_consumption_no_storage[interval])
plt.plot(env.net_electric_consumption[interval], '--')
plt.xlabel('time (hours)')
plt.ylabel('kW')
plt.legend(['Electricity demand without storage or generation (kW)', 
            'Electricity demand with PV generation and without storage(kW)', 
            'Electricity demand with PV generation and using RBC for storage(kW)'])


# In[18]:


building_number = 'Building_1'
interval = (range(24*30*6 + 8760*3,24*30*6 + 8760*3 + 24*4))
plt.figure(figsize=(12,8))
plt.plot(env.buildings[building_number].cooling_demand_building[interval])
plt.plot(env.buildings[building_number].cooling_storage_to_building[interval] - env.buildings[building_number].cooling_device_to_storage[interval])
plt.plot(env.buildings[building_number].cooling_device.cooling_supply[interval])
plt.plot(env.electric_consumption_cooling[interval])
plt.plot(env.buildings[building_number].cooling_device.cop_cooling[interval]*100,'--')
plt.plot(env.buildings[building_number].cooling_storage.soc[interval],'--')
plt.xlabel('time (hours)')
plt.ylabel('kW')
plt.legend(['Cooling Demand (kWh)',
            'Energy Balance of Chilled Water Tank (kWh)', 
            'Heat Pump Total Cooling Supply (kWh)', 
            'Heat Pump Electricity Consumption (kWh)',
            'Heat Pump COP x100',
            'Cooling Storage State of Charge (kWh)'])


# In[19]:


building_number = 'Building_9'
interval = range(8760*3 + 24*30*6, 8760*3 + 24*30*6 + 24*4)
plt.figure(figsize=(12,8))
plt.plot(env.buildings[building_number].cooling_storage_soc[interval])
plt.plot(env.buildings[building_number].dhw_storage_soc[interval])
plt.plot(env.buildings[building_number].electrical_storage_soc[interval])
plt.xlabel('time (hours)')
plt.ylabel('kW')
plt.legend(['Cooling Storage Device SoC',
            'Heating Storage Device SoC', 
            'Electrical Storage Device SoC'])


# In[ ]:



"""
