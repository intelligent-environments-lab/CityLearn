from csv import DictWriter
import json
import time
from matplotlib import dates
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# RESULT TABLE METHODS

def append_dict_as_row(file_name, dict_of_elem, field_names):
	# Open file in append mode
	with open(file_name, 'a+', newline='') as write_obj:
		# Create a writer object from csv module
		dict_writer = DictWriter(write_obj, fieldnames=field_names)
		# Add dictionary as wor in the csv
		dict_writer.writerow(dict_of_elem)

def tabulate_table(env, timer, algo, agent, climate_zone, building_ids, building_attributes, parent_dir, num_episodes, episode_scores):
    run_results = {}
    ## TABULATE KEY VALIDATION RESULTS---------------------------------------------
    run_results['Time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    run_results['Time_Training'] = timer
    run_results['Time_Training_per_Step'] = timer / (num_episodes*8759)
    run_results['Climate'] = climate_zone
    run_results['Building'] = building_ids
    run_results['Building_Attributes'] = json.load(open(building_attributes))
    if env.central_agent == True:
        run_results['Reward_Function'] = 'reward_function_sa'
    else:
        run_results['Reward_Function'] = 'reward_function_ma'
    run_results['Central_Agent'] = env.central_agent
    run_results['Model'] = ''
    run_results['Algorithm'] = algo
    run_results['Directory'] = parent_dir
    run_results['Train_Episodes'] = num_episodes
    run_results['Ramping'] = env.cost()['ramping']
    run_results['1-Load_Factor'] = env.cost()['1-load_factor']
    run_results['Average_Daily_Peak'] = env.cost()['average_daily_peak']
    run_results['Peak_Demand'] = env.cost()['peak_demand']
    run_results['Net_Electricity_Consumption'] = env.cost()['net_electricity_consumption']
    run_results['Total'] = env.cost()['total']
    run_results['Reward'] = episode_scores[-1]
    run_results['Learning_Rate'] = agent.lr
    run_results['Gamma'] = agent.gamma
    run_results['Tau'] = agent.tau
    run_results['Replay_Size'] = agent.replay_size
    run_results['Batch_Size'] = agent.batch_size
    run_results['Neural_Size'] = agent.hidden_size
    run_results['Alpha'] = agent.alpha
        
    field_names = ['Time','Time_Training','Time_Training_per_Step','Climate','Building','Building_Attributes',
                'Reward_Function','Central_Agent','Model', 'Algorithm', 'Directory',
                'Train_Episodes','Ramping','1-Load_Factor','Average_Daily_Peak','Peak_Demand',
                'Net_Electricity_Consumption','Total','Reward','Learning_Rate','Gamma','Tau',
                'Replay_Size','Batch_Size','Neural_Size','Alpha']

    append_dict_as_row('test_results.csv', run_results, field_names)

# GRAPH RESULTS METHODS

# Graphing for District Behaviour
def graph_total(env, agent, parent_dir, start_date, end_date, algo='SAC'):
    # Convert output to dataframes for easy plotting
    time_periods = pd.date_range('2017-01-01 T01:00', '2017-12-31 T23:00', freq='1H')
    output = pd.DataFrame(index = time_periods)

    # Extract building behaviour
    output['Electricity demand without storage or generation (kW)'] = env.net_electric_consumption_no_pv_no_storage[-8759:]
    output['Electricity demand with PV generation and without storage(kW)'] = env.net_electric_consumption_no_storage[-8759:]
    output['Electricity demand with PV generation and using {} for storage(kW)'.format(algo)] = env.net_electric_consumption[-8759:]

    output_filtered = output.loc[start_date:end_date]

    # Create plot showing electricity demand profile with RL agent, cooling storage behaviour and DHW storage behaviour
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(20,12), squeeze = False)
    output_filtered['Electricity demand without storage or generation (kW)'].plot(ax = ax[0,0], color='blue', label='Electricity demand without storage or generation (kW)')
    output_filtered['Electricity demand with PV generation and without storage(kW)'].plot(ax = ax[0,0], color='orange', label='Electricity demand with PV generation and without storage(kW)')
    output_filtered['Electricity demand with PV generation and using {} for storage(kW)'.format(algo)].plot(ax = ax[0,0], color = 'green', ls = '--', label='Electricity demand with PV generation and using {} for storage(kW)'.format(algo))
    ax[0,0].set_title('District Electricity Demand')
    ax[0,0].set(ylabel="Demand [kW]")
    ax[0,0].legend(loc="upper right")
    ax[0,0].xaxis.set_major_locator(dates.DayLocator())
    ax[0,0].xaxis.set_major_formatter(dates.DateFormatter('\n%d/%m'))
    #ax[0,0].xaxis.set_minor_locator(dates.HourLocator(interval=6))
    #ax[0,0].xaxis.set_minor_formatter(dates.DateFormatter('%H'))
    # Set minor grid lines
    ax[0,0].xaxis.grid(False) # Just x
    ax[0,0].yaxis.grid(False) # Just x
    for xmin in ax[0,0].xaxis.get_minorticklocs():
        ax[0,0].axvline(x=xmin, ls='-', color = 'lightgrey')
    #ax[0,0].tick_params(direction='out', length=6, width=2, colors='black', top=0, right=0)
    #plt.setp( ax[0,0].xaxis.get_minorticklabels(), rotation=0, ha="center" )
    plt.setp( ax[0,0].xaxis.get_majorticklabels(), rotation=0, ha="center" )
    # Export Figure
    plt.savefig(parent_dir + r"district.jpg", bbox_inches='tight', dpi = 300)
    plt.close()

# Graphing for Individual Buildings
def graph_building(building_number, env, agent, parent_dir, start_date, end_date, action_index, algo='SAC'):
    # Convert output to dataframes for easy plotting
    time_periods = pd.date_range('2017-01-01 T01:00', '2017-12-31 T23:00', freq='1H')
    output = pd.DataFrame(index = time_periods)

    # Extract building behaviour
    output['Electricity demand for building {} without storage or generation (kW)'.format(building_number)] = env.buildings[building_number].net_electric_consumption_no_pv_no_storage[-8759:]
    output['Electricity demand for building {} with PV generation and without storage(kW)'.format(building_number)] = env.buildings[building_number].net_electric_consumption_no_storage[-8759:]
    output['Electricity demand for building {} with PV generation and using {} for storage(kW)'.format(building_number, algo)] = env.buildings[building_number].net_electric_consumption[-8759:]
    # Cooling Storage
    output['Cooling Demand (kWh)'] = env.buildings[building_number].cooling_demand_building[-8759:]
    output['Energy Storage State of Charge - SOC (kWh)'] = env.buildings[building_number].cooling_storage_soc[-8759:]
    output['Heat Pump Total Cooling Supply (kW)'] = env.buildings[building_number].cooling_device_to_building[-8759:] + env.buildings[building_number].cooling_device_to_storage[-8759:]
    if env.central_agent == False:
        # THIS IS WRONG - CURRENTLY IT ONLY PLOTS THE ACTIONS OF BUILDING 1!!!! TO FIX
        output['Cooling Action - Increase or Decrease of SOC (kW)'] = [k[0][0]*env.buildings[building_number].cooling_storage.capacity for k in [j for j in np.array(agent.action_tracker[-8759:])]]
    else:
        output['Cooling Action - Increase or Decrease of SOC (kW)'] = [k[action_index]*env.buildings[building_number].cooling_storage.capacity for k in [j for j in np.array(agent.action_tracker[-8759:])]]
    if building_number != 'Building_3' and building_number != 'Building_4':
        # DHW
        output['DHW Demand (kWh)'] = env.buildings[building_number].dhw_demand_building[-8759:]
        #output['Energy Balance of DHW Tank (kWh)'] = -env.buildings[building_number].dhw_storage.energy_balance[-8759:]
        output['Energy Balance of DHW Tank (kWh)'] = env.buildings[building_number].dhw_storage_soc[-8759:]
        output['DHW Heater Total Heating Supply (kWh)'] = env.buildings[building_number].dhw_heating_device.heat_supply[-8759:]
        if env.central_agent == False:
            output['DHW Action - Increase or Decrease of SOC (kW)'] = [k[0][1]*env.buildings[building_number].dhw_storage.capacity for k in [j for j in np.array(agent.action_tracker[-8759:])]]
        else:
            output['DHW Action - Increase or Decrease of SOC (kW)'] = [k[action_index+1]*env.buildings[building_number].dhw_storage.capacity for k in [j for j in np.array(agent.action_tracker[-8759:])]]
        output['DHW Heater Electricity Consumption (kWh)'] = env.buildings[building_number].electric_consumption_dhw[-8759:]

    output_filtered = output.loc[start_date:end_date]

    # Create plot showing electricity demand profile with RL agent, cooling storage behaviour and DHW storage behaviour
    fig, ax = plt.subplots(nrows = 3, figsize=(20,12), sharex = True) if building_number != 'Building_3' and building_number != 'Building_4' else plt.subplots(nrows = 2, figsize=(20,8), sharex = True)
    output_filtered['Electricity demand for building {} without storage or generation (kW)'.format(building_number)].plot(ax = ax[0], color='blue', label='Electricity demand without storage or generation (kW)', x_compat=True)
    output_filtered['Electricity demand for building {} with PV generation and without storage(kW)'.format(building_number)].plot(ax = ax[0], color='orange', label='Electricity demand with PV generation and without storage(kW)')
    output_filtered['Electricity demand for building {} with PV generation and using {} for storage(kW)'.format(building_number, algo)].plot(ax = ax[0], color = 'green', ls = '--', label='Electricity demand with PV generation and using {} for storage(kW)'.format(algo))
    ax[0].set_title('(a) - {} Electricity Demand'.format(building_number))
    ax[0].set(ylabel="Demand [kW]")
    ax[0].legend(loc="upper right")
    ax[0].xaxis.set_major_locator(dates.DayLocator())
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('\n%d/%m'))
    ax[0].xaxis.set_minor_locator(dates.HourLocator(interval=6))
    ax[0].xaxis.set_minor_formatter(dates.DateFormatter('%H'))
    output_filtered['Cooling Demand (kWh)'].plot(ax = ax[1], color='blue', label='Cooling Demand (kWh)', x_compat=True)
    output_filtered['Energy Storage State of Charge - SOC (kWh)'].plot(ax = ax[1], color='orange', label='Energy Storage State of Charge - SOC (kWh)')
    output_filtered['Heat Pump Total Cooling Supply (kW)'].plot(ax = ax[1], color = 'green', label='Heat Pump Total Cooling Supply (kW)')
    output_filtered['Cooling Action - Increase or Decrease of SOC (kW)'].plot(ax = ax[1], color = 'red', label='Controller Action - Increase or Decrease of SOC (kW)')
    ax[1].set_title('(b) - {} Cooling Storage Utilisation'.format(building_number))
    ax[1].set(ylabel="Power [kW]")
    ax[1].legend(loc="upper right")
    if building_number != 'Building_3' and building_number != 'Building_4':
        output_filtered['DHW Demand (kWh)'].plot(ax = ax[2], color='blue', label='DHW Demand (kWh)', x_compat=True)
        output_filtered['Energy Balance of DHW Tank (kWh)'].plot(ax = ax[2], color='orange', label='Energy Balance of DHW Tank (kWh)')
        output_filtered['DHW Heater Total Heating Supply (kWh)'].plot(ax = ax[2], color = 'green', label='DHW Heater Total Heating Supply (kWh)')
        output_filtered['DHW Action - Increase or Decrease of SOC (kW)'].plot(ax = ax[2], color = 'red', label='Controller Action - Increase or Decrease of SOC (kW)')
        output_filtered['DHW Heater Electricity Consumption (kWh)'].plot(ax = ax[2], color = 'purple', ls = '--', label='DHW Heater Electricity Consumption (kWh)')
        ax[2].set_title('(c) - {} DWH Storage Utilisation'.format(building_number))
        ax[2].set(ylabel="Power [kW]")
        ax[2].legend(loc="upper right")
    # Set minor grid lines
    ax[0].xaxis.grid(False) # Just x
    ax[0].yaxis.grid(False) # Just x
    if building_number != 'Building_3' and building_number != 'Building_4':
        for j in range(3):
            for xmin in ax[j].xaxis.get_minorticklocs():
                ax[j].axvline(x=xmin, ls='-', color = 'lightgrey')
    else:
        for j in range(2):
            for xmin in ax[j].xaxis.get_minorticklocs():
                ax[j].axvline(x=xmin, ls='-', color = 'lightgrey')
    ax[0].tick_params(direction='out', length=6, width=2, colors='black', top=0, right=0)
    plt.setp( ax[0].xaxis.get_minorticklabels(), rotation=0, ha="center" )
    plt.setp( ax[0].xaxis.get_majorticklabels(), rotation=0, ha="center" )
    # Export Figure
    plt.savefig(parent_dir + r"train" + "{}.jpg".format(building_number[-1]), bbox_inches='tight', dpi = 300)
    plt.close()
    
    # Plot action history over training - currently just last episode is plotted
    fig, ax = plt.subplots(nrows = 2, figsize=(20,12), sharex = True) if building_number != 'Building_3' and building_number != 'Building_4' else plt.subplots(nrows = 1, figsize=(20,6), sharex = True)
    if building_number != 'Building_3' and building_number != 'Building_4':
        output['Cooling Action - Increase or Decrease of SOC (kW)'].plot(ax = ax[0], color='blue', label='Cooling Demand (kWh)')
        ax[0].set_title('(a) - Cooling Storage Utilisation')
        ax[0].set(ylabel="Power [kW]")
        ax[0].legend(loc="upper right")
    else:
        output['Cooling Action - Increase or Decrease of SOC (kW)'].plot(ax = ax, color='blue', label='Cooling Demand (kWh)')
        ax.set_title('(a) - Cooling Storage Utilisation')
        ax.set(ylabel="Power [kW]")
        ax.legend(loc="upper right")
    if building_number != 'Building_3' and building_number != 'Building_4':
        output['DHW Action - Increase or Decrease of SOC (kW)'].plot(ax = ax[1], color='blue', label='DHW Demand (kWh)')
        ax[1].set_title('(b) - DWH Storage Utilisation')
        ax[1].set(ylabel="Power [kW]")
        ax[1].legend(loc="upper right")
    # Set minor grid lines
    if building_number != 'Building_3' and building_number != 'Building_4':
        ax[0].xaxis.grid(False) # Just x
        ax[0].yaxis.grid(False) # Just x
    else:
        ax.xaxis.grid(False) # Just x
        ax.yaxis.grid(False) # Just x
    if building_number != 'Building_3' and building_number != 'Building_4':
        for j in range(2):
            for xmin in ax[j].xaxis.get_minorticklocs():
                ax[j].axvline(x=xmin, ls='-', color = 'lightgrey')
    else:
        for xmin in ax.xaxis.get_minorticklocs():
            ax.axvline(x=xmin, ls='-', color = 'lightgrey')
    if building_number != 'Building_3' and building_number != 'Building_4':       
        ax[0].tick_params(direction='out', length=6, width=2, colors='black', top=0, right=0)
        plt.setp( ax[0].xaxis.get_minorticklabels(), rotation=0, ha="center" )
        plt.setp( ax[0].xaxis.get_majorticklabels(), rotation=0, ha="center" )
    else:
        ax.tick_params(direction='out', length=6, width=2, colors='black', top=0, right=0)
        plt.setp( ax.xaxis.get_minorticklabels(), rotation=0, ha="center" )
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=0, ha="center" )
    # Export Figure
    plt.savefig(parent_dir + r"actions" + "{}.jpg".format(building_number[-1]), bbox_inches='tight', dpi = 300)
    plt.close()
    