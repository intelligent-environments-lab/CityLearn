#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To run this example, move this file to the main directory of this repository
from copy import deepcopy
from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from agents.rbc import RBC
import torch
import time
import json
from tqdm import tqdm


def gen(noise):
    climate_zone = 5
    sim_period = (0, 8760-1)
    costs = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
    params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
            'building_attributes':'building_attributes.json', 
            'weather_file':'weather_data.csv', 
            'solar_profile':'solar_generation_1kW.csv', 
            'carbon_intensity':'carbon_intensity.csv',
            'building_ids':["Building_"+str(i) for i in [1,2,3,4,5,6,7,8,9]],
            'buildings_states_actions':'buildings_state_action_space.json', 
            'simulation_period': sim_period, 
            'cost_function': costs, 
            'central_agent': False,
            'save_memory': False }

    env = CityLearn(**params)
    observations_spaces, actions_spaces = env.get_state_action_spaces()

    # Instantiating the control agent(s)
    agents = RBC(actions_spaces)

    # Finding which state 
    with open('buildings_state_action_space.json') as file:
        actions_ = json.load(file)

    indx_hour = -1
    for obs_name, selected in list(actions_.values())[0]['states'].items():
        indx_hour += 1
        if obs_name=='hour':
            break
        assert indx_hour < len(list(actions_.values())[0]['states'].items()) - 1, "Please, select hour as a state for Building_1 to run the RBC"

    data = {}
    data['observation_space'] = observations_spaces
    data['action_space'] = actions_spaces
    data['costs'] = costs
    data['data'] = {'electric_consumption': [], 'noise': [], 'c': []}

    n = 1000
    t = time.time()
    #pbar = tqdm(total=n*11)
    #for noise in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    for i in range(n):
        #S, A = [[] for _ in range(9)], [[] for _ in range(9)] # 9 agents
        EC = []

        state = env.reset()

        done = False
        rewards_list = []
        start = time.time()
        while not done:
            hour_state = np.array([[state[0][indx_hour]]])
            if np.random.rand() < noise:
                action = [act.sample() for act in actions_spaces]
            else:
                action = agents.select_action(hour_state)
            next_state, rewards, done, _ = env.step(action)
            #for ai in range(9):
            #    S[ai].append(torch.from_numpy(state[ai]))
            #    A[ai].append(torch.from_numpy(np.array(action[ai])))
            state = next_state
            EC.append(torch.FloatTensor(rewards))
            #rewards_list.append(rewards)

        cost = env.cost()
        #for ai in range(9):
        #    S[ai] = torch.stack(S[ai]).float()
        #    A[ai] = torch.stack(A[ai]).float()
        #data['data']['s'].append(S)
        #data['data']['a'].append(A)
        import pdb; pdb.set_trace()
        data['data']['electric_consumption'].append(torch.stack(EC))
        cc = []
        for c in costs:
            cc.append(cost[c])
        data['data']['c'].append(torch.FloatTensor(cc))
    torch.save(data, f"reward_data/data{n}_noise{noise}.pt")

#from joblib import Parallel, delayed
#t = time.time()
#results = Parallel(n_jobs=11)(delayed(gen)(noise) for noise in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
#print(time.time() - t)
import sys
noise = float(sys.argv[1])
print("noise is ", noise)
gen(noise)
