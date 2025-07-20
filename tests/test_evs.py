# Run using python test_evs.py

import sys
sys.path.insert(0, "..")
import citylearn
from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent
# RandomAgent, RLAgent
from citylearn.citylearn import CityLearnEnv

dataset_name = '/mnt/c/Users/Tiago Fonseca/Documents/GitHub/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'

# dataset_name = 'citylearn_challenge_2023_phase_2_local_evaluation'
env = CityLearnEnv(dataset_name, central_agent=True)
model = Agent(env)
model.learn(episodes=1, logging_level=1)

# print cost functions at the end of episode
kpis = model.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')
print(kpis)
