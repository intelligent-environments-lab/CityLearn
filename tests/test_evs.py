"""
Run from tests folder: python3 test_evs.py

Ensures parent repo root is on sys.path so local 'citylearn' package is importable
without installing. Alternative: run from repo root using `python -m tests.test_evs`.
"""
import os
import sys
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
import citylearn
from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent
# RandomAgent, RLAgent
from citylearn.citylearn import CityLearnEnv

dataset_name = '../data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json'

# dataset_name = 'citylearn_challenge_2023_phase_2_local_evaluation'
env = CityLearnEnv(dataset_name, central_agent=True, render=True)
model = Agent(env)
model.learn(episodes=1, logging_level=1)

# print cost functions at the end of episode
kpis = model.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')
print(kpis)

# Print where results were saved when rendering is enabled
if hasattr(env, 'new_folder_path'):
    print(f"Results folder: {env.new_folder_path}")
