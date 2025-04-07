# Run using python test_evs.py
import sys
sys.path.insert(0, "..")
import citylearn
from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent
# RandomAgent, RLAgent
from citylearn.citylearn import CityLearnEnv
dataset_name = 'citylearn_challenge_2022_phase_all_plus_evs'
# dataset_name = 'citylearn_challenge_2023_phase_2_local_evaluation'
env = CityLearnEnv(dataset_name, central_agent=False)
model = Agent(env)
model.learn(episodes=1)#, logging_level=1)
# print cost functions at the end of episode

# Get KPIs and pivot

env.export_final_kpis(model=model)

