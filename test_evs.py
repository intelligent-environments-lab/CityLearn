# Run using python test_file.py

import sys
sys.path.insert(0, "..")
import citylearn
from citylearn.agents.rbc import BasicRBC as Agent
# RandomAgent, RLAgent
from citylearn.citylearn import CityLearnEnv

# dataset_name = 'citylearn_challenge_2022_phase_all_plus_evs'

dataset_name = 'citylearn_challenge_2022_phase_all_plus_evs'
env = CityLearnEnv(dataset_name, central_agent=True)
model = Agent(env)
model.learn(episodes=1)#, logging_level=1)