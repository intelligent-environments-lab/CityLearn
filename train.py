#!/usr/bin/env python
# coding: utf-8

# In[1]:

from agent_D4PG import train_params, RL_Agents, Learner, GaussianNoiseGenerator, PrioritizedReplayBuffer
import numpy as np
import random
from pathlib import Path

import tensorflow as tf

import threading

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
env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function, central_agent = True, verbose = 1)

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()


# In[ ]:

tf.compat.v1.reset_default_graph()

# Set random seeds for reproducability
np.random.seed(train_params.RANDOM_SEED)
random.seed(train_params.RANDOM_SEED)
tf.compat.v1.set_random_seed(train_params.RANDOM_SEED)
    
# Initialise prioritised experience replay memory
PER_memory = PrioritizedReplayBuffer(train_params.REPLAY_MEM_SIZE, train_params.PRIORITY_ALPHA)

# Initialise Gaussian noise generator
gaussian_noise = GaussianNoiseGenerator(actions_spaces[0].shape, env.action_space.low, env.action_space.high, train_params.NOISE_SCALE)
            
# Create session
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)  
    
# Create threads for learner process and agent processes       
threads = []
# Create threading events for communication and synchronisation between the learner and agent threads
run_agent_event = threading.Event()
stop_agent_event = threading.Event()
    
# with tf.device('/device:GPU:0'):
# Initialise learner
learner = Learner(env, sess, PER_memory, run_agent_event, stop_agent_event)
# Build learner networks
learner.build_network()
# Build ops to update target networks
learner.build_update_ops()
# Initialise variables (either from ckpt file if given, or from random)
learner.initialise_vars()
# Get learner policy (actor) network params - agent needs these to copy latest policy params periodically
learner_policy_params = learner.actor_net.network_params + learner.actor_net.bn_params
    
threads.append(threading.Thread(target=learner.run))

for n_agent in range(train_params.NUM_AGENTS):
    # Initialise agent and build network
    agent = RL_Agents(sess, env, train_params.RANDOM_SEED, True, n_agent = n_agent)
    # Build op to periodically update agent network params from learner network
    agent.build_update_op(learner_policy_params)
    # Create Tensorboard summaries to save episode rewards
    if train_params.LOG_DIR is not None:
        agent.build_summaries(train_params.LOG_DIR + ('/agent_%02d' % n_agent))
        
    threads.append(threading.Thread(target=agent.run, args=(PER_memory, gaussian_noise, run_agent_event, stop_agent_event)))
    
for t in threads:
    t.start()
        
for t in threads:
    t.join()
    
sess.close()



