import numpy as np
from citylearn.reward import Reward
from citylearn.controller.rlc import SAC
from citylearn.citylearn import District

schema_filepath = 'data/schema.json'

# load environment
district = District.load(schema_filepath)
agent_kwargs = {}
agents = [
    SAC(action_spaces, observation_spaces, encoders = observation_encoders, **agent_kwargs) 
    for action_spaces, observation_spaces, observation_encoders 
    in zip(district.action_spaces, district.observation_spaces, district.observation_encoders)
]

while not district.terminal:
    print(f'\rTimestep: {district.time_step}/{district.time_steps - 1}',end='')
    states_list = district.states
    actions_list = [agent.select_actions(states) for agent, states in zip(agents, states_list)]
    district.step(actions_list)
    rewards  = [district.net_electricity_consumption[-1] for _ in agents]
    next_state_list = district.states
    _ = [
        agent.add_to_buffer(states, actions, reward, next_states) 
        for agent, states, actions, reward, next_states 
        in zip(agents, states_list, actions_list, rewards, district.states)
    ]