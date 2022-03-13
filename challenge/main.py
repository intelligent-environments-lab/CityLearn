from copy import deepcopy
import os
from citylearn.citylearn import District
from citylearn.utilities import read_json
from agent import Agent
from reward import Reward

def main():
    # load environment
    settings_filepath = os.path.join(os.path.dirname(__file__),'settings.json')
    settings = read_json(settings_filepath)
    schema = read_json(settings['schema_filepath'])
    schema['central_agent'] = settings['central_agent']
    district = District.load(schema)
    agents = [Agent(index, action_spaces.shape[0], deepcopy(district)) for index, action_spaces in enumerate(district.action_spaces)]

    # run simulation
    # while not district.terminal:
    for i in range(100):
        print(f'Timestep: {district.time_step}/{district.time_steps - 1}')
        states_list = district.states

        # select actions
        results = [agent.select_actions(states) if agent.actions_dimension > 0 else ([], {}) for agent, states in zip(agents, states_list)]
        actions_list, kwargs_list = [r[0] for r in results], [r[1] for r in results]
        district.step(actions_list)
        rewards = [Reward.get(agent.index, deepcopy(district)) if agent.actions_dimension > 0 else None for agent in agents]
        _ = [
            agent.add_to_buffer(states, actions, reward, next_states, done = district.terminal, **kwargs)
            for agent, states, actions, reward, next_states, kwargs
            in zip(agents, states_list, actions_list, rewards, district.states, kwargs_list) if agent.actions_dimension > 0
        ]

if __name__ == '__main__':
    main()
else:
    pass