from envs.citylearn import CityLearnEnv
import json
import numpy as np

class RBC:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day = states[0][0]
        multiplier = 0.4 / 0.5
        # Daytime: release stored energy  2*0.08 + 0.1*7 + 0.09
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 7 and hour_day <= 11:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 12 and hour_day <= 15:
            a = [[-0.05 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 16 and hour_day <= 18:
            a = [[-0.11 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 19 and hour_day <= 22:
            a = [[-0.06 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        # Early nightime: store DHW and/or cooling energy
        if hour_day >= 23 and hour_day <= 24:
            a = [[0.085 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        elif hour_day >= 1 and hour_day <= 6:
            a = [[0.1383 * multiplier for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]

        self.action_tracker.append(a)
        return np.array(a, dtype = 'object')

with open('envs/citylearn/buildings_state_action_space.json') as file:
    actions_ = json.load(file)
indx_hour = -1
for obs_name, selected in list(actions_.values())[0]['states'].items():
    indx_hour += 1
    if obs_name=='hour':
        break
    assert indx_hour < len(list(actions_.values())[0]['states'].items()) - 1, "Please, select hour as a state for Building_1 to run the RBC"

env = CityLearnEnv(None, env_args={"seed":1})
#agent = RBC(env.original_action_space)
agent = RBC(env.action_space)
#state = env.env.reset()
env.reset()
done = False
R = 0
R_list = []
indx_hour = 11
while not done:
    #states = env.raw_state.copy()
    state = env.state.copy()
    #import pdb; pdb.set_trace()
    #hour_state = np.array([[state[0][indx_hour]]])
    hour_state = np.array([[state[0][indx_hour]]])
    action = agent.select_action(hour_state)
    reward, done, info = env.step(action)
    #state, reward, done, _ = env.env.step(action)
    #R += sum(reward)
    R += reward

print("RBC...")
print(R)
print(info["cost"])
#a = env.env.cost()
#print(a["total"])
"""
state = env.env.reset()
done = False
rewards_list = []
while not done:
    hour_state = np.array([[state[0][indx_hour]]])
    action = agent.select_action(hour_state)
    #print(action)
    next_state, rewards, done, _ = env.env.step(action)
    state = next_state
    rewards_list.append(rewards)
cost_rbc = env.env.cost()
print(cost_rbc)
"""

"""
env = CityLearnEnv(env_args={"seed":1})
done = False
R = 0
R_list = []

x = env.get_env_info()

env.reset()
done = False
while not done:
    action = [act.sample() for act in env.action_space]
    r, done, info = env.step(action)
    #r, done, info = env.step(agent.select_action(env.get_obs()))
    R += r
    R_list.append(r)

print("Random...")
print(R)
print(info["cost"])
print("env cost", env.env.cost())
"""
