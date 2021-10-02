from envs.citylearn import CityLearnEnv
import sys
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
        multiplier = 0.8
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

k = int(sys.argv[1])

if k == 0:
    env = CityLearnEnv(None, env_args={"seed":1})
    agent = RBC(env.action_space)
    env.reset()
    done = False
    indx_hour = 11
    R_raw = []
    R_vec = []
    R = []
    Rtot = 0
    while not done:
        state = env.state.copy()
        cos = np.array([[state[0][indx_hour]]])*2-1
        sin = np.array([[state[0][indx_hour-1]]])*2-1
        hour_state = np.arctan2(sin, cos) * 24 / (2*np.pi)
        hour_state[0][0] = round(hour_state[0][0])
        if hour_state <= 0.01:
            hour_state = hour_state + 24
        action = agent.select_action(hour_state)
        reward, done, info = env.step(action)
        R_raw.append(sum(env.raw_reward))
        R_vec.append(env.raw_reward)
        R.append(reward)
        Rtot += reward

    R = np.array(R)
    print("RBC "+"="*100)
    #R_raw = np.array(R_raw)
    #print("raw reward", R_raw.sum(), R_raw.mean(), R_raw.std())
    #R = np.array(R)
    #print("reward", R.sum(), R.mean(), R.std())
    print("cost ", info["cost"])
    print("rew ", R.max(), R.min(), R.mean(), R.std())
elif k == 1:
    env = CityLearnEnv(env_args={"seed":1})
    env.reset()
    done = False
    R_raw = []
    R = []
    R_vec = []
    state = []

    Rtot = 0
    while not done:
        action = [act.sample() for act in env.action_space]
        r, done, info = env.step(action)
        state.append(env.state.copy())
        R_raw.append(sum(env.raw_reward))
        R_vec.append(env.raw_reward)
        R.append(r)
        Rtot += r

    state = np.stack(state)
    print(state.mean(0))
    print(state.std(0))
    f = open("state_info", "wb")
    #np.savez(f, mean=state.mean(0), std=state.std(0))

    print("Random "+"="*100)
    R_raw = np.array(R_raw)
    print("raw reward", R_raw.sum(), R_raw.mean(), R_raw.std())
    R = np.array(R)
    print("reward", R.sum(), R.mean(), R.std())
    print("cost", info["cost"])
elif k == 2:
    env = CityLearnEnv(None, env_args={"seed":1})
    agent = RBC(env.action_space)
    env.reset()
    done = False
    indx_hour = 11
    states = []

    R_raw = []
    R = []
    Rtot = 0
    cnt = 0
    while not done:
        cnt+= 1
        state = env.state.copy()
        states.append(state.copy())
        cos = np.array([[state[0][indx_hour]]])*2-1
        sin = np.array([[state[0][indx_hour-1]]])*2-1
        hour_state = np.arctan2(sin, cos) * 24 / (2*np.pi)
        hour_state[0][0] = round(hour_state[0][0])
        if hour_state <= 0.01:
            hour_state = hour_state + 24
        action = agent.select_action(hour_state)
        reward, done, info = env.step(action)
        R_raw.append(sum(env.raw_reward))
        R.append(reward)
        Rtot += reward
        #if cnt == 100:
        #    import pdb; pdb.set_trace()

    env = CityLearnEnv(env_args={"seed":1})
    env.reset()
    done = False
    #R_raw = []
    #R = []
    #R_vec = []
    R1 = np.array(R_raw)
    print("rbc reward", R1.sum(), R1.mean(), R1.std())


    Rtot = 0
    RR = []
    while not done:
        action = [act.sample() for act in env.action_space]
        r, done, info = env.step(action)
        states.append(env.state.copy())
        R_raw.append(sum(env.raw_reward))
        R.append(r)
        RR.append(r)
        Rtot += r

    states = np.stack(states)
    #import pdb; pdb.set_trace()
    print(states.mean(0))
    print(states.std(0))
    f = open("state_info", "wb")
    np.savez(f, mean=states.mean(0), std=states.std(0))

    R_raw = np.array(R_raw)
    print("raw reward", R_raw.sum(), R_raw.mean(), R_raw.std())
    R = np.array(R)
    print("reward", R.sum(), R.mean(), R.std())
    RR = np.array(RR)
    print("random reward", R.sum(), R.mean(), R.std())
