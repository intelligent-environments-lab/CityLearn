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

    if False:
        #agent = RBC(env.original_action_space)
        agent = RBC(env.action_space)
        #state = env.env.reset()
        env.reset()
        done = False
        R = 0
        R_list = []
        hour_list = []
        indx_hour = 11
        while not done:
            #print(state[0][2])
            state = env.raw_state
            action = agent.select_action(state)
            hour_list.append(state[0][2])

            #import pdb; pdb.set_trace()
            #act = []
            #act.append(action[0])
            #act.append(action[1])
            #act.append(action[2][:2])
            #act.append(action[3][:2])
            #act.append(action[4])
            #state, reward, done, _ = env.env.step(action)
            #state, reward, done, _ = env.env.step(act)
            reward, done, info = env.step(action)
        print("RBC...")
        print(hour_list[:100])
        #print(env.env.cost())
        print(info["cost"])
    else:
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
            #hour_list.append(hour_state[0][0])
            #raw_state = env.raw_state.copy()
            #if np.abs(raw_state[0][2] - hour_state[0][0]) > 0.01:
            #    import pdb; pdb.set_trace()
            #print(state[0][2])
            #print(hour_state)
            action = agent.select_action(hour_state)
            reward, done, info = env.step(action)
            R_raw.append(sum(env.raw_reward))
            R_vec.append(env.raw_reward)
            R.append(reward)
            Rtot += reward
            print(reward)

        print("RBC "+"="*100)
        print("reward", Rtot)
        print("cost ", info["cost"])

        R_raw = np.array(R_raw)
        print(R_raw.mean())
        print(R_raw.std())

        R = np.array(R)
        print(R.mean(), R.std())
        print(R)
        import pdb; pdb.set_trace()
        #print("=====")
        #R_vec = np.array(R_vec)
        #print(R_vec.mean(0))
        #print(R_vec.std(0))
else: 
    env = CityLearnEnv(env_args={"seed":1})
    env.reset()
    done = False
    R_raw = []
    R = []
    R_vec = []
    Rtot = 0
    while not done:
        action = [act.sample() for act in env.action_space]
        r, done, info = env.step(action)
        R_raw.append(sum(env.raw_reward))
        R_vec.append(env.raw_reward)
        R.append(r)
        Rtot += r

    print("Random "+"="*100)
    print("reward", Rtot)
    print("cost", info["cost"])
    R_raw = np.array(R_raw)
    print(R_raw.mean())
    print(R_raw.std())
    print("=====")
    R_vec = np.array(R_vec)
    print(R_vec.mean(0))
    print(R_vec.std(0))

    R= np.array(R)
    print(R.mean(), R.std())
