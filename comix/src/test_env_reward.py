from envs.citylearn import CityLearnEnv
import numpy as np

class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day = states[0]
        
        # Daytime: release stored energy
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 9 and hour_day <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        # Early nightime: store DHW and/or cooling energy
        if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
            a = []
            for i in range(len(self.actions_spaces)):
                if len(self.actions_spaces[i].sample()) == 2:
                    a.append([0.091, 0.091])
                else:
                    a.append([0.091])
   
        self.action_tracker.append(a)
        
        return np.array(a)

env = CityLearnEnv(None, env_args={"seed":1})
agent = RBC_Agent(env.action_space)
env.reset()
done = False
R = 0
R_list = []
while not done:
    action = agent.select_action([list(env.env.buildings.values())[0].sim_results['hour'][env.env.time_step]])
    reward, done, _ = env.step(action)
    R += reward
    R_list.append(reward)
print(R)
print(env.env.get_baseline_cost())
R_list = np.array(R_list)
print(R_list.mean(), R_list.std())

env = CityLearnEnv(None, env_args={"seed":1})
done = False
R = 0
R_list = []

for _ in range(10):
    env.reset()
    done = False
    while not done:
        action = [act.sample() for act in env.action_space]
        r, done, info = env.step(action)
        #r, done, info = env.step(agent.select_action(env.get_obs()))
        R += r
        R_list.append(r)

print(R)
print(info)
print(env.env.cost())

R_list = np.array(R_list)
print(R_list.mean(), R_list.std())
