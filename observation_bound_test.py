from citylearn.citylearn import CityLearnEnv
    
schema = 'citylearn_challenge_2022_phase_1'
env = CityLearnEnv(schema=schema)
print("Citylearn environment created")
all_obs = env.reset()
num_buildings = len(env.buildings)
done = False
tsteps = 0
while not done:
    actions = [action_space.sample() for action_space in env.action_space]
    all_obs, all_rew, done, all_info = env.step(actions)
    for i, obs in enumerate(all_obs):
        if not env.observation_space[i].contains(obs): ###### This should not trigger
            for n, l, h, v in zip(env.buildings[i].active_observations, env.observation_space[i].low, env.observation_space[i].high, obs):
                try:
                    assert l <= v <= h
                except AssertionError:
                    print(n, l, v, h, l <= v <= h)
            
            assert False
    tsteps += 1
    if tsteps % 50 == 0:
        print(f"Time steps: {tsteps}")
print(f"Episode completed in {tsteps} time steps")