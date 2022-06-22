from citylearn.citylearn import CityLearnEnv

if __name__ == '__main__':
    schema_path = '/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/citylearn/CityLearn/data/cc2022_d1/schema.json'
    env = CityLearnEnv(schema=schema_path)
    print("Citylearn environment created")
    all_obs = env.reset()
    num_buildings = len(env.buildings)
    done = False
    tsteps = 0
    while not done:
        actions = [action_space.sample() for action_space in env.action_space]
        all_obs, all_rew, done, all_info = env.step(actions)
        tsteps += 1
        if tsteps % 50 == 0:
            print(f"Time steps: {tsteps}")
    print(f"Episode completed in {tsteps} time steps")