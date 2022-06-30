from citylearn.citylearn import CityLearnEnv

if __name__ == '__main__':
    schema = './citylearn/data/citylearn_challenge_2022_phase_1/schema.json'
    env = CityLearnEnv(schema=schema)
    print("Citylearn environment created")
    for _ in range(2):
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