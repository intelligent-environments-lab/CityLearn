import functools
import warnings
from pettingzoo import ParallelEnv

from citylearn.citylearn import CityLearnEnv

def make_citylearn_env(schema):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(schema)
    # # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # # Provides a wide vareity of helpful user errors
    # # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(schema):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = CityLearnPettingZooEnv(schema)
    # env = supersuit.aec_wrappers.pad_observations(env)
    # env = aec_to_parallel(env)
    # env = parallel_to_aec(env)
    return env


class CityLearnPettingZooEnv(ParallelEnv):
    metadata = {"render_modes": [], "is_parallelizable": True}

    def __init__(self, schema):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self.citylearnenv = CityLearnEnv(schema=schema)
        self.possible_agents = [f'building_{r}' for r in range(len(self.citylearnenv.buildings))]


    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=1000)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        agent_num = int(agent.split('_')[1])
        return self.citylearnenv.observation_space[agent_num]

    @functools.lru_cache(maxsize=1000)
    def action_space(self, agent):
        agent_num = int(agent.split('_')[1])
        return self.citylearnenv.action_space[agent_num]

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        all_obs = self.citylearnenv.reset()
        observations = {agent: all_obs[i] for i, agent in enumerate(self.agents) }
        return observations

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            actions_list = [acs.sample() for acs in self.citylearnenv.action_space] 
            warnings.warn("Empty actions array provided, using randomly sampled actions")
        else:
            actions_list = [actions[agent] for agent in self.agents]

        all_obs, all_rew, done, all_info = self.citylearnenv.step(actions_list)

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {a: r for a, r in zip(self.agents, all_rew)}

        self.num_moves += 1
        dones = {agent: done for agent in self.agents}

        observations = {agent: all_obs[i] for i, agent in enumerate(self.agents) }
        infos = {agent: all_info for agent in self.agents}

        if done:
            self.agents = [] # Required feature in pettingzoo

        return observations, rewards, dones, infos

def main():
    schema_path = '/home/dipam/aicrowd/citylearn/cc2022_d1/schema.json'
    citylearn_env = make_citylearn_env(schema=schema_path)

    citylearn_env.reset()

    def random_policy(observation, agent_id):
        return citylearn_env.action_space(agent_id).sample()

    done = False
    tsteps = 0
    
    while not done:
        for agent_id in citylearn_env.agents:
            # agent_action = random_policy(observations[agent_id], agent_id)
            agent_action = random_policy(None, agent_id)
            citylearn_env.step(agent_action)

        observations, rewards, done, infos = citylearn_env.last()

        tsteps += 1
        if tsteps % 50 == 0:
            print(f"Time steps: {tsteps}")
    
    print(f"Episode completed in {tsteps} time steps")

if __name__ == '__main__':
    main()

