import logging
from typing import List
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import Agent

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True

class Simulator:
    def __init__(self, citylearn_env: CityLearnEnv, agents: List[Agent], episodes: int = None):
        r"""Initialize `Simulator`.

        Parameters
        ----------
        citylearn_env : CityLearnEnv
            Simulation environment.
        agents : List[Agent]
            Simulation agents for `citylearn_env.buildings` energy storage charging/discharging management.
        episodes : int
            Number of times to simulate until terminal state is reached.
        """

        self.citylearn_env = citylearn_env
        self.agents = agents
        self.episodes = episodes

    @property
    def citylearn_env(self) -> CityLearnEnv:
        """Simulation environment."""

        return self.__citylearn_env

    @property
    def agents(self) -> List[Agent]:
        """Simulation agents for `citylearn_env.buildings` energy storage charging/discharging management."""

        return self.__agents

    @property
    def episodes(self) -> int:
        """Number of times to simulate until terminal state is reached."""

        return self.__episodes

    @citylearn_env.setter
    def citylearn_env(self, citylearn_env: CityLearnEnv):
        self.__citylearn_env = citylearn_env

    @agents.setter
    def agents(self, agents: List[Agent]):
        if self.citylearn_env.central_agent:
            assert len(agents) == 1, 'Only 1 agent is expected when `citylearn_env.central_agent` = True.'
        else:
            assert len(agents) == len(self.citylearn_env.buildings), 'Length of `agents` and `citylearn_env.buildings` must be equal when using `citylearn_env.central_agent` = False.'

        self.__agents = agents

    @episodes.setter
    def episodes(self, episodes: int):
        episodes = 1 if episodes is None else int(episodes)
        assert episodes > 0, ':attr:`episodes` must be >= 0.'
        self.__episodes = episodes

    def simulate(self):
        """traditional simulation.
        
        Runs central or multi agent simulation.
        """

        for episode in range(self.episodes):
            observations_list = self.citylearn_env.reset()

            while not self.citylearn_env.done:
                actions_list = []

                # select actions
                for agent, observations in zip(self.agents, observations_list):
                    if agent.action_dimension > 0:
                        actions = agent.select_actions(observations)
                        
                    else:
                        actions = []
                    
                    actions_list.append(actions)

                # apply actions to citylearn_env
                next_observations_list, reward_list, _, _ = self.citylearn_env.step(actions_list)

                # update
                for agent, observations, actions, reward, next_observations in zip(self.agents, observations_list, actions_list, reward_list, next_observations_list):
                    if agent.action_dimension > 0:
                        agent.add_to_buffer(observations, actions, reward, next_observations, done = self.citylearn_env.done)
                    else:
                        continue

                observations_list = [o for o in next_observations_list]

                logging.debug(
                    f'Time step: {self.citylearn_env.time_step}/{self.citylearn_env.time_steps - 1},'\
                        f' Episode: {episode}/{self.episodes - 1},'\
                            f' Actions: {actions_list},'\
                                f' Rewards: {reward_list}'
                )