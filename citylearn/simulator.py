import logging
from typing import List
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import Agent

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True

class Simulator:
    def __init__(self, citylearn_env: CityLearnEnv, agent: Agent, episodes: int = None):
        r"""Initialize `Simulator`.

        Parameters
        ----------
        citylearn_env : CityLearnEnv
            Simulation environment.
        agent : Agent
            Simulation agent(s) for `citylearn_env.buildings` energy storage charging/discharging management.
        episodes : int
            Number of times to simulate until terminal state is reached.
        """

        self.citylearn_env = citylearn_env
        self.agent = agent
        self.episodes = episodes

    @property
    def citylearn_env(self) -> CityLearnEnv:
        """Simulation environment."""

        return self.__citylearn_env

    @property
    def agent(self) -> Agent:
        """Simulation agent(s) for `citylearn_env.buildings` energy storage charging/discharging management."""

        return self.__agent

    @property
    def episodes(self) -> int:
        """Number of times to simulate until terminal state is reached."""

        return self.__episodes

    @citylearn_env.setter
    def citylearn_env(self, citylearn_env: CityLearnEnv):
        self.__citylearn_env = citylearn_env

    @agent.setter
    def agent(self, agent: Agent):
        if self.citylearn_env.central_agent:
            assert len(agent.action_space) == 1, 'Only 1 agent is expected when `CityLearnEnv.central_agent` = True.'
        else:
            assert len(agent.action_space) == len(self.citylearn_env.buildings), 'Length of `Simulator.agent` and `CityLearnEnv.buildings` must be equal when using `citylearn_env.central_agent` = False.'

        self.__agent = agent

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
            observations = self.citylearn_env.reset()

            while not self.citylearn_env.done:
                actions = self.agent.select_actions(observations)

                # apply actions to citylearn_env
                next_observations, rewards, _, _ = self.citylearn_env.step(actions)

                # update
                self.agent.add_to_buffer(observations, actions, rewards, next_observations, done=self.citylearn_env.done)
                observations = [o for o in next_observations]

                logging.debug(
                    f'Time step: {self.citylearn_env.time_step}/{self.citylearn_env.time_steps - 1},'\
                        f' Episode: {episode}/{self.episodes - 1},'\
                            f' Actions: {actions},'\
                                f' Rewards: {rewards}'
                )