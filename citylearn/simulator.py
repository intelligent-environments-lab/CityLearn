from copy import deepcopy
import logging
from typing import Tuple
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import Agent

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True

class Simulator:
    def __init__(self, env: CityLearnEnv, agent: Agent, episodes: int = None, store_env_history: bool = None):
        r"""Initialize `Simulator`.

        Parameters
        ----------
        env : CityLearnEnv
            Simulation environment.
        agent : Agent
            Simulation agent(s) for `env.buildings` energy storage charging/discharging management.
        episodes : int, default: 1
            Number of times to simulate until terminal state is reached.
        store_env_history : bool, default: False
            Indicator to store env state at the end of each episode.
        """

        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.store_env_history = store_env_history
        self.__env_history = ()

    @property
    def env(self) -> CityLearnEnv:
        """Simulation environment."""

        return self.__env

    @property
    def agent(self) -> Agent:
        """Simulation agent(s) for `env.buildings` energy storage charging/discharging management."""

        return self.__agent

    @property
    def episodes(self) -> int:
        """Number of times to simulate until terminal state is reached."""

        return self.__episodes

    @property
    def store_env_history(self) -> bool:
        """Indicator to store env state at the end of each episode."""

        return self.__store_env_history

    @property
    def env_history(self) -> Tuple[CityLearnEnv]:
        """List of `env` at the end of each episode."""

        return self.__env_history

    @env.setter
    def env(self, env: CityLearnEnv):
        self.__env = env

    @agent.setter
    def agent(self, agent: Agent):
        if self.env.central_agent:
            assert len(agent.action_space) == 1, 'Only 1 agent is expected when `CityLearnEnv.central_agent` = True.'
        else:
            assert len(agent.action_space) == len(self.env.buildings), 'Length of `Simulator.agent` and `CityLearnEnv.buildings` must be equal when using `citylearn_env.central_agent` = False.'

        self.__agent = agent

    @episodes.setter
    def episodes(self, episodes: int):
        episodes = 1 if episodes is None else int(episodes)
        assert episodes > 0, ':attr:`episodes` must be >= 0.'
        self.__episodes = episodes

    @store_env_history.setter
    def store_env_history(self, store_env_history: bool):
        self.__store_env_history = False if store_env_history is None else store_env_history

    def simulate(self):
        """traditional simulation.
        
        Runs central or multi agent simulation.
        """

        for episode in range(self.episodes):
            observations = self.env.reset()

            while not self.env.done:
                actions = self.agent.select_actions(observations)

                # apply actions to citylearn_env
                next_observations, rewards, _, _ = self.env.step(actions)

                # update
                self.agent.add_to_buffer(observations, actions, rewards, next_observations, done=self.env.done)
                observations = [o for o in next_observations]

                logging.debug(
                    f'Time step: {self.env.time_step}/{self.env.time_steps - 1},'\
                        f' Episode: {episode}/{self.episodes - 1},'\
                            f' Actions: {actions},'\
                                f' Rewards: {rewards}'
                )

            # store environment
            if self.store_env_history:
                self.__env_history += (deepcopy(self.env),)
            else:
                pass