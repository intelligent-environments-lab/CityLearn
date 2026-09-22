"""A complete flat-interface example with a custom agent and reward."""

import numpy as np

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction


class ZeroActionAgent(Agent):
    """Minimal controller template; replace the action rule for your study."""

    def predict(self, observations, deterministic=None):
        actions = [
            np.clip(np.zeros(space.shape), space.low, space.high).tolist()
            for space in self.action_space
        ]
        self.actions = actions
        self.next_time_step()
        return actions


class GridImportReward(RewardFunction):
    required_observation_names = ("net_electricity_consumption",)

    def calculate(self, observations):
        rewards = [-max(float(o["net_electricity_consumption"]), 0.0) for o in observations]
        return [sum(rewards)] if self.central_agent else rewards


def run(central_agent=False):
    env = CityLearnEnv(
        "citylearn_challenge_2022_phase_all_plus_evs",
        central_agent=central_agent,
        episode_time_steps=24,
        interface="flat",
        reward_function=GridImportReward,
        render_mode="none",
        random_seed=0,
    )
    try:
        agent = ZeroActionAgent(env)
        observations, info = env.reset(seed=0)
        terminated = truncated = False
        while not (terminated or truncated):
            actions = agent.predict(observations)
            observations, rewards, terminated, truncated, info = env.step(actions)
        kpis = env.evaluate_v2(include_business_as_usual=False)
        print(kpis.loc[kpis["cost_function"] == "district_cost_total_control_eur"])
        return kpis
    finally:
        env.close()


if __name__ == "__main__":
    run()
