"""Export a complete public-data example for CityLearn UI."""

from pathlib import Path

from citylearn.agents.baseline import BusinessAsUsualAgent
from citylearn.citylearn import CityLearnEnv


def run(output_directory="outputs/ui_exports", session_name="bau_48h"):
    env = CityLearnEnv(
        "citylearn_challenge_2022_phase_all_plus_evs",
        episode_time_steps=48,
        interface="flat",
        render_mode="end",
        render_directory=Path(output_directory).resolve(),
        render_session_name=session_name,
        random_seed=0,
    )
    try:
        agent = BusinessAsUsualAgent(env)
        observations, info = env.reset(seed=0)
        terminated = truncated = False
        while not (terminated or truncated):
            actions = agent.predict(observations)
            observations, rewards, terminated, truncated, info = env.step(actions)

        output = Path(env.new_folder_path).resolve()
        print(f"Import this folder in CityLearn UI: {output}")
        return output
    finally:
        env.close()


if __name__ == "__main__":
    run()
