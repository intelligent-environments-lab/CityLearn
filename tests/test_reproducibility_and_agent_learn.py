from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent
from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def test_agent_learn_handles_single_timestep_episode_without_calling_step(monkeypatch):
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=1, render_mode="none", random_seed=7)
    agent = Agent(env)

    step_called = False

    def _unexpected_step(_actions):
        nonlocal step_called
        step_called = True
        raise AssertionError("env.step() should not be called when episode is terminal after reset.")

    monkeypatch.setattr(env, "step", _unexpected_step)

    try:
        agent.learn(episodes=1, deterministic=True, logging_level=40)
        assert not step_called
    finally:
        env.close()


def _run_rbc_episode(render_mode: str, output_root: Path, seed: int = 7):
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=128,
        render_mode=render_mode,
        render_directory=output_root,
        random_seed=seed,
    )
    agent = Agent(env)

    try:
        observations, _ = env.reset()
        rewards = []

        while not env.terminated:
            actions = agent.predict(observations, deterministic=True)
            observations, reward, terminated, truncated, _ = env.step(actions)
            rewards.append(np.array(reward, dtype="float64").reshape(-1))

            if terminated or truncated:
                break

        reward_trace = np.vstack(rewards) if rewards else np.zeros((0, len(env.action_space)), dtype="float64")
        ev_soc_trace = {
            ev.name: np.array(ev.battery.soc[: env.time_step + 1], dtype="float64")
            for ev in env.electric_vehicles
        }
        return reward_trace, ev_soc_trace
    finally:
        env.close()


def test_same_seed_is_reproducible_across_render_modes_without_global_numpy_seed(tmp_path):
    during_rewards, during_ev_soc = _run_rbc_episode("during", tmp_path / "during")
    end_rewards, end_ev_soc = _run_rbc_episode("end", tmp_path / "end")

    assert during_rewards.shape == end_rewards.shape
    assert np.allclose(during_rewards, end_rewards)
    assert set(during_ev_soc.keys()) == set(end_ev_soc.keys())

    for ev_name in during_ev_soc:
        assert during_ev_soc[ev_name].shape == end_ev_soc[ev_name].shape
        assert np.allclose(during_ev_soc[ev_name], end_ev_soc[ev_name])
