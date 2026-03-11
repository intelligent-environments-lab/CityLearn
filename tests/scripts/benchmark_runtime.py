#!/usr/bin/env python3
"""Benchmark CityLearn rollout throughput for selected time resolutions and render modes."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = ROOT / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def run_case(seconds_per_time_step: int, render_mode: str, episode_time_steps: int, seed: int):
    render_directory = Path(tempfile.mkdtemp(prefix=f"citylearn_bench_{render_mode}_"))

    kwargs = {
        "central_agent": True,
        "episode_time_steps": episode_time_steps,
        "seconds_per_time_step": seconds_per_time_step,
        "random_seed": seed,
        "debug_timing": True,
    }

    if render_mode != "none":
        kwargs.update(
            {
                "render_mode": render_mode,
                "render_directory": render_directory,
                "render_session_name": f"bench_{render_mode}_{seconds_per_time_step}s",
            }
        )

    t0 = time.perf_counter()
    from citylearn.citylearn import CityLearnEnv

    env = CityLearnEnv(str(SCHEMA), **kwargs)
    t1 = time.perf_counter()

    observations, _ = env.reset()
    t2 = time.perf_counter()

    action = np.zeros(env.action_space[0].shape[0], dtype="float32")
    step_times = []
    retrieval_total = 0.0
    render_total = 0.0
    export_total = 0.0

    while not env.terminated:
        s0 = time.perf_counter()
        observations, _, terminated, truncated, info = env.step([action])
        s1 = time.perf_counter()
        step_times.append(s1 - s0)
        retrieval_total += float(info.get("building_observations_retrieval_time", 0.0))
        render_total += float(info.get("partial_render_time", 0.0))
        export_total += float(info.get("end_export_time", 0.0))

        if terminated or truncated:
            break

    t3 = time.perf_counter()
    env.close()

    avg_step_ms = 1000.0 * float(np.mean(step_times)) if step_times else 0.0
    p95_step_ms = 1000.0 * float(np.percentile(step_times, 95)) if step_times else 0.0

    return {
        "seconds_per_time_step": seconds_per_time_step,
        "render_mode": render_mode,
        "configured_steps": episode_time_steps,
        "executed_steps": len(step_times),
        "env_init_s": round(t1 - t0, 4),
        "reset_s": round(t2 - t1, 4),
        "rollout_s": round(t3 - t2, 4),
        "avg_step_ms": round(avg_step_ms, 4),
        "p95_step_ms": round(p95_step_ms, 4),
        "obs_retrieval_s": round(retrieval_total, 4),
        "partial_render_s": round(render_total, 4),
        "end_export_s": round(export_total, 4),
        "render_dir": str(render_directory),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        nargs="+",
        type=int,
        default=[5, 60],
        help="Seconds per environment step (default: 5 60).",
    )
    parser.add_argument(
        "--render-modes",
        nargs="+",
        default=["none", "end"],
        choices=["none", "during", "end"],
        help="Render modes to benchmark (default: none end).",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=1200,
        help="Episode time steps for each benchmark case (default: 1200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for seconds in args.seconds:
        for render_mode in args.render_modes:
            result = run_case(seconds, render_mode, args.episode_steps, args.seed)
            print(result)


if __name__ == "__main__":
    main()
