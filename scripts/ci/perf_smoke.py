#!/usr/bin/env python3
"""Lightweight performance smoke test for CI.

This is meant to catch major performance regressions in the rollout loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citylearn.citylearn import CityLearnEnv  # noqa: E402

SCHEMA = ROOT / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def run_case(render_mode: str, episode_steps: int, seconds_per_time_step: int, seed: int) -> dict:
    render_dir = Path(tempfile.mkdtemp(prefix=f"citylearn_perf_{render_mode}_"))

    kwargs = {
        "central_agent": True,
        "episode_time_steps": episode_steps,
        "seconds_per_time_step": seconds_per_time_step,
        "random_seed": seed,
        "debug_timing": True,
    }

    if render_mode != "none":
        kwargs.update(
            {
                "render_mode": render_mode,
                "render_directory": render_dir,
                "render_session_name": f"perf_{render_mode}",
            }
        )

    t0 = time.perf_counter()
    env = CityLearnEnv(str(SCHEMA), **kwargs)
    t1 = time.perf_counter()

    observations, _ = env.reset()
    t2 = time.perf_counter()

    action = np.zeros(env.action_space[0].shape[0], dtype="float32")
    step_times = []
    end_export_s = 0.0

    while not env.terminated:
        s0 = time.perf_counter()
        observations, _, terminated, truncated, info = env.step([action])
        s1 = time.perf_counter()
        step_times.append(s1 - s0)
        end_export_s += float(info.get("end_export_time", 0.0))

        if terminated or truncated:
            break

    t3 = time.perf_counter()
    env.close()

    avg_step_ms = float(np.mean(step_times) * 1000.0) if step_times else 0.0
    p95_step_ms = float(np.percentile(step_times, 95) * 1000.0) if step_times else 0.0

    return {
        "render_mode": render_mode,
        "configured_steps": episode_steps,
        "executed_steps": len(step_times),
        "seconds_per_time_step": seconds_per_time_step,
        "init_s": round(t1 - t0, 4),
        "reset_s": round(t2 - t1, 4),
        "rollout_s": round(t3 - t2, 4),
        "avg_step_ms": round(avg_step_ms, 4),
        "p95_step_ms": round(p95_step_ms, 4),
        "end_export_s": round(end_export_s, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--none-max-ms", type=float, default=30.0)
    parser.add_argument("--end-max-ms", type=float, default=45.0)
    parser.add_argument("--ratio-max", type=float, default=2.0)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    none_case = run_case("none", args.episode_steps, args.seconds, args.seed)
    end_case = run_case("end", args.episode_steps, args.seconds, args.seed)

    report = {
        "none": none_case,
        "end": end_case,
        "thresholds": {
            "none_max_ms": args.none_max_ms,
            "end_max_ms": args.end_max_ms,
            "ratio_max": args.ratio_max,
        },
    }
    print(json.dumps(report, indent=2))

    errors = []

    if none_case["executed_steps"] <= 0 or end_case["executed_steps"] <= 0:
        errors.append("No steps executed in one or more perf-smoke runs.")

    if none_case["avg_step_ms"] > args.none_max_ms:
        errors.append(
            f"none avg_step_ms too high: {none_case['avg_step_ms']} > {args.none_max_ms}"
        )

    if end_case["avg_step_ms"] > args.end_max_ms:
        errors.append(
            f"end avg_step_ms too high: {end_case['avg_step_ms']} > {args.end_max_ms}"
        )

    if none_case["avg_step_ms"] > 0:
        ratio = end_case["avg_step_ms"] / none_case["avg_step_ms"]
        if ratio > args.ratio_max:
            errors.append(f"end/none ratio too high: {ratio:.3f} > {args.ratio_max}")

    if errors:
        for error in errors:
            print(f"PERF_SMOKE_ERROR: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
