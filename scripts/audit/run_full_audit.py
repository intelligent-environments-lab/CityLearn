#!/usr/bin/env python3
"""Run full simulator audit suite and aggregate reports."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "scripts" / "audit"
OUTPUT_DIR_DEFAULT = ROOT / "outputs" / "audit"


def _run(cmd: List[str], cwd: Path) -> Dict[str, Any]:
    started = dt.datetime.utcnow()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    ended = dt.datetime.utcnow()

    return {
        "cmd": cmd,
        "returncode": int(result.returncode),
        "started_utc": started.replace(microsecond=0).isoformat() + "Z",
        "ended_utc": ended.replace(microsecond=0).isoformat() + "Z",
        "stdout": result.stdout[-20000:],
        "stderr": result.stderr[-20000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--strict", action="store_true", help="Fail if any component fails.")
    parser.add_argument("--episode-steps", type=int, default=2000, help="Passed to perf/results audit where supported.")
    parser.add_argument("--seed", type=int, default=0, help="Passed to sub-audits where supported.")
    parser.add_argument("--update-snapshots", action="store_true", help="Update contract/KPI baselines in snapshot-aware audits.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_scripts = [
        {
            "name": "physics",
            "path": AUDIT_DIR / "audit_physics.py",
            "extra_args": [
                "--output",
                str(args.output_dir / "physics_audit.json"),
            ],
        },
        {
            "name": "entity_contract",
            "path": AUDIT_DIR / "audit_entity_contract.py",
            "extra_args": [
                "--output-dir",
                str(args.output_dir),
                "--strict",
            ] + (["--update-snapshots"] if args.update_snapshots else []),
        },
        {
            "name": "datasets",
            "path": AUDIT_DIR / "audit_datasets.py",
            "extra_args": [
                "--output",
                str(args.output_dir / "datasets_audit.json"),
            ],
        },
        {
            "name": "performance_results",
            "path": AUDIT_DIR / "audit_performance_results.py",
            "extra_args": [
                "--output",
                str(args.output_dir / "performance_results_audit.json"),
                "--episode-steps",
                str(args.episode_steps),
                "--seed",
                str(args.seed),
                "--strict",
            ] + (["--update-baselines"] if args.update_snapshots else []),
        },
    ]

    runs: List[Dict[str, Any]] = []
    hard_fail = False
    for item in audit_scripts:
        script_path: Path = item["path"]
        if not script_path.is_file():
            runs.append(
                {
                    "name": item["name"],
                    "status": "missing",
                    "path": str(script_path.relative_to(ROOT)),
                }
            )
            hard_fail = True
            continue

        cmd = [sys.executable, str(script_path), *item["extra_args"]]
        run = _run(cmd, ROOT)
        run["name"] = item["name"]
        run["path"] = str(script_path.relative_to(ROOT))

        if run["returncode"] == 0:
            run["status"] = "ok"
        elif run["returncode"] == 2:
            run["status"] = "skipped_deps"
        else:
            run["status"] = "failed"
            hard_fail = True

        runs.append(run)

    summary = {
        "generated_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "python": sys.version,
        "strict": bool(args.strict),
        "output_dir": str(args.output_dir),
        "runs": runs,
    }
    summary_path = args.output_dir / "full_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[audit] wrote summary: {summary_path}")
    for run in runs:
        print(f"[audit] {run['name']}: {run['status']}")

    if args.strict and hard_fail:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
