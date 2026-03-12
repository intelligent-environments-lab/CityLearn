#!/usr/bin/env python3
"""Compare equity KPIs between two CityLearn runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


EQUITY_KPIS = [
    "equity_gini_benefit",
    "equity_cr20_benefit",
    "equity_losers_percent",
    "equity_bpr_asset_poor_over_rich",
]

LOWER_IS_BETTER = {
    "equity_gini_benefit",
    "equity_cr20_benefit",
    "equity_losers_percent",
}


def _resolve_kpis_path(path: Path) -> Path:
    if path.is_dir():
        return path / "exported_kpis.csv"

    return path


def _load_district_equity(path: Path) -> Dict[str, float]:
    kpi_path = _resolve_kpis_path(path)

    if not kpi_path.is_file():
        raise FileNotFoundError(f"KPI file not found: {kpi_path}")

    df = pd.read_csv(kpi_path)
    district = df[df["name"] == "District"].set_index("cost_function")["value"]

    values = {}
    for kpi in EQUITY_KPIS:
        values[kpi] = float(district[kpi]) if kpi in district.index and pd.notna(district[kpi]) else np.nan

    return values


def _safe_pct(delta: float, base: float) -> float:
    if not np.isfinite(base) or base == 0.0:
        return np.nan

    return float(100.0 * delta / base)


def _is_improved(kpi: str, delta: float) -> bool:
    if not np.isfinite(delta):
        return False

    if kpi in LOWER_IS_BETTER:
        return delta < 0.0

    return delta > 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, type=Path, help="Run A folder or exported_kpis.csv path.")
    parser.add_argument("--run-b", required=True, type=Path, help="Run B folder or exported_kpis.csv path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV filepath.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_a = _load_district_equity(args.run_a)
    run_b = _load_district_equity(args.run_b)
    rows = []

    for kpi in EQUITY_KPIS:
        a_value = run_a[kpi]
        b_value = run_b[kpi]
        delta = b_value - a_value if np.isfinite(a_value) and np.isfinite(b_value) else np.nan
        rows.append(
            {
                "cost_function": kpi,
                "run_a": a_value,
                "run_b": b_value,
                "delta_absolute_b_minus_a": delta,
                "delta_percent_b_vs_a": _safe_pct(delta, a_value) if np.isfinite(delta) else np.nan,
                "direction": "lower_is_better" if kpi in LOWER_IS_BETTER else "higher_is_better",
                "improved_in_b": _is_improved(kpi, delta),
            }
        )

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"\nSaved comparison to: {args.output}")


if __name__ == "__main__":
    main()
