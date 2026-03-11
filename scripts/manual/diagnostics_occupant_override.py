#!/usr/bin/env python3
"""Generate diagnostics for the occupant interaction/LSTM module.

The script rolls out the Quebec dataset that ships with logistic-regression
occupant models, captures the key time-series, stores them to CSV, and prints a
few quick stats so you can confirm overrides still occur after code changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citylearn.citylearn import CityLearnEnv  # noqa: E402
from citylearn.agents.rbc import BasicRBC  # noqa: E402

DEFAULT_DATASET = (
    ROOT / "data/datasets/quebec_neighborhood_without_demand_response_set_points/schema.json"
)
DEFAULT_OUTPUT = ROOT / "SimulationData/occupant_override_diagnostics.csv"


def _build_time_index(start_date: pd.Timestamp, seconds_per_step: float, count: int) -> pd.DatetimeIndex:
    """Return a datetime index aligned with the simulation horizon."""
    freq = pd.Timedelta(seconds=seconds_per_step)
    return pd.date_range(start=start_date, periods=count, freq=freq)


def run(
    dataset: Path,
    output: Path,
    building_id: str | None,
    plot: bool,
    plot_output: Path | None,
    controller: str,
) -> None:
    env = CityLearnEnv(str(dataset), central_agent=True, render_mode="none", random_seed=0)

    try:
        observations, _ = env.reset()
        zeros = [np.zeros(env.action_space[0].shape[0], dtype="float32")]
        controller_obj = _make_controller(controller, env)

        while not env.terminated:
            if controller_obj is None:
                actions = zeros
            else:
                actions = controller_obj.predict(observations, deterministic=True)

            observations, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

        building = _select_building(env, building_id)
        df = _collect_series(env, building)

        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

        _print_summary(df, building.name, output, controller)

        if plot:
            target = plot_output or output.with_suffix(".png")
            _plot(df, building.name, target)

    finally:
        env.close()


def _make_controller(kind: str, env: CityLearnEnv):
    if kind == "zero":
        return None
    if kind == "basic-rbc":
        return BasicRBC(env)
    raise ValueError(f"Unsupported controller '{kind}'. Expected one of: zero, basic-rbc.")


def _select_building(env: CityLearnEnv, building_id: str | None):
    if building_id is None:
        return env.buildings[0]

    for building in env.buildings:
        if building.name == building_id:
            return building

    available = ", ".join(b.name for b in env.buildings)
    raise ValueError(f"Building '{building_id}' not found in dataset. Available: {available}")


def _collect_series(env: CityLearnEnv, building) -> pd.DataFrame:
    sim = building.energy_simulation
    params = getattr(building, "occupant", None)
    if params is None:
        raise RuntimeError(f"Building '{building.name}' does not expose an occupant model.")

    data = {
        "timestamp": _build_time_index(
            pd.Timestamp(env.render_start_date),
            env.seconds_per_time_step,
            len(sim.indoor_dry_bulb_temperature),
        ),
        "indoor_temp_c": sim.indoor_dry_bulb_temperature,
        "heating_setpoint_ctrl_c": sim.indoor_dry_bulb_temperature_heating_set_point,
        "heating_setpoint_base_c": sim.indoor_dry_bulb_temperature_heating_set_point_without_control,
        "cooling_setpoint_ctrl_c": sim.indoor_dry_bulb_temperature_cooling_set_point,
        "cooling_setpoint_base_c": sim.indoor_dry_bulb_temperature_cooling_set_point_without_control,
        "occupant_delta_c": building.occupant.parameters.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta,
        "hp_electricity_kwh": building.heating_device.electricity_consumption,
        "hvac_mode": sim.hvac_mode,
    }

    df = pd.DataFrame(data)
    df["heating_setpoint_delta_c"] = (
        df["heating_setpoint_ctrl_c"] - df["heating_setpoint_base_c"]
    )
    return df


def _print_summary(df: pd.DataFrame, building_name: str, output: Path, controller: str) -> None:
    override_mask = np.abs(df["occupant_delta_c"]) > 0
    overrides = int(override_mask.sum())
    max_delta = float(np.abs(df["occupant_delta_c"]).max())
    discomfort_mask = np.abs(df["indoor_temp_c"] - df["heating_setpoint_ctrl_c"]) > 2.0
    discomfort = int(discomfort_mask.sum())
    avg_energy_override = float(df.loc[override_mask, "hp_electricity_kwh"].mean())
    avg_energy_no_override = float(df.loc[~override_mask, "hp_electricity_kwh"].mean())

    print(f"Building: {building_name}")
    print(f"Controller: {controller}")
    print(f"Saved diagnostics to: {output}")
    print(f"Override samples (>0 delta): {overrides}")
    print(f"Max override magnitude: {max_delta:.3f} °C")
    print(f"Discomfort samples (>2 °C deviation): {discomfort}")
    print(
        f"Mean heat-pump energy (override / no-override): "
        f"{avg_energy_override:.3f} / {avg_energy_no_override:.3f} kWh"
    )


def _plot(df: pd.DataFrame, building_name: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # pragma: no cover
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting. Re-run without --plot.") from exc

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    df.plot(
        x="timestamp",
        y=["indoor_temp_c", "heating_setpoint_base_c", "heating_setpoint_ctrl_c"],
        ax=axes[0],
        title=f"{building_name}: indoor temperature & heating setpoints",
    )

    df.plot(
        x="timestamp",
        y="occupant_delta_c",
        ax=axes[1],
        color="tab:orange",
        title="Occupant override delta (°C)",
    )

    df.plot(
        x="timestamp",
        y="hp_electricity_kwh",
        ax=axes[2],
        color="tab:red",
        title="Heat pump electricity consumption (kWh)",
    )

    axes[2].set_xlabel("Timestamp")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def parse_args(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a CityLearn schema with occupant interaction enabled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV file where diagnostics should be written.",
    )
    parser.add_argument(
        "--building",
        type=str,
        default=None,
        help="Specific building name to analyse (defaults to the first building).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save quick matplotlib plots alongside the CSV output.",
    )
    parser.add_argument(
        "--list-buildings",
        action="store_true",
        help="List buildings available in the dataset and exit.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Explicit path for the saved plot (defaults to CSV path with .png extension).",
    )
    parser.add_argument(
        "--controller",
        choices=["zero", "basic-rbc"],
        default="basic-rbc",
        help="Controller policy to use during the rollout.",
    )

    return parser.parse_args(args)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_buildings:
        env = CityLearnEnv(str(args.dataset), central_agent=True, render_mode="none", random_seed=0)
        try:
            print("Available buildings:")
            for b in env.buildings:
                print(f" - {b.name}")
        finally:
            env.close()
        return

    run(
        args.dataset,
        args.output,
        args.building,
        args.plot,
        args.plot_output,
        args.controller,
    )


if __name__ == "__main__":
    main()
