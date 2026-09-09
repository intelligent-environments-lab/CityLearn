"""Unit conversion helpers shared by runtime services."""

from __future__ import annotations

from typing import Optional


def seconds_to_hours(seconds_per_time_step: float) -> float:
    """Convert a control-step duration in seconds to hours."""

    seconds = max(float(seconds_per_time_step), 0.0)
    return seconds / 3600.0


def power_kw_to_energy_kwh(power_kw: float, seconds_per_time_step: float) -> float:
    """Convert power in kW to control-step energy in kWh."""

    return float(power_kw) * seconds_to_hours(seconds_per_time_step)


def energy_kwh_to_power_kw(energy_kwh: float, seconds_per_time_step: float) -> float:
    """Convert control-step energy in kWh to average power in kW."""

    hours = max(seconds_to_hours(seconds_per_time_step), 1.0e-12)
    return float(energy_kwh) / hours


def normalized_power_action_to_energy_kwh(
    action: float,
    nominal_power_kw: float,
    seconds_per_time_step: float,
) -> float:
    """Convert a normalized power action to control-step energy in kWh."""

    return power_kw_to_energy_kwh(float(action) * float(nominal_power_kw), seconds_per_time_step)


def normalized_capacity_action_to_energy_kwh(
    action: float,
    capacity_kwh: float,
    *,
    seconds_per_time_step: Optional[float] = None,
    scale_with_time: bool = False,
) -> float:
    """Convert a normalized capacity action to control-step energy in kWh."""

    energy_kwh = float(action) * float(capacity_kwh)
    if not scale_with_time:
        return energy_kwh

    if seconds_per_time_step is None:
        raise ValueError("seconds_per_time_step must be provided when scale_with_time=True.")

    return energy_kwh * seconds_to_hours(seconds_per_time_step)


def to_dataset_resolution_energy(energy_kwh: float, time_step_ratio: Optional[float]) -> float:
    """Convert control-step energy to dataset-resolution energy when needed."""

    ratio = 1.0 if time_step_ratio in (None, 0) else float(time_step_ratio)
    return float(energy_kwh) / ratio
