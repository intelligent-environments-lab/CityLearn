"""UI export contract regression tests.

These tests validate the CSV folder/file/header contract that CityLearn UI
consumes in RecDashboard/KPIs parsing logic.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[2] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _control_action(env: CityLearnEnv) -> np.ndarray:
    """Deterministic action vector with EV/BESS activity when available."""

    names = env.action_names[0]
    action = np.zeros(env.action_space[0].shape[0], dtype="float32")

    ev_indices = [i for i, name in enumerate(names) if name.startswith("electric_vehicle_storage_")]
    battery_indices = [i for i, name in enumerate(names) if "electrical_storage" in name]

    if ev_indices:
        action[ev_indices] = 0.7

    if battery_indices:
        action[battery_indices] = 0.4

    return action


def _assert_required_columns(path: Path, required: set[str]):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        rows = list(reader)

    missing = sorted(required - header)
    assert not missing, f"{path.name} is missing required UI columns: {missing}"
    assert rows, f"{path.name} must contain at least one data row."

    for idx, row in enumerate(rows[:2]):
        timestamp = row.get("timestamp")
        assert timestamp, f"{path.name} row {idx} has empty timestamp."
        datetime.fromisoformat(timestamp)


def test_ui_export_layout_files_and_headers_contract(tmp_path):
    """Validate that export output is directly ingestible by CityLearn UI."""

    simulation_data_root = tmp_path / "SimulationData"
    simulation_name = "Simulation_Contract"

    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=48,
        render_mode="end",
        render_directory=simulation_data_root,
        render_session_name=simulation_name,
        random_seed=0,
    )

    try:
        env.reset()

        while not env.terminated:
            _, _, terminated, truncated, _ = env.step([_control_action(env)])
            if terminated or truncated:
                break

        outputs_path = Path(env.new_folder_path)
        assert outputs_path.is_dir()
        assert outputs_path.parent == simulation_data_root
        assert outputs_path.name == simulation_name

        data_files = sorted(outputs_path.glob("exported_data_*_ep0.csv"))
        assert data_files, "Expected exported_data_*_ep0.csv files for UI dashboard import."

        categories = {
            "community": [],
            "building": [],
            "battery": [],
            "charger": [],
            "ev": [],
            "pricing": [],
        }

        for path in data_files:
            stem = path.stem
            assert stem.startswith("exported_data_"), f"Unexpected data file name: {path.name}"
            assert re.search(r"_ep\d+$", stem), f"Missing episode suffix in: {path.name}"

            cleaned = stem.replace("exported_data_", "").lower()

            if cleaned.startswith("community_"):
                categories["community"].append(path)
            elif re.match(r"^building_\d+_charger_\d+_\d+_ep\d+$", cleaned):
                categories["charger"].append(path)
            elif re.match(r"^building_\d+_battery_ep\d+$", cleaned):
                categories["battery"].append(path)
            elif re.match(r"^building_\d+_ep\d+$", cleaned):
                categories["building"].append(path)
            elif cleaned.startswith("electric_vehicle_"):
                categories["ev"].append(path)
            elif cleaned.startswith("pricing_"):
                categories["pricing"].append(path)

        assert categories["community"], "UI contract requires community CSV."
        assert categories["building"], "UI contract requires at least one building CSV."
        assert categories["battery"], "UI contract requires at least one battery CSV."
        assert categories["pricing"], "UI contract requires pricing CSV."

        if any((building.electric_vehicle_chargers or []) for building in env.buildings):
            assert categories["charger"], "UI contract requires charger CSVs when chargers exist."

        if env.electric_vehicles:
            assert categories["ev"], "UI contract requires EV CSVs when EVs exist."

        _assert_required_columns(
            categories["community"][0],
            {
                "timestamp",
                "Net Electricity Consumption-kWh",
                "Self Consumption-kWh",
                "Total Solar Generation-kWh",
            },
        )
        _assert_required_columns(
            categories["building"][0],
            {
                "timestamp",
                "Net Electricity Consumption-kWh",
                "Non-shiftable Load-kWh",
                "Energy Production from PV-kWh",
            },
        )
        _assert_required_columns(
            categories["battery"][0],
            {
                "timestamp",
                "Battery Soc-%",
                "Battery (Dis)Charge-kWh",
            },
        )
        _assert_required_columns(
            categories["pricing"][0],
            {
                "timestamp",
                "electricity_pricing-$/kWh",
                "electricity_pricing_predicted_1-$/kWh",
                "electricity_pricing_predicted_2-$/kWh",
                "electricity_pricing_predicted_3-$/kWh",
            },
        )

        if categories["charger"]:
            _assert_required_columns(
                categories["charger"][0],
                {
                    "timestamp",
                    "Charger Consumption-kWh",
                    "Charger Production-kWh",
                    "EV Required SOC Departure-%",
                    "EV Estimated SOC Arrival-%",
                    "EV Arrival Time",
                    "EV Departure Time",
                    "EV Name",
                },
            )

        if categories["ev"]:
            _assert_required_columns(
                categories["ev"][0],
                {
                    "timestamp",
                    "name",
                    "Battery capacity",
                    "electric_vehicle_soc",
                },
            )

        kpis_path = outputs_path / "exported_kpis.csv"
        assert kpis_path.is_file(), "UI KPI page expects exported_kpis.csv."

        with kpis_path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
        assert header and header[0] == "KPI"

        with kpis_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        kpi_names = {row["KPI"] for row in rows}

        # Keep KPI CSV schema stable even when some metrics are undefined (all-NaN before CSV fill).
        for expected_kpi in {
            "equity_gini_benefit",
            "equity_cr20_benefit",
            "equity_bpr_asset_poor_over_rich",
            "equity_losers_percent",
            "equity_relative_benefit_percent",
        }:
            assert expected_kpi in kpi_names, f"Missing KPI row in export contract: {expected_kpi}"
    finally:
        env.close()
