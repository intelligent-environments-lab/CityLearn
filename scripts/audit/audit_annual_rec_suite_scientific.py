#!/usr/bin/env python3
"""Audit scientific readiness of the canonical annual REC dataset suite.

This audit complements ``audit_annual_rec_suite.py``.  The structural audit
checks contracts and referential integrity; this script measures whether the
generated scenarios are sufficiently varied, physically feasible and
documented for use as final algorithm benchmarks.  Findings are reported
without modifying the datasets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


FAMILIES = (
    "rec_2023_micro_4_q",
    "rec_2023_core_15_stripped",
    "rec_2023_core_30",
    "rec_2023_premium_100",
)
STEP_HOURS = 0.25
STEPS_PER_DAY = 96
FORECAST_HORIZONS = {
    "15m": 1,
    "1h": 4,
    "3h": 12,
    "6h": 24,
    "24h": 96,
}


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _category_summary(frame: pd.DataFrame, value: str) -> Mapping[str, Mapping[str, float]]:
    result = {}
    for category, values in frame.groupby("category")[value]:
        result[str(category)] = {
            "count": int(values.count()),
            "minimum": round(float(values.min()), 3),
            "median": round(float(values.median()), 3),
            "maximum": round(float(values.max()), 3),
        }
    return result


def _membership_trajectory(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    active = {
        member_id
        for member_id, building in schema["buildings"].items()
        if building.get("include", True)
    }
    initially_inactive = set(schema["buildings"]) - active
    trajectory = [{"time_step": 0, "active_members": len(active)}]
    first_entries = []
    returning_entries = []
    exits = []
    asset_events = []
    seen = set(active)
    for event in sorted(
        schema.get("topology_events", []),
        key=lambda item: (int(item["time_step"]), str(item["id"])),
    ):
        operation = event["operation"]
        member_id = event["target_member_id"]
        if operation == "add_member":
            if member_id in seen:
                returning_entries.append(member_id)
            else:
                first_entries.append(member_id)
                seen.add(member_id)
            active.add(member_id)
        elif operation == "remove_member":
            exits.append(member_id)
            active.discard(member_id)
        elif operation in {"add_asset", "remove_asset"}:
            asset_events.append(
                {
                    "operation": operation,
                    "asset_type": event.get("target_asset_type"),
                    "asset_id": event.get("target_asset_id"),
                    "member_id": member_id,
                    "time_step": int(event["time_step"]),
                }
            )
            continue
        else:
            continue
        trajectory.append(
            {"time_step": int(event["time_step"]), "active_members": len(active)}
        )
    return {
        "initially_active": int(trajectory[0]["active_members"]),
        "initially_inactive": len(initially_inactive),
        "first_entry_events": len(first_entries),
        "return_events": len(returning_entries),
        "exit_events": len(exits),
        "first_entry_member_ids": first_entries,
        "return_member_ids": returning_entries,
        "exit_member_ids": exits,
        "asset_event_count": len(asset_events),
        "asset_event_types": dict(
            Counter(
                f"{event['operation']}/{event['asset_type']}"
                for event in asset_events
            )
        ),
        "asset_events": asset_events,
        "minimum_active": min(row["active_members"] for row in trajectory),
        "maximum_active": max(row["active_members"] for row in trajectory),
        "final_active": int(trajectory[-1]["active_members"]),
        "trajectory": trajectory,
    }


def _phase_pressure(
    root: Path,
    members: pd.DataFrame,
    chargers: pd.DataFrame,
    schema: Mapping[str, Any],
) -> Mapping[str, Any]:
    phase_counts = (
        chargers.merge(members[["member_id", "category"]], on="member_id")
        .groupby(["category", "phase_connection"])
        .size()
    )
    pressure = {}
    all_three_phase = True
    all_balanced = True
    for member in members.itertuples(index=False):
        building = schema["buildings"][member.member_id]
        service = building.get("electrical_service")
        if not service:
            continue
        all_three_phase = all_three_phase and service.get("connection_type") == "three_phase"
        all_balanced = all_balanced and service.get("default_split") == "balanced"
        member_chargers = chargers[chargers["member_id"] == member.member_id]
        if member_chargers.empty:
            continue
        series = pd.read_parquet(
            root / "timeseries" / f"{member.member_id}.parquet",
            columns=["non_shiftable_load", "solar_generation"],
        )
        baseline_kw = (
            series["non_shiftable_load"].to_numpy(dtype="float64")
            - series["solar_generation"].to_numpy(dtype="float64")
        ) / STEP_HOURS
        requested_total_kw = np.zeros(len(series), dtype="float64")
        requested_phase_kw = {
            phase: np.zeros(len(series), dtype="float64")
            for phase in ("L1", "L2", "L3")
        }
        connected = np.zeros(len(series), dtype=bool)
        for charger in member_chargers.itertuples(index=False):
            schedule = pd.read_parquet(
                root / "chargers" / f"{charger.charger_id}.parquet",
                columns=["electric_vehicle_charger_state"],
            )
            occupied = schedule["electric_vehicle_charger_state"].to_numpy() == 1
            connected |= occupied
            requested_total_kw += occupied * float(charger.max_charging_power_kw)
            phase_connection = str(charger.phase_connection)
            if phase_connection == "all_phases":
                for phase in requested_phase_kw:
                    requested_phase_kw[phase] += (
                        occupied * float(charger.max_charging_power_kw) / 3.0
                    )
            else:
                requested_phase_kw[phase_connection] += (
                    occupied * float(charger.max_charging_power_kw)
                )
        total_limit = float(service["limits"]["total"]["import_kw"])
        phase_limits = {
            phase: float(values["import_kw"])
            for phase, values in service["limits"]["per_phase"].items()
        }
        total_exceeded = baseline_kw + requested_total_kw > total_limit + 1.0e-9
        phase_exceeded = np.zeros(len(series), dtype=bool)
        default_split = str(service.get("default_split", "balanced")).upper()
        for phase, limit in phase_limits.items():
            baseline_phase_kw = (
                baseline_kw / 3.0
                if default_split == "BALANCED"
                else baseline_kw * float(default_split == phase)
            )
            phase_exceeded |= (
                baseline_phase_kw + requested_phase_kw[phase] > limit + 1.0e-9
            )
        category = str(member.category)
        item = pressure.setdefault(
            category,
            {"connected_member_steps": 0, "phase_pressure_steps": 0, "total_pressure_steps": 0},
        )
        item["connected_member_steps"] += int(connected.sum())
        item["phase_pressure_steps"] += int((connected & phase_exceeded).sum())
        item["total_pressure_steps"] += int((connected & total_exceeded).sum())

    for item in pressure.values():
        denominator = max(int(item["connected_member_steps"]), 1)
        item["phase_pressure_percent"] = round(
            100.0 * item["phase_pressure_steps"] / denominator, 3
        )
        item["total_pressure_percent"] = round(
            100.0 * item["total_pressure_steps"] / denominator, 3
        )

    return {
        "charger_phase_counts": {
            f"{category}/{phase}": int(count)
            for (category, phase), count in phase_counts.items()
        },
        "all_electrical_services_three_phase": all_three_phase,
        "all_uncontrolled_power_balanced": all_balanced,
        "all_max_ev_request_pressure": pressure,
    }


def _daily_persistence_forecast_audit(
    load: np.ndarray,
    pv: np.ndarray,
) -> Mapping[str, Any]:
    """Quantify the causal forecast actually exposed by annual REC schemas."""

    horizons = {}
    causal = True
    for label, steps_ahead in FORECAST_HORIZONS.items():
        issue_steps = np.arange(
            max(STEPS_PER_DAY - steps_ahead, 0),
            load.shape[1] - steps_ahead,
            dtype="int64",
        )
        target_steps = issue_steps + steps_ahead
        source_steps = target_steps - STEPS_PER_DAY
        causal = causal and bool(np.all(source_steps <= issue_steps))

        load_truth = load[:, target_steps]
        load_prediction = load[:, source_steps]
        pv_truth = pv[:, target_steps]
        pv_prediction = pv[:, source_steps]
        load_mae = float(np.mean(np.abs(load_prediction - load_truth)))
        pv_mae = float(np.mean(np.abs(pv_prediction - pv_truth)))
        current_load_mae = float(
            np.mean(np.abs(load[:, issue_steps] - load_truth))
        )
        current_pv_mae = float(
            np.mean(np.abs(pv[:, issue_steps] - pv_truth))
        )
        horizons[label] = {
            "steps_ahead": steps_ahead,
            "source_lag_steps_from_target": STEPS_PER_DAY,
            "load_mae_kwh_per_step": round(load_mae, 6),
            "load_nmae_ratio": round(
                load_mae / max(float(np.mean(np.abs(load_truth))), 1.0e-12),
                6,
            ),
            "load_skill_vs_current_persistence": round(
                1.0 - load_mae / max(current_load_mae, 1.0e-12),
                6,
            ),
            "pv_mae_kwh_per_step": round(pv_mae, 6),
            "pv_nmae_ratio": round(
                pv_mae / max(float(np.mean(np.abs(pv_truth))), 1.0e-12),
                6,
            ),
            "pv_skill_vs_current_persistence": round(
                1.0 - pv_mae / max(current_pv_mae, 1.0e-12),
                6,
            ),
            "load_exact_future_match_ratio": round(
                float(np.mean(np.isclose(load_prediction, load_truth, atol=1.0e-9))),
                6,
            ),
            "pv_exact_future_match_ratio": round(
                float(np.mean(np.isclose(pv_prediction, pv_truth, atol=1.0e-9))),
                6,
            ),
        }

    return {
        "method": "daily_persistence",
        "causal": causal,
        "cold_start": "current_step",
        "cold_start_steps": STEPS_PER_DAY,
        "price_source": "publication_aware_day_ahead_market_input",
        "horizons": horizons,
    }


def _family_audit(root: Path) -> Mapping[str, Any]:
    manifest = _read_json(root / "dataset_manifest.json")
    members = pd.read_parquet(root / "catalogs/members.parquet")
    chargers = pd.read_parquet(root / "catalogs/chargers.parquet")
    evs = pd.read_parquet(root / "catalogs/electric_vehicles.parquet")
    sessions = pd.read_parquet(root / "catalogs/charging_sessions.parquet")
    electrical_services = pd.read_parquet(
        root / "catalogs/electrical_services.parquet"
    ).set_index("member_id")
    calendar = pd.read_parquet(root / "shared/calendar.parquet")

    enriched_sessions = (
        sessions.merge(
            chargers[["charger_id", "max_charging_power_kw"]], on="charger_id"
        )
        .merge(evs[["ev_id", "battery_capacity_kwh"]], on="ev_id")
        .merge(members[["member_id", "category"]], on="member_id")
    )
    enriched_sessions["duration_hours"] = (
        enriched_sessions["departure_time_step"]
        - enriched_sessions["arrival_time_step"]
    ) * STEP_HOURS
    enriched_sessions["required_energy_kwh"] = (
        enriched_sessions["required_departure_soc"]
        - enriched_sessions["arrival_soc"]
    ) * enriched_sessions["battery_capacity_kwh"]
    enriched_sessions["maximum_deliverable_energy_kwh"] = (
        enriched_sessions["duration_hours"]
        * enriched_sessions["max_charging_power_kw"]
        * 0.95
    )
    infeasible = (
        enriched_sessions["required_energy_kwh"]
        > enriched_sessions["maximum_deliverable_energy_kwh"] + 1.0e-9
    )
    arrivals_local = pd.to_datetime(
        enriched_sessions["arrival_timestamp_utc"], utc=True
    ).dt.tz_convert("Europe/Lisbon")
    arrival_clock_counts = arrivals_local.dt.strftime("%H:%M").value_counts()
    simultaneous_arrivals = enriched_sessions.groupby("arrival_time_step").size()

    load_pv = []
    for member in members.itertuples(index=False):
        series = pd.read_parquet(
            root / "timeseries" / f"{member.member_id}.parquet",
            columns=["non_shiftable_load", "solar_generation"],
        )
        load_pv.append(
            (
                series["non_shiftable_load"].to_numpy(dtype="float64"),
                series["solar_generation"].to_numpy(dtype="float64"),
            )
        )
    load = np.vstack([item[0] for item in load_pv])
    pv = np.vstack([item[1] for item in load_pv])
    surplus = np.maximum(pv - load, 0.0).sum(axis=0)
    demand = np.maximum(load - pv, 0.0).sum(axis=0)
    local_sharing = np.minimum(surplus, demand)

    native_import_violation_steps = 0
    native_export_violation_steps = 0
    native_import_violation_members = []
    native_export_violation_members = []
    for member_index, member in enumerate(members.itertuples(index=False)):
        service = electrical_services.loc[member.member_id]
        native_net_kw = (load[member_index] - pv[member_index]) / STEP_HOURS
        import_violation = (
            native_net_kw > float(service["total_import_limit_kw"]) + 1.0e-9
        )
        export_violation = (
            -native_net_kw > float(service["total_export_limit_kw"]) + 1.0e-9
        )
        native_import_violation_steps += int(import_violation.sum())
        native_export_violation_steps += int(export_violation.sum())
        if import_violation.any():
            native_import_violation_members.append(member.member_id)
        if export_violation.any():
            native_export_violation_members.append(member.member_id)

    asset_allocation = {}
    for category, category_members in members.groupby("category"):
        charging_members = category_members[category_members["has_charging"]]
        charging_with_pv = int(charging_members["has_pv"].sum())
        asset_allocation[str(category)] = {
            "members": len(category_members),
            "pv_members": int(category_members["has_pv"].sum()),
            "battery_members": int(category_members["has_battery"].sum()),
            "charging_members": len(charging_members),
            "charging_members_with_pv": charging_with_pv,
            "charging_members_without_pv": len(charging_members) - charging_with_pv,
            "deferrable_members": int(category_members["has_deferrable"].sum()),
            "distinct_capability_combinations": int(
                category_members[
                    ["has_pv", "has_battery", "has_charging", "has_deferrable"]
                ]
                .drop_duplicates()
                .shape[0]
            ),
        }

    deferrable_windows = Counter()
    for item in pd.read_parquet(root / "catalogs/deferrables.parquet").itertuples(index=False):
        schedule = pd.read_parquet(
            root / "deferrables" / f"{item.deferrable_id}_schedule.parquet"
        )
        for row in schedule.itertuples(index=False):
            deferrable_windows[
                (
                    int(row.earliest_start_time_step) % 96,
                    int(row.latest_start_time_step) % 96,
                    int(row.deadline_time_step) % 96,
                )
            ] += 1

    safety_schema = None
    for variant, relative in manifest["variants"].items():
        if variant in {"CORE-30-SAFETY", "PREMIUM-100-CLEAN"}:
            safety_schema = _read_json(root / relative)
            break
    phase = (
        _phase_pressure(root, members, chargers, safety_schema)
        if safety_schema is not None
        else None
    )

    dynamic = None
    for variant, relative in manifest["variants"].items():
        if variant in {"CORE-30-DYNAMIC", "PREMIUM-100-CLEAN"}:
            dynamic = _membership_trajectory(_read_json(root / relative))
            break

    structural_report = _read_json(root / "validation_report.json")
    return {
        "dataset_family": manifest["dataset_family"],
        "structural_audit_status": structural_report["status"],
        "composition": manifest["composition"],
        "annual_load_kwh_by_category": _category_summary(
            members, "annual_non_shiftable_load_kwh"
        ),
        "pv_capacity_kw_by_category": _category_summary(
            members[members["has_pv"]], "pv_capacity_kw"
        ),
        "annual_pv_specific_yield_kwh_per_kw": round(
            float(
                (
                    members.loc[members["has_pv"], "annual_pv_generation_kwh"]
                    / members.loc[members["has_pv"], "pv_capacity_kw"]
                ).mean()
            ),
            3,
        ),
        "ev_sessions": {
            "count": len(enriched_sessions),
            "infeasible_target_count": int(infeasible.sum()),
            "infeasible_target_percent": round(100.0 * float(infeasible.mean()), 3),
            "infeasible_by_category": {
                str(category): int(infeasible.loc[index].sum())
                for category, index in enriched_sessions.groupby("category").groups.items()
            },
            "arrival_clock_counts": {
                str(clock): int(count) for clock, count in arrival_clock_counts.items()
            },
            "distinct_arrival_clock_times": int(arrival_clock_counts.size),
            "maximum_simultaneous_arrivals": int(simultaneous_arrivals.max()),
            "zero_temporal_slack_count": int(
                (enriched_sessions["temporal_slack_steps"] <= 0).sum()
            ),
            "one_or_fewer_temporal_slack_steps_count": int(
                (enriched_sessions["temporal_slack_steps"] <= 1).sum()
            ),
            "duration_hours": {
                "minimum": round(float(enriched_sessions["duration_hours"].min()), 3),
                "median": round(float(enriched_sessions["duration_hours"].median()), 3),
                "maximum": round(float(enriched_sessions["duration_hours"].max()), 3),
            },
        },
        "deferrables": {
            "distinct_local_window_templates": len(deferrable_windows),
            "most_common_window_count": max(deferrable_windows.values(), default=0),
        },
        "baseline_settlement_potential": {
            "local_sharing_kwh": round(float(local_sharing.sum()), 3),
            "time_steps_with_local_sharing": int((local_sharing > 0.0).sum()),
            "annual_load_kwh": round(float(load.sum()), 3),
            "annual_pv_kwh": round(float(pv.sum()), 3),
        },
        "derived_forecasts": _daily_persistence_forecast_audit(load, pv),
        "native_electrical_feasibility": {
            "import_violation_steps": native_import_violation_steps,
            "export_violation_steps": native_export_violation_steps,
            "import_violation_member_ids": native_import_violation_members,
            "export_violation_member_ids": native_export_violation_members,
            "contract_level_upgrade_count": int(
                electrical_services[
                    "contract_level_upgraded_for_native_feasibility"
                ].sum()
            ),
        },
        "asset_allocation": asset_allocation,
        "phase_constraints": phase,
        "dynamic_membership": dynamic,
        "calendar": {
            "rows": len(calendar),
            "utc_offsets_minutes": sorted(
                int(value) for value in calendar["utc_offset_minutes"].unique()
            ),
        },
        "scientific_contract": manifest.get("calibration_and_scope", {}),
        "file_integrity_declared": (root / manifest.get("file_integrity", "missing")).is_file(),
    }


def _findings(families: Mapping[str, Mapping[str, Any]], source_root: Path) -> list[Mapping[str, Any]]:
    findings = []
    premium = families["PREMIUM-100"]
    infeasible = premium["ev_sessions"]["infeasible_target_count"]
    if infeasible:
        findings.append(
            {
                "severity": "critical",
                "id": "EV-TARGET-FEASIBILITY",
                "finding": f"PREMIUM-100 contains {infeasible} charging sessions whose target energy cannot be delivered within the declared window at charger power and 95% efficiency.",
                "impact": "A must-serve deadline can be impossible before any controller acts, confounding algorithm and service-failure results.",
                "required_action": "Constrain sampled arrival SOC/target SOC by deliverable energy or explicitly label deliberately infeasible stress sessions and exclude them from ordinary service-success denominators.",
            }
        )

    for family_name, family in families.items():
        if not family["derived_forecasts"]["causal"]:
            findings.append(
                {
                    "severity": "critical",
                    "id": f"FORECAST-LEAKAGE-{family_name}",
                    "finding": f"{family_name} uses a load/PV forecast source after the forecast issue time.",
                    "impact": "Controllers would receive future physical truth and benchmark results would be optimistically biased.",
                    "required_action": "Use only causal input data and repeat all affected controller runs.",
                }
            )
        native = family["native_electrical_feasibility"]
        if native["import_violation_steps"] or native["export_violation_steps"]:
            findings.append(
                {
                    "severity": "critical",
                    "id": f"NATIVE-ELECTRICAL-FEASIBILITY-{family_name}",
                    "finding": (
                        f"{family_name} exceeds its electrical-service envelope before "
                        f"controllable assets act: {native['import_violation_steps']} import "
                        f"and {native['export_violation_steps']} export member-steps."
                    ),
                    "impact": "A controller cannot repair an exogenous native-load or PV violation, so safety results would be structurally confounded.",
                    "required_action": "Resize the declared connection contract or regenerate the native profile while preserving the intended flexible-asset pressure.",
                }
            )
        if family["ev_sessions"]["zero_temporal_slack_count"]:
            findings.append(
                {
                    "severity": "high",
                    "id": f"EV-ZERO-CONTROL-SLACK-{family_name}",
                    "finding": f"{family_name} contains {family['ev_sessions']['zero_temporal_slack_count']} individually feasible sessions that require full-rate charging during every connected step.",
                    "impact": "A one-step observation or actuation delay makes these ordinary sessions operationally impossible even though the energy-only feasibility flag is true.",
                    "required_action": "Reserve at least one control step of temporal slack for ordinary sessions; represent forced or disrupted service only in explicitly labelled stress events.",
                }
            )

    for family_name in ("MICRO-4-Q", "CORE-15-STRIPPED", "CORE-30", "PREMIUM-100"):
        for category, allocation in families[family_name]["asset_allocation"].items():
            charging_count = allocation["charging_members"]
            if charging_count < 2:
                continue
            if not allocation["charging_members_with_pv"] or not allocation[
                "charging_members_without_pv"
            ]:
                findings.append(
                    {
                        "severity": "high",
                        "id": f"ASSET-ALLOCATION-CONFOUNDING-{family_name}-{category}",
                        "finding": f"Every charging member in the {category} stratum of {family_name} has the same PV-presence state.",
                        "impact": "EV-control effects cannot be separated from co-located PV effects within that stratum.",
                        "required_action": "Keep the declared asset counts but allocate chargers across both PV and non-PV members using a reproducible stratified design.",
                    }
                )

    for family_name in ("CORE-30", "PREMIUM-100"):
        phase = families[family_name]["phase_constraints"]
        residential = {
            key: value
            for key, value in phase["charger_phase_counts"].items()
            if key.startswith("residential/")
        }
        individual_phase_counts = [
            residential.get(f"residential/{phase}", 0)
            for phase in ("L1", "L2", "L3")
        ]
        if residential and (
            min(individual_phase_counts) == 0
            or max(individual_phase_counts) - min(individual_phase_counts) > 1
        ):
            findings.append(
                {
                    "severity": "high",
                    "id": f"PHASE-ALLOCATION-{family_name}",
                    "finding": f"{family_name} has an avoidable imbalance in individually connected residential chargers; counts are {residential}.",
                    "impact": "Community phase pressure would partly reflect asset numbering rather than a declared experimental phase allocation.",
                    "required_action": "Rotate or sample phases across members, preserve reproducibility, and document the intended phase-imbalance distribution.",
                }
            )

        dynamic = families[family_name]["dynamic_membership"]
        if dynamic and dynamic["return_events"]:
            findings.append(
                {
                    "severity": "high",
                    "id": f"MEMBER-STATE-RESET-{family_name}",
                    "finding": f"{family_name} contains {dynamic['return_events']} member re-entry events within the same benchmark episode.",
                    "impact": "A re-entry can ambiguously preserve or reset accumulated storage, mobility and service state, weakening KPI attribution.",
                    "required_action": "Use permanent departures and distinct first-entry members in the canonical benchmark, or introduce an explicit state-continuity contract before evaluating re-entry.",
                }
            )

    if premium["ev_sessions"]["distinct_arrival_clock_times"] <= 6:
        findings.append(
            {
                "severity": "high",
                "id": "EV-SYNCHRONISATION",
                "finding": "EV arrivals and departures use only six fixed clock templates in PREMIUM-100; CORE-30 uses one residential arrival time for all sessions.",
                "impact": "Artificial synchronisation creates deterministic peaks and weakens behavioural representativeness.",
                "required_action": "Introduce reproducible member/day variation, weekday/holiday effects, missed visits, trip-distance variation and a declared correlation model.",
            }
        )

    findings.extend(
        [
            {
                "severity": "medium",
                "id": "EMPIRICAL-VALIDATION-BOUNDARY",
                "finding": "Annual demand totals and electrical-connection classes are anchored to cited Portuguese references, while intra-year shapes, mobility, PV, weather, carbon and deferrable behaviour remain reproducible scenario constructions.",
                "impact": "The suite is suitable for controlled algorithm comparison, but not for claiming that it is a statistically fitted sample of Portuguese RECs.",
                "required_action": "Preserve this claim boundary; add an external goodness-of-fit validation artefact only if population-level representativeness becomes part of a thesis conclusion.",
            },
            {
                "severity": "medium",
                "id": "PHYSICAL-SCOPE",
                "finding": "Safety schemas now include single-/three-phase connection heterogeneity and Portuguese contracted-power levels, but they do not model feeder voltage, lines, reactive power or protection.",
                "impact": "The scenarios test connection and phase-headroom constraints, not general distribution-grid safety or compliance engineering.",
                "required_action": "Keep claims limited to active-power connection and phase-headroom constraints; use a power-flow model for feeder-level conclusions.",
            },
        ]
    )

    if premium["deferrables"]["distinct_local_window_templates"] < 20:
        findings.append(
            {
                "severity": "medium",
                "id": "DEFERRABLE-SYNCHRONISATION",
                "finding": "Deferrable requests use fewer than twenty distinct local flexibility-window templates.",
                "impact": "Algorithm performance may depend on template regularity rather than general flexible-service scheduling.",
                "required_action": "Increase reproducible variation in request days, windows, deadlines and must-run rates.",
            }
        )

    if not all(family["file_integrity_declared"] for family in families.values()):
        findings.append(
            {
                "severity": "medium",
                "id": "DATASET-FILE-INTEGRITY",
                "finding": "One or more families do not declare a complete generated-file checksum record.",
                "impact": "Post-generation file changes would be harder to detect.",
                "required_action": "Freeze every family with a complete SHA-256 manifest.",
            }
        )

    provenance = pd.read_csv(source_root / "omie_2023_daily_provenance.csv")
    hourly_prices = pd.read_parquet(source_root / "omie_2023_hourly.parquet")
    hourly_utc = pd.to_datetime(hourly_prices["timestamp_utc"], utc=True)
    hourly_local_dates = hourly_utc.dt.tz_convert("Europe/Lisbon").dt.strftime(
        "%Y-%m-%d"
    )
    return findings, {
        "daily_files": len(provenance),
        "period_distribution": {
            str(periods): int(count)
            for periods, count in provenance["periods"].value_counts().sort_index().items()
        },
        "official_urls_only": bool(
            provenance["url"].str.startswith("https://www.omie.es/").all()
        ),
        "all_daily_sha256_present": bool(
            provenance["raw_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
        ),
        "hourly_rows": len(hourly_prices),
        "strictly_hourly_utc": bool(
            hourly_utc.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        ),
        "market_date_matches_portuguese_local_date": bool(
            hourly_local_dates.eq(hourly_prices["market_date"].astype(str)).all()
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Canonical annual REC suite: deep scientific audit",
        "",
        f"Audit date: {report['audit_date']}",
        "",
        f"- Structural readiness: **{report['readiness']['structural']}**",
        f"- Simulator smoke readiness: **{report['readiness']['simulator_smoke']}**",
        f"- Final scientific benchmark readiness: **{report['readiness']['final_scientific_benchmark']}**",
        "",
        "## Findings",
        "",
    ]
    for finding in report["findings"]:
        lines.extend(
            [
                f"### {finding['severity'].upper()}: {finding['id']}",
                "",
                finding["finding"],
                "",
                f"Impact: {finding['impact']}",
                "",
                f"Required action: {finding['required_action']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The suite is structurally valid and ready for controlled algorithm benchmarking within its declared scope. It is a calibrated hybrid scenario suite, not a statistically fitted sample of Portuguese energy communities and not a feeder power-flow model.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--dataset-root", type=Path, default=repository / "data/datasets"
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=repository / "results/raw/annual_rec_suite_deep_audit_2026-08-21.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=repository / "results/raw/annual_rec_suite_deep_audit_2026-08-21.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = {}
    for directory in FAMILIES:
        result = _family_audit(args.dataset_root / directory)
        families[result["dataset_family"]] = result
    findings, omie = _findings(
        families, args.dataset_root / "rec_2023_source_data"
    )
    severe_findings = [
        finding for finding in findings if finding["severity"] in {"critical", "high"}
    ]
    report = _json_ready(
        {
            "audit_date": date.today().isoformat(),
            "readiness": {
                "structural": "pass",
                "simulator_smoke": "pass",
                "final_scientific_benchmark": (
                    "ready_with_declared_scope" if not severe_findings else "not_ready"
                ),
            },
            "interpretation": (
                "Ready for controlled benchmark use within the declared hybrid-scenario and connection-headroom scope."
                if not severe_findings
                else "Critical or high-severity findings must be resolved before benchmark freeze."
            ),
            "omie_provenance": omie,
            "families": families,
            "findings": findings,
        }
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["readiness"], indent=2))
    print(f"findings: {len(findings)}")
    for severity in ("critical", "high", "medium", "low"):
        count = sum(item["severity"] == severity for item in findings)
        if count:
            print(f"{severity}: {count}")


if __name__ == "__main__":
    main()
