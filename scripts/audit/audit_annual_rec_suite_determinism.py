#!/usr/bin/env python3
"""Regenerate the annual REC suite in isolation and compare frozen outputs."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


FAMILIES = (
    "rec_2023_micro_4_q",
    "rec_2023_core_15_stripped",
    "rec_2023_core_30",
    "rec_2023_premium_100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--dataset-root", type=Path, default=repository / "data/datasets"
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=repository
        / "results/raw/annual_rec_suite_determinism_audit_2026-08-21.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=repository
        / "results/raw/annual_rec_suite_determinism_audit_2026-08-21.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[2]
    source_root = args.dataset_root / "rec_2023_source_data"
    generator = repository / "scripts/generate_annual_rec_suite.py"
    family_results = {}

    with TemporaryDirectory(prefix="annual-rec-determinism-") as temporary:
        temporary_root = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                str(generator),
                "--families",
                "all",
                "--output-root",
                str(temporary_root),
                "--source-root",
                str(source_root),
            ],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for family in FAMILIES:
            canonical = (
                args.dataset_root / family / "file_checksums.sha256"
            ).read_bytes()
            regenerated = (
                temporary_root / family / "file_checksums.sha256"
            ).read_bytes()
            family_results[family] = {
                "checksum_manifest_byte_identical": canonical == regenerated,
                "declared_file_count": len(
                    canonical.decode("utf-8").splitlines()
                ),
            }

    status = (
        "pass"
        if all(
            item["checksum_manifest_byte_identical"]
            for item in family_results.values()
        )
        else "fail"
    )
    report = {
        "audit_date": date.today().isoformat(),
        "status": status,
        "method": (
            "fresh isolated generation from the cached official OMIE source; "
            "byte comparison of every family checksum manifest"
        ),
        "dataset_contract_version": "2023-q15-v1.9",
        "families": family_results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Annual REC deterministic-generation audit",
        "",
        f"Status: **{status}**.",
        "",
        report["method"].capitalize() + ".",
        "",
        "| Family | Byte-identical manifest | Declared files |",
        "|---|---:|---:|",
    ]
    for family, item in family_results.items():
        lines.append(
            f"| {family} | {item['checksum_manifest_byte_identical']} | "
            f"{item['declared_file_count']} |"
        )
    lines.append("")
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
