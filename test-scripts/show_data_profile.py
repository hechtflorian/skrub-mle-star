#!/usr/bin/env python3
"""Show the compact data_profile string fed into refinement agent prompts.

Agents receive format_ablation_profile(...) plain text — NOT raw table_report.json.
Full JSON is archived on disk only (workspace/<task>/table_report.json).

Usage:
  cd agents/machine-learning-engineering && uv run python ../../test-scripts/show_data_profile.py \\
    --report ../../experiments/phase1/california-housing-prices/skrub-full/gpt-5.4-mini/run2/table_report.json

  python test-scripts/show_data_profile.py --scan experiments/phase1
  (requires skrub/pandas in env — use uv run from agents/machine-learning-engineering)

  python test-scripts/show_data_profile.py \\
    --report path/to/table_report.json --target-col median_house_value
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_AGENT_PKG = _REPO / "agents" / "machine-learning-engineering" / "machine_learning_engineering"
_UTIL_PATH = _AGENT_PKG / "shared_libraries" / "table_report_util.py"


def _load_table_report_util():
    """Import table_report_util without pulling in machine_learning_engineering.__init__."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("table_report_util", _UTIL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_UTIL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


table_report_util = _load_table_report_util()


def _load_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _summarize_report_path(path: Path, target_col: str | None) -> None:
    report = _load_report(path)
    profile = table_report_util.format_ablation_profile(
        report, target_col=target_col
    )
    raw_bytes = path.stat().st_size
    raw_json_chars = len(json.dumps(report))
    lines = profile.splitlines()

    print("=" * 72)
    print(f"Report file: {path}")
    print(f"  rows={report.get('n_rows')} cols={report.get('n_columns')} "
          f"associations={len(report.get('top_associations') or [])}")
    print(f"  Raw JSON on disk:     {raw_bytes:,} bytes")
    print(f"  Raw JSON serialized:  {raw_json_chars:,} chars")
    print(f"  Agent data_profile:   {len(profile):,} chars, {len(lines)} lines")
    print(f"  Compression ratio:    {raw_json_chars / max(len(profile), 1):.1f}x smaller")
    if target_col:
        print(f"  target_col override:  {target_col}")
    print("-" * 72)
    print("--- data_profile (exact text in {data_profile}) ---")
    print(profile)
    print("--- end data_profile ---")
    print()


def _scan(root: Path) -> list[Path]:
    return sorted(root.rglob("table_report.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to a saved table_report.json",
    )
    parser.add_argument(
        "--scan",
        type=Path,
        help="Scan directory for all table_report.json files (summary only)",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Optional target column name (as extracted from train0.py in runs)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="With --scan, print full profile for each file (default: sizes only)",
    )
    args = parser.parse_args()

    if args.report:
        report_path = args.report.resolve()
        if not report_path.is_file():
            raise SystemExit(f"Not found: {report_path}")
        _summarize_report_path(report_path, args.target_col)
        return

    if args.scan:
        scan_root = args.scan.resolve()
        paths = _scan(scan_root)
        if not paths:
            raise SystemExit(f"No table_report.json under {scan_root}")
        print(f"Found {len(paths)} table_report.json file(s)\n")
        for path in paths:
            report = _load_report(path)
            profile = table_report_util.format_ablation_profile(
                report, target_col=args.target_col
            )
            raw = path.stat().st_size
            try:
                rel = path.resolve().relative_to(_REPO)
            except ValueError:
                rel = path
            print(
                f"{rel}: "
                f"json={raw:,}B → profile={len(profile):,} chars "
                f"({report.get('n_columns')} cols)"
            )
            if args.full:
                print("-" * 40)
                print(profile)
                print()
        return

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
