#!/usr/bin/env python3
"""Legacy helper: aggregate run analyses into a CSV / table.

Prefer `python automated_evaluation/evaluate.py --summarize-only --runs-root ...`
for batch reports. This script remains useful for quick CSV exports.

Examples:
    python test-scripts/aggregate_runs.py --root automated_evaluation/runs/<stamp>
    python test-scripts/aggregate_runs.py --root automated_evaluation/runs/<stamp> --out runs.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_run import analyze_run, to_row  # noqa: E402

LABEL_COLS = ["task", "variant", "run"]
SUMMARY_COLS = [
    "metric",
    "primary_score",
    "score_submission_source",
    "gain_refinement",
    "gain_tuning",
    "gain_ensemble",
    "gain_total",
    "refine_promoted_over_init",
    "tune_ran",
    "debug_total",
    "skill_tool_calls",
    "agent_calls_total",
    "wall_seconds",
    "dataops_adherence",
]


def find_run_dirs(root: Path) -> list[Path]:
    seen: set[Path] = set()
    dirs: list[Path] = []
    for fs in sorted(root.rglob("final_state.json")):
        d = fs.parent
        # A workspace dir nested inside an already-found run bundle counts
        # as the same run; keep the shallowest dir per bundle.
        if any(parent in seen for parent in d.parents):
            continue
        seen.add(d)
        dirs.append(d)
    return dirs


def labels_for(run_dir: Path, root: Path) -> dict[str, str]:
    try:
        parts = run_dir.resolve().relative_to(root.resolve()).parts
    except ValueError:
        parts = (run_dir.name,)
    if not parts:
        parts = (run_dir.name,)
    task = parts[0]
    variant = "/".join(parts[1:-1]) if len(parts) > 2 else (parts[1] if len(parts) == 2 else "")
    run = parts[-1] if len(parts) > 1 else ""
    return {"task": task, "variant": variant, "run": run}


def collect_rows(root: Path) -> list[dict]:
    rows = []
    for run_dir in find_run_dirs(root):
        analysis = analyze_run(run_dir)
        if analysis.extras.get("error"):
            print(f"  skip (no usable state): {run_dir}", file=sys.stderr)
            continue
        rows.append({**labels_for(run_dir, root), **to_row(analysis)})
    return rows


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def print_table(rows: list[dict]) -> None:
    cols = LABEL_COLS + SUMMARY_COLS
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def print_group_summary(rows: list[dict]) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["task"], r["variant"]), []).append(r)
    if all(len(g) < 2 for g in groups.values()):
        return
    print("\nPer (task, variant) summary:")
    for (task, variant), g in sorted(groups.items()):
        subs = [r["primary_score"] for r in g if r.get("primary_score") is not None]
        walls = [r["wall_seconds"] for r in g if r.get("wall_seconds") is not None]
        mean_sub = statistics.mean(subs) if subs else None
        std_sub = statistics.stdev(subs) if len(subs) > 1 else None
        mean_wall = statistics.mean(walls) if walls else None
        print(
            f"  {task} / {variant or '-'}: n={len(g)}"
            f"  submission={_fmt(mean_sub)}"
            + (f" ±{std_sub:.2f}" if std_sub is not None else "")
            + f"  wall={_fmt(mean_wall)}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("submissions"),
        help="Directory tree to scan for run bundles (default: submissions/)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write full rows to this CSV")
    args = parser.parse_args()

    if not args.root.is_dir():
        sys.exit(f"Root directory not found: {args.root}")

    rows = collect_rows(args.root)
    if not rows:
        sys.exit(f"No run bundles (final_state.json) found under {args.root}")

    print(f"Found {len(rows)} runs under {args.root}\n")
    print_table(rows)
    print_group_summary(rows)

    if args.out:
        fieldnames = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

#python test-scripts/aggregate_runs.py --root submissions --out runs.csv