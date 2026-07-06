#!/usr/bin/env python3
"""Execute MLE-STAR experiment runs and archive artifacts.

This script runs agents only. Use evaluate.py --summarize-only to analyze
archived runs and write aggregated reports. This script is a wrapper around evaluate.py to seperate concerns.

Examples:
  python automated_evaluation/run_experiments.py --dry-run
  python automated_evaluation/run_experiments.py --tasks spaceship-titanic --systems improved
  python automated_evaluation/run_experiments.py --repeat-count 3 --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluate import (
    MANIFEST_PATH,
    REPO_ROOT,
    _utc_stamp,
    build_run_parser,
    execute_evaluation_matrix,
    parse_config,
    scan_tasks,
    summarize_run_outcomes,
    task_status_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_run_parser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output: no live ADK log stream or per-run status lines",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summarize_only = getattr(args, "summarize_only", False)
    if summarize_only:
        print(
            "Use evaluate.py --summarize-only for result analysis.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    eval_stamp = _utc_stamp()
    config = parse_config(args, eval_stamp=eval_stamp)
    config.summarize_only = False
    config.verbose = not args.quiet
    config.live_log = not args.quiet

    tasks = scan_tasks()
    if config.tasks:
        wanted = set(config.tasks)
        tasks = [task for task in tasks if task.task_name in wanted]
    task_status = task_status_payload(tasks)

    if not MANIFEST_PATH.is_file():
        print(
            "Warning: tasks_manifest.json not found. "
            "Run generate_tasks_manifest.py first.",
            file=sys.stderr,
        )

    config.runs_root.mkdir(parents=True, exist_ok=True)
    experiment_meta = {
        "started_at_utc": eval_stamp,
        "runs_root": str(config.runs_root),
        "tasks": config.tasks or [t.task_name for t in tasks],
        "systems": config.systems,
        "repeats": config.repeats,
        "seed": config.seed,
        "model_label": config.model_label,
        "uv_sync_before_run": config.uv_sync_before_run,
        "sync_tasks_to_vanilla": config.sync_tasks_to_vanilla,
        "task_status": task_status,
    }
    (config.runs_root / "experiment_meta.json").write_text(
        json.dumps(experiment_meta, indent=2),
        encoding="utf-8",
    )

    run_log = execute_evaluation_matrix(config)
    outcomes = summarize_run_outcomes(run_log)

    print(
        f"\nExperiment batch complete under {config.runs_root}\n"
        f"  Success: {outcomes['ok']} · Failed: {outcomes['failed']} · "
        f"Skipped: {outcomes['skipped']} · Planned: {outcomes['planned']}"
    )
    print(f"  Run matrix: {config.runs_root / 'run_matrix.json'}")
    if not config.dry_run:
        print(
            "\nNext: analyze results with\n"
            f"  python automated_evaluation/evaluate.py --summarize-only "
            f"--runs-root {config.runs_root.relative_to(REPO_ROOT)}"
        )


if __name__ == "__main__":
    main()
