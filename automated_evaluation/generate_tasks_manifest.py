#!/usr/bin/env python3
"""Generate tasks_manifest.json from bundled task packs under tasks/.

Run this once before experiment batches (or after adding/updating task packs).
The manifest is the single source of truth for task metadata used by
run_experiments.py and evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = (
    REPO_ROOT
    / "agents"
    / "machine-learning-engineering"
    / "machine_learning_engineering"
    / "tasks"
)
MANIFEST_PATH = Path(__file__).resolve().parent / "tasks_manifest.json"

LOWER_IS_BETTER_METRICS = {
    "root_mean_squared_error",
    "root_mean_squared_log_error",
    "mean_absolute_error",
    "mean_squared_error",
    "mean_squared_log_error",
}

HIGHER_IS_BETTER_METRICS = {
    "accuracy_score",
    "roc_auc_score",
    "f1_score",
    "r2_score",
}

TASK_OVERRIDES: dict[str, dict[str, Any]] = {
    "spaceship-titanic": {
        "task_type": "Tabular Classification",
        "metric": "accuracy_score",
        "lower": False,
        "target_column": "Transported",
        "id_column": "PassengerId",
    },
    "abalone-regression": {
        "task_type": "Tabular Regression",
        "metric": "root_mean_squared_log_error",
        "lower": True,
        "target_column": "Rings",
        "id_column": "id",
    },
    "blueberry-yield-regression": {
        "task_type": "Tabular Regression",
        "metric": "mean_absolute_error",
        "lower": True,
        "target_column": "yield",
        "id_column": "id",
    },
    "calories-burned-regression": {
        "task_type": "Tabular Regression",
        "metric": "root_mean_squared_log_error",
        "lower": True,
        "target_column": "Calories",
        "id_column": "id",
    },
    "covid19-forecasting-regression": {
        "task_type": "Tabular Regression",
        "metric": "root_mean_squared_log_error",
        "lower": True,
        "target_column": "ConfirmedCases,Fatalities",
        "id_column": "ForecastId",
        "notes": "Multi-target regression (ConfirmedCases + Fatalities).",
    },
    "podcast-listening-hours-regression": {
        "task_type": "Tabular Regression",
        "metric": "root_mean_squared_error",
        "lower": True,
        "target_column": "Listening_Time_minutes",
        "id_column": "id",
    },
    "diabetes-classification": {
        "task_type": "Tabular Classification",
        "metric": "roc_auc_score",
        "lower": False,
        "target_column": "diagnosed_diabetes",
        "id_column": "id",
    },
    "introverts-extroverts-classification": {
        "task_type": "Tabular Classification",
        "metric": "accuracy_score",
        "lower": False,
        "target_column": "Personality",
        "id_column": "id",
    },
    "bank-dataset-classification": {
        "task_type": "Tabular Classification",
        "metric": "roc_auc_score",
        "lower": False,
        "target_column": "y",
        "id_column": "id",
    },
    "multi-class-pred-obesity-risk": {
        "task_type": "Tabular Classification",
        "metric": "accuracy_score",
        "lower": False,
        "target_column": "NObeyesdad",
        "id_column": "id",
    },
}


def _parse_metric(text: str) -> str:
    match = re.search(r"(?m)^#\s*Metric\s*\n+\s*([^\n#]+)", text)
    if not match:
        return ""
    return match.group(1).strip()


def _infer_task_type(task_name: str, metric: str) -> str:
    if task_name.endswith("-regression"):
        return "Tabular Regression"
    if task_name.endswith("-classification"):
        return "Tabular Classification"
    if metric in LOWER_IS_BETTER_METRICS:
        return "Tabular Regression"
    return "Tabular Classification"


def _infer_lower(metric: str, task_type: str) -> bool:
    if metric in LOWER_IS_BETTER_METRICS:
        return True
    if metric in HIGHER_IS_BETTER_METRICS:
        return False
    return task_type == "Tabular Regression"


def _task_ready(task_dir: Path) -> bool:
    return (
        (task_dir / "train.csv").is_file()
        and (task_dir / "test.csv").is_file()
        and (task_dir / "task_description.txt").is_file()
    )


def build_task_spec(task_name: str, task_dir: Path) -> dict[str, Any]:
    override = TASK_OVERRIDES.get(task_name, {})
    description = ""
    if (task_dir / "task_description.txt").is_file():
        description = (task_dir / "task_description.txt").read_text(encoding="utf-8")

    metric = override.get("metric") or _parse_metric(description)
    task_type = override.get("task_type") or _infer_task_type(task_name, metric)
    lower = override.get("lower", _infer_lower(metric, task_type))

    spec: dict[str, Any] = {
        "task_name": task_name,
        "task_type": task_type,
        "metric": metric,
        "lower": lower,
        "target_column": override.get("target_column"),
        "id_column": override.get("id_column"),
        "source": "bundled",
        "ready": _task_ready(task_dir),
    }
    if notes := override.get("notes"):
        spec["notes"] = notes
    return spec


def generate_manifest(
    tasks_dir: Path = TASKS_DIR,
    *,
    only: list[str] | None = None,
    default_seed: int = 42,
) -> dict[str, Any]:
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    tasks: list[dict[str, Any]] = []
    for entry in sorted(tasks_dir.iterdir()):
        if not entry.is_dir():
            continue
        task_name = entry.name
        if only and task_name not in only:
            continue
        tasks.append(build_task_spec(task_name, entry))

    if not tasks:
        raise SystemExit("No task folders matched.")

    return {
        "version": 1,
        "description": (
            "Bundled MLE-STAR benchmark tasks under "
            "machine_learning_engineering/tasks/ (regression + classification mix)."
        ),
        "default_seed": default_seed,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to write tasks_manifest.json",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=TASKS_DIR,
        help="Directory containing task folders",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Generate manifest entries for a subset of task folder names",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifest JSON to stdout instead of writing",
    )
    args = parser.parse_args()

    manifest = generate_manifest(
        args.tasks_dir,
        only=args.only,
        default_seed=args.seed,
    )
    payload = json.dumps(manifest, indent=2) + "\n"
    if args.dry_run:
        print(payload, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    ready = sum(1 for task in manifest["tasks"] if task.get("ready"))
    print(
        f"Wrote {args.output} ({len(manifest['tasks'])} tasks, "
        f"{ready} ready with train/test/description)."
    )


if __name__ == "__main__":
    main()
