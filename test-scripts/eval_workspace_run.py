#!/usr/bin/env python3
"""Lightweight evaluation for one workspace run.

This script is intentionally simple: it reads one `final_state.json` artifact and
reports whether the latest run looks healthy and DataOps-aligned.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATAOPS_ANCHORS: dict[str, str] = {
    "skrub_var": r"skrub\.var\(",
    "skrub_x": r"skrub\.X\(",
    "skrub_y": r"skrub\.y\(",
    "mark_as_x": r"\.skb\.mark_as_X\(",
    "mark_as_y": r"\.skb\.mark_as_y\(",
    "skb_apply": r"\.skb\.apply\(",
    "choose": r"choose_(from|int|float|bool)\(",
}


@dataclass
class ExecStats:
    total: int
    success: int
    failed: int
    scored: int
    success_without_score: int
    sentinel_scores: int
    best_score: float | None
    worst_score: float | None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_dicts_with_paths(value: Any, path: str = "$"):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield from _iter_dicts_with_paths(child, child_path)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            child_path = f"{path}[{i}]"
            yield from _iter_dicts_with_paths(child, child_path)


def _is_exec_result_dict(value: dict[str, Any]) -> bool:
    return "returncode" in value and (
        "stdout" in value or "stderr" in value or "execution_time" in value
    )


def _extract_exec_results(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    results: list[tuple[str, dict[str, Any]]] = []
    for path, value in _iter_dicts_with_paths(state):
        if isinstance(value, dict) and _is_exec_result_dict(value):
            results.append((path, value))
    return results


def _collect_scores(exec_results: list[tuple[str, dict[str, Any]]]) -> list[float]:
    scores: list[float] = []
    for _, result in exec_results:
        score = result.get("score")
        if isinstance(score, (int, float)) and math.isfinite(score):
            scores.append(float(score))
    return scores


def _pick_primary_code(state: dict[str, Any]) -> tuple[str, str]:
    if isinstance(state.get("submission_code"), str) and state["submission_code"].strip():
        return "submission_code", state["submission_code"]

    ensemble_candidates: list[tuple[int, str]] = []
    for key, value in state.items():
        if not isinstance(value, str) or not value.strip():
            continue
        m = re.fullmatch(r"ensemble_code_(\d+)", key)
        if m:
            ensemble_candidates.append((int(m.group(1)), key))
    if ensemble_candidates:
        _, selected_key = max(ensemble_candidates, key=lambda x: x[0])
        return selected_key, state[selected_key]

    train_candidates: list[tuple[int, int, str]] = []
    for key, value in state.items():
        if not isinstance(value, str) or not value.strip():
            continue
        m = re.fullmatch(r"train_code_(\d+)_(\d+)", key)
        if m:
            train_candidates.append((int(m.group(1)), int(m.group(2)), key))
    if train_candidates:
        _, _, selected_key = max(train_candidates, key=lambda x: (x[0], x[1]))
        return selected_key, state[selected_key]

    for key, value in state.items():
        if key.endswith("_code") and isinstance(value, str) and value.strip():
            return key, value

    return "", ""


def _evaluate_dataops(code: str) -> dict[str, Any]:
    if not code:
        return {
            "anchor_hits": 0,
            "anchor_total": len(DATAOPS_ANCHORS),
            "anchors": {k: False for k in DATAOPS_ANCHORS},
            "has_dataops_core": False,
            "has_sklearn_pipeline": False,
        }
    anchors = {
        name: bool(re.search(pattern, code)) for name, pattern in DATAOPS_ANCHORS.items()
    }
    has_dataops_core = anchors["skb_apply"] and (
        anchors["skrub_var"] or anchors["skrub_x"] or anchors["mark_as_x"]
    )
    return {
        "anchor_hits": sum(anchors.values()),
        "anchor_total": len(anchors),
        "anchors": anchors,
        "has_dataops_core": has_dataops_core,
        "has_sklearn_pipeline": ("Pipeline(" in code or "sklearn.pipeline" in code),
    }


def _evaluate_python_artifacts(run_dir: Path) -> dict[str, int]:
    # Keep this simple and robust across task layouts.
    py_files = list(run_dir.rglob("*.py"))
    ablation_files = [p for p in py_files if re.fullmatch(r"ablation_\d+\.py", p.name)]
    improve_files = [
        p for p in py_files if re.fullmatch(r"train\d+_improve\d+\.py", p.name)
    ]
    return {
        "python_files_total": len(py_files),
        "ablation_files_total": len(ablation_files),
        "ablation_files_non_empty": sum(p.stat().st_size > 0 for p in ablation_files),
        "improve_files_total": len(improve_files),
        "improve_files_non_empty": sum(p.stat().st_size > 0 for p in improve_files),
    }


def _build_exec_stats(exec_results: list[tuple[str, dict[str, Any]]], lower: bool) -> ExecStats:
    success = sum(result.get("returncode", 1) == 0 for _, result in exec_results)
    failed = len(exec_results) - success
    scored = sum("score" in result for _, result in exec_results)
    success_without_score = sum(
        (result.get("returncode", 1) == 0 and "score" not in result)
        for _, result in exec_results
    )
    sentinel_scores = sum(
        result.get("score") in (1_000_000_000.0, 0.0) for _, result in exec_results
    )
    scores = _collect_scores(exec_results)
    if not scores:
        best_score = None
        worst_score = None
    else:
        best_score = min(scores) if lower else max(scores)
        worst_score = max(scores) if lower else min(scores)
    return ExecStats(
        total=len(exec_results),
        success=success,
        failed=failed,
        scored=scored,
        success_without_score=success_without_score,
        sentinel_scores=sentinel_scores,
        best_score=best_score,
        worst_score=worst_score,
    )


def _derive_flags(exec_stats: ExecStats, dataops_eval: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if exec_stats.total == 0:
        flags.append("no_exec_results_found")
    if exec_stats.success_without_score > 0:
        flags.append("successful_runs_without_score")
    if exec_stats.sentinel_scores > 0:
        flags.append("sentinel_score_detected")
    if not dataops_eval["has_dataops_core"]:
        flags.append("dataops_core_pattern_missing")
    if dataops_eval["has_sklearn_pipeline"]:
        flags.append("sklearn_pipeline_detected")
    return flags


def _print_human_summary(report: dict[str, Any]) -> None:
    print("\nRun summary")
    print("-----------")
    print(f"Task: {report['task_name']}")
    print(f"Model: {report['agent_model']}")
    print(f"Primary code key: {report['primary_code_key'] or 'N/A'}")
    print(f"Best score observed: {report['exec_stats']['best_score']}")
    print(f"Exec results: {report['exec_stats']['total']} total")
    print(
        "  - success/failed: "
        f"{report['exec_stats']['success']}/{report['exec_stats']['failed']}"
    )
    print(
        "  - scored/success_without_score: "
        f"{report['exec_stats']['scored']}/{report['exec_stats']['success_without_score']}"
    )
    print(
        "DataOps anchors: "
        f"{report['dataops']['anchor_hits']}/{report['dataops']['anchor_total']}"
    )
    print(f"Has DataOps core pattern: {report['dataops']['has_dataops_core']}")
    print(f"Has sklearn pipeline markers: {report['dataops']['has_sklearn_pipeline']}")
    print(f"Flags: {', '.join(report['flags']) if report['flags'] else 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one workspace run from final_state.json artifacts."
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(
            "agents/machine-learning-engineering/machine_learning_engineering/workspace/"
            "california-housing-prices/final_state.json"
        ),
        help="Path to final_state.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for JSON summary.",
    )
    args = parser.parse_args()

    state_path = args.state.resolve()
    state = _load_json(state_path)
    run_dir = state_path.parent

    lower = bool(state.get("lower", True))
    exec_results = _extract_exec_results(state)
    exec_stats = _build_exec_stats(exec_results, lower=lower)
    primary_code_key, primary_code = _pick_primary_code(state)
    dataops_eval = _evaluate_dataops(primary_code)
    artifact_eval = _evaluate_python_artifacts(run_dir)
    flags = _derive_flags(exec_stats, dataops_eval)

    report = {
        "task_name": state.get("task_name", "unknown"),
        "agent_model": state.get("agent_model", "unknown"),
        "state_path": str(state_path),
        "primary_code_key": primary_code_key,
        "exec_stats": exec_stats.__dict__,
        "dataops": dataops_eval,
        "artifacts": artifact_eval,
        "flags": flags,
    }

    print(json.dumps(report, indent=2))
    _print_human_summary(report)

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote summary to: {output_path}")


if __name__ == "__main__":
    main()
