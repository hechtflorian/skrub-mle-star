#!/usr/bin/env python3
"""Analyze an MLE-STAR + skrub run from final_state.json, workspace scripts, and ADK log."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# DataOps anchors we expect in skill-integrated pipelines.
DATAOPS_CHECKS: list[tuple[str, str]] = [
    ("skrub.var or skrub.X/y", r"skrub\.(var|X|y)\("),
    ("mark_as_X / mark_as_y", r"\.skb\.(mark_as_X|mark_as_y)\("),
    ("skb.apply", r"\.skb\.apply\("),
    ("skb.apply_func", r"\.skb\.apply_func\("),
    ("make_learner", r"\.skb\.make_learner\("),
    ("predict env dict", r'\.predict\(\{"data"'),
    ("TableVectorizer (when encoding)", r"TableVectorizer\("),
    ("choose_* (tune search)", r"choose_(int|float|from|bool)\("),
    ("make_randomized_search (tune)", r"make_randomized_search\("),
    ("TUNING_BEST_PARAMS (tune search)", r"TUNING_BEST_PARAMS"),
]

STAGE_AGENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "initialization": (
        "task_summarization",
        "model_retriever",
        "model_eval",
        "merger",
        "rank_agent",
        "selection",
        "check_data_use",
    ),
    "refinement": (
        "ablation",
        "init_plan",
        "plan_refine",
        "plan_implement",
    ),
    "tuning": (
        "tune_plan",
        "tune_implement",
        "tune_bake",
    ),
    "ensemble": ("ensemble", "init_ensemble"),
    "submission": ("submission",),
}

STAGE_SCORE_KEYS: dict[str, list[str]] = {
    "initialization": [
        "train_code_exec_result_0_{task}",
        "init_code_exec_result_{task}_1",
    ],
    "refinement": [
        "train_code_exec_result_0_{task}",
        "train_code_improve_exec_result_0_0_{task}",
        "train_code_improve_exec_result_1_0_{task}",
        "train_code_exec_result_1_{task}",
    ],
    "tuning": [
        "train_code_tune_search_exec_result_{task}",
        "train_code_tune_exec_result_{task}",
    ],
    "ensemble": [
        "ensemble_code_exec_result_0",
        "ensemble_code_exec_result_1",
    ],
    "submission": ["submission_code_exec_result"],
}

SCRIPT_CANDIDATES = [
    "1/train0.py",
    "1/train1.py",
    "1/train1_tuned.py",
    "1/train_tune_search.py",
    "1/train_tune_baked.py",
    "1/ablation_0.py",
    "ensemble/ensemble0.py",
    "ensemble/final_solution.py",
]


@dataclass
class StageTiming:
    exec_seconds: float = 0.0
    exec_runs: int = 0


@dataclass
class RunAnalysis:
    run_path: Path
    final_state_path: Path | None = None
    log_path: Path | None = None
    workspace_dir: Path | None = None
    task_id: str = "1"
    lower_is_better: bool = True
    model: str = ""
    scores: dict[str, float | None] = field(default_factory=dict)
    stage_timing: dict[str, StageTiming] = field(default_factory=dict)
    log_agent_counts: dict[str, int] = field(default_factory=dict)
    log_stage_counts: dict[str, int] = field(default_factory=dict)
    log_debug_counts: dict[str, int] = field(default_factory=dict)
    bug_summary_nonempty: dict[str, int] = field(default_factory=dict)
    dataops_by_file: dict[str, dict[str, bool]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def _resolve_paths(run_path: Path) -> tuple[Path | None, Path | None, Path | None]:
    run_path = run_path.resolve()
    final_state: Path | None = None
    workspace: Path | None = None
    log_path: Path | None = None

    candidates = [run_path, run_path / "workspace"]
    if run_path.name == "machine-learning-engineering":
        candidates.append(run_path / "machine_learning_engineering" / "workspace")

    for base in candidates:
        if base.is_file() and base.name == "final_state.json":
            final_state = base
            workspace = base.parent
            break
        direct = base / "final_state.json"
        if direct.is_file():
            final_state = direct
            workspace = base
            break
        for nested in sorted(base.glob("*/final_state.json")):
            final_state = nested
            workspace = nested.parent
            break
        if final_state:
            break

    log_dirs = [
        run_path / "run-logs",
        run_path.parent / "run-logs",
        run_path,
    ]
    logs: list[Path] = []
    for d in log_dirs:
        if d.is_dir():
            logs.extend(d.glob("adk_run_*.log"))
        logs.extend(d.glob("adk_run_*.log") if d.is_file() else [])
    if logs:
        log_path = max(logs, key=lambda p: p.stat().st_mtime)

    return final_state, workspace, log_path


def _stage_for_agent(agent: str) -> str:
    if "ensemble" in agent:
        return "ensemble"
    if agent.startswith("tune_") or "_tune_" in agent:
        return "tuning"
    if "submission" in agent:
        return "submission"
    for stage, prefixes in STAGE_AGENT_PREFIXES.items():
        if stage in ("ensemble", "submission", "tuning"):
            continue
        if any(p in agent for p in prefixes):
            return stage
    if "debug" in agent or "bug_summary" in agent:
        return "debug_unscoped"
    return "other"


def _exec_stage_for_key(key: str) -> str:
    if key.startswith(("init_code", "merger_code", "train_code_0", "model_eval")):
        return "initialization"
    if key.startswith(("ablation_code", "train_code_improve")) or key == "train_code_exec_result":
        return "refinement"
    if "train_code_exec_result_" in key and "tune" not in key:
        return "refinement"
    if "tune" in key:
        return "tuning"
    if key.startswith("ensemble"):
        return "ensemble"
    if key.startswith("submission"):
        return "submission"
    return "other"


def _score_from_result(state: dict, key: str) -> float | None:
    val = state.get(key)
    if isinstance(val, dict) and "score" in val:
        try:
            return float(val["score"])
        except (TypeError, ValueError):
            return None
    return None


def _exec_meta(state: dict, key: str) -> tuple[float | None, float | None, int | None]:
    val = state.get(key, {})
    if not isinstance(val, dict):
        return None, None, None
    score = val.get("score")
    t = val.get("execution_time")
    rc = val.get("returncode")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None
    try:
        t_f = float(t) if t is not None else None
    except (TypeError, ValueError):
        t_f = None
    return score_f, t_f, rc


def _analyze_dataops(path: Path, code: str) -> dict[str, bool]:
    rel = str(path.name)
    return {label: bool(re.search(pat, code)) for label, pat in DATAOPS_CHECKS}


def _parse_log_wall_clock(log_path: Path) -> tuple[str | None, str | None, float | None]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    start_m = re.search(
        r"Script started on (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text
    )
    end_m = re.search(
        r"Script done on (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text
    )
    fmt = "%Y-%m-%d %H:%M:%S"
    start_s = start_m.group(1) if start_m else None
    end_s = end_m.group(1) if end_m else None
    duration = None
    if start_s and end_s:
        duration = (
            datetime.strptime(end_s, fmt) - datetime.strptime(start_s, fmt)
        ).total_seconds()
    return start_s, end_s, duration


def _parse_log_agents(log_path: Path) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(r"\[([^\]:]+)(?::[^\]]+)?\]:")
    agent_counts: dict[str, int] = defaultdict(int)
    stage_counts: dict[str, int] = defaultdict(int)
    debug_counts: dict[str, int] = defaultdict(int)

    for m in pat.finditer(text):
        agent = m.group(1)
        agent_counts[agent] += 1
        stage = _stage_for_agent(agent)
        stage_counts[stage] += 1
        if "bug_summary" in agent or "debug_agent" in agent:
            dbg_stage = stage if stage not in ("debug_unscoped", "other") else "other"
            if "ensemble" in agent:
                dbg_stage = "ensemble"
            elif "tune_" in agent:
                dbg_stage = "tuning"
            elif any(p in agent for p in STAGE_AGENT_PREFIXES["refinement"]):
                dbg_stage = "refinement"
            elif any(p in agent for p in STAGE_AGENT_PREFIXES["initialization"]):
                dbg_stage = "initialization"
            debug_counts[dbg_stage] += 1

    return dict(agent_counts), dict(stage_counts), dict(debug_counts)


def analyze_run(run_path: Path, task_id: str | None = None) -> RunAnalysis:
    final_state_path, workspace, log_path = _resolve_paths(run_path)
    analysis = RunAnalysis(run_path=run_path, final_state_path=final_state_path, log_path=log_path, workspace_dir=workspace)

    if not final_state_path:
        analysis.extras["error"] = "Could not find final_state.json under --path"
        return analysis

    state = json.loads(final_state_path.read_text(encoding="utf-8"))
    analysis.task_id = task_id or str(state.get("num_solutions", 1) and "1")
    analysis.lower_is_better = bool(state.get("lower", True))
    analysis.model = str(state.get("agent_model", "unknown"))

    tid = analysis.task_id
    for stage, keys in STAGE_SCORE_KEYS.items():
        for pattern in keys:
            key = pattern.format(task=tid)
            score = _score_from_result(state, key)
            if score is not None:
                analysis.scores[f"{stage}:{key}"] = score

    # Collect all exec results for timing + score inventory.
    all_scores: list[tuple[str, float]] = []
    for key, val in state.items():
        if not isinstance(val, dict) or "score" not in val:
            continue
        score = _score_from_result(state, key)
        if score is None:
            continue
        all_scores.append((key, score))
        stage = _exec_stage_for_key(key)
        if stage not in analysis.stage_timing:
            analysis.stage_timing[stage] = StageTiming()
        _, t, _ = _exec_meta(state, key)
        if t is not None:
            analysis.stage_timing[stage].exec_seconds += t
            analysis.stage_timing[stage].exec_runs += 1

    analysis.extras["all_scored_exec_keys"] = [
        {"key": k, "score": s} for k, s in sorted(all_scores, key=lambda x: x[1])
    ]

    # Canonical stage summary scores.
    analysis.scores["refinement:best_train"] = _best_of(
        [
            _score_from_result(state, f"train_code_exec_result_0_{tid}"),
            _score_from_result(state, f"train_code_exec_result_1_{tid}"),
        ],
        analysis.lower_is_better,
    )
    analysis.scores["tuning:search"] = _score_from_result(
        state, f"train_code_tune_search_exec_result_{tid}"
    )
    analysis.scores["tuning:bake"] = _score_from_result(
        state, f"train_code_tune_exec_result_{tid}"
    )
    analysis.scores["submission:holdout"] = _score_from_result(
        state, "submission_code_exec_result"
    )

    # All refinement inner-loop improve attempts (iter, step are dynamic).
    improve_scores: dict[str, float] = {}
    improve_pat = re.compile(
        rf"^train_code_improve_exec_result_(\d+)_(\d+)_{re.escape(tid)}$"
    )
    for key in state:
        m = improve_pat.match(key)
        if m:
            s = _score_from_result(state, key)
            if s is not None:
                improve_scores[f"iter{m.group(1)}_step{m.group(2)}"] = s
    analysis.extras["refinement_improve_scores"] = improve_scores

    # All ensemble attempts.
    ensemble_scores: dict[str, float] = {}
    for key in state:
        m = re.match(r"^ensemble_code_exec_result_(\d+)$", key)
        if m:
            s = _score_from_result(state, key)
            if s is not None:
                ensemble_scores[f"ensemble{m.group(1)}"] = s
    analysis.extras["ensemble_scores"] = ensemble_scores

    analysis.extras["tune_param_source"] = state.get(f"tune_param_source_{tid}")
    analysis.extras["tune_winner_source"] = state.get(f"tune_winner_source_{tid}")
    analysis.extras["tune_best_params"] = state.get(f"tune_best_params_{tid}")
    analysis.extras["tune_plan_focus"] = (state.get(f"tune_plan_{tid}") or {}).get(
        "focus_block"
    )
    analysis.extras["table_report_present"] = bool(
        state.get(f"ablation_table_report_profile_{tid}")
    )
    analysis.extras["table_report_chars"] = len(
        state.get(f"ablation_table_report_profile_{tid}", "") or ""
    )

    for key, val in state.items():
        if "bug_summary" in key and isinstance(val, str) and val.strip():
            stage = "other"
            for s, prefixes in STAGE_AGENT_PREFIXES.items():
                if any(p in key for p in prefixes):
                    stage = s
                    break
            analysis.bug_summary_nonempty[stage] = analysis.bug_summary_nonempty.get(stage, 0) + 1

    if workspace:
        for rel in SCRIPT_CANDIDATES:
            p = workspace / rel
            if p.is_file():
                analysis.dataops_by_file[rel] = _analyze_dataops(p, p.read_text(encoding="utf-8"))

    if log_path:
        (
            analysis.extras["log_started"],
            analysis.extras["log_ended"],
            analysis.extras["log_wall_seconds"],
        ) = _parse_log_wall_clock(log_path)
        (
            analysis.log_agent_counts,
            analysis.log_stage_counts,
            analysis.log_debug_counts,
        ) = _parse_log_agents(log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        analysis.extras["contract_violations"] = len(
            re.findall(r"contract violation", log_text, flags=re.IGNORECASE)
        )
        analysis.extras["context_errors"] = len(
            re.findall(r"ContextWindowExceededError", log_text)
        )
        analysis.extras["log_exit_code"] = None
        exit_m = re.search(r"COMMAND_EXIT_CODE=\"(\d+)\"", log_text.splitlines()[-1])
        if exit_m:
            analysis.extras["log_exit_code"] = int(exit_m.group(1))

    return analysis


def to_row(analysis: RunAnalysis) -> dict[str, Any]:
    """Flatten one run analysis into a single comparison-ready row."""
    tid = analysis.task_id
    lower = analysis.lower_is_better
    ex = analysis.extras

    sentinel_seen = False

    def _clean(score: float | None) -> float | None:
        """Drop sentinel failure scores (1e9 / 0) so they don't poison stats."""
        nonlocal sentinel_seen
        if score is None:
            return None
        if (lower and score >= 1e8) or (not lower and score <= 0):
            sentinel_seen = True
            return None
        return score

    init0 = _clean(analysis.scores.get(f"initialization:train_code_exec_result_0_{tid}"))
    refine_promoted = _clean(
        analysis.scores.get(f"refinement:train_code_exec_result_1_{tid}")
    )
    improve_scores = [
        s for s in map(_clean, (ex.get("refinement_improve_scores") or {}).values())
        if s is not None
    ]
    ensemble_scores = [
        s for s in map(_clean, (ex.get("ensemble_scores") or {}).values())
        if s is not None
    ]
    tune_bake = _clean(analysis.scores.get("tuning:bake"))
    submission = _clean(analysis.scores.get("submission:holdout"))
    final_best = _best_of(
        [refine_promoted, tune_bake, *ensemble_scores, submission], lower
    )

    def gain(before: float | None, after: float | None) -> float | None:
        """Signed improvement (positive = better), metric-direction aware."""
        if before is None or after is None:
            return None
        return before - after if lower else after - before

    exec_total = sum(t.exec_seconds for t in analysis.stage_timing.values())
    dataops_fracs = [
        sum(checks.values()) / len(checks)
        for checks in analysis.dataops_by_file.values()
        if checks
    ]
    return {
        "run_path": str(analysis.run_path),
        "model": analysis.model,
        "lower_is_better": lower,
        "score_init": init0,
        "score_refine_promoted": refine_promoted,
        "score_refine_improve_best": _best_of(improve_scores, lower),
        "score_tune_search": _clean(analysis.scores.get("tuning:search")),
        "score_tune_bake": tune_bake,
        "score_ensemble_best": _best_of(ensemble_scores, lower),
        "score_submission": submission,
        "score_final_best": final_best,
        "gain_refinement": gain(init0, refine_promoted),
        "gain_tuning": gain(refine_promoted, tune_bake),
        "gain_total": gain(init0, final_best),
        "tune_param_source": ex.get("tune_param_source"),
        "tune_winner_source": ex.get("tune_winner_source"),
        "tune_plan_focus": ex.get("tune_plan_focus"),
        "table_report_chars": ex.get("table_report_chars"),
        "debug_refinement": analysis.log_debug_counts.get("refinement", 0),
        "debug_tuning": analysis.log_debug_counts.get("tuning", 0),
        "debug_ensemble": analysis.log_debug_counts.get("ensemble", 0),
        "contract_violations": ex.get("contract_violations"),
        "context_errors": ex.get("context_errors"),
        "had_sentinel_failure": sentinel_seen,
        "wall_seconds": ex.get("log_wall_seconds"),
        "exec_seconds": round(exec_total, 1),
        "dataops_adherence": (
            round(sum(dataops_fracs) / len(dataops_fracs), 3)
            if dataops_fracs
            else None
        ),
        "log_exit_code": ex.get("log_exit_code"),
    }


def _best_of(scores: list[float | None], lower: bool) -> float | None:
    vals = [s for s in scores if s is not None]
    if not vals:
        return None
    return min(vals) if lower else max(vals)


def _fmt_score(score: float | None, lower: bool) -> str:
    if score is None:
        return "n/a"
    if lower and score >= 1e8:
        return f"{score:.3e} (sentinel/fail)"
    return f"{score:.4f}"


def _print_report(analysis: RunAnalysis) -> None:
    print("=" * 72)
    print("MLE-STAR Run Analysis")
    print("=" * 72)
    print(f"Path:         {analysis.run_path}")
    print(f"final_state:  {analysis.final_state_path or 'NOT FOUND'}")
    print(f"log:          {analysis.log_path or 'NOT FOUND'}")
    print(f"workspace:    {analysis.workspace_dir or 'NOT FOUND'}")
    print(f"model:        {analysis.model}")
    print(f"metric:       RMSE (lower is better={analysis.lower_is_better})")

    if analysis.extras.get("error"):
        print(f"\nERROR: {analysis.extras['error']}")
        return

    lower = analysis.lower_is_better
    print("\n--- Stage scores (holdout metric from exec results) ---")
    init0 = analysis.scores.get(f"initialization:train_code_exec_result_0_{analysis.task_id}")
    ref1 = analysis.scores.get(f"refinement:train_code_exec_result_1_{analysis.task_id}")
    tune_s = analysis.scores.get("tuning:search")
    tune_b = analysis.scores.get("tuning:bake")
    sub = analysis.scores.get("submission:holdout")
    improve_scores = analysis.extras.get("refinement_improve_scores") or {}
    ensemble_scores = analysis.extras.get("ensemble_scores") or {}

    print(f"  Init baseline (train0):       {_fmt_score(init0, lower)}")
    for label, s in sorted(improve_scores.items()):
        print(f"  Refinement improve {label}: {_fmt_score(s, lower)}")
    print(f"  Refinement promoted (train1): {_fmt_score(ref1, lower)}")
    print(f"  Tuning search script:         {_fmt_score(tune_s, lower)}")
    print(f"  Tuning baked script:          {_fmt_score(tune_b, lower)}")
    for label, s in sorted(ensemble_scores.items()):
        print(f"  Ensemble {label}:            {_fmt_score(s, lower)}")
    print(f"  Submission script holdout:    {_fmt_score(sub, lower)}  <-- final artifact metric line")

    print("\n--- Tuning / TableReport novelty ---")
    print(f"  tune_param_source:          {analysis.extras.get('tune_param_source')}")
    print(f"  tune_winner_source:         {analysis.extras.get('tune_winner_source')}")
    print(f"  tune_plan focus:            {analysis.extras.get('tune_plan_focus')}")
    print(f"  tune_best_params:           {analysis.extras.get('tune_best_params')}")
    print(f"  table_report in state:      {analysis.extras.get('table_report_present')} ({analysis.extras.get('table_report_chars')} chars)")

    def _delta_line(label: str, before: float | None, after: float | None) -> None:
        if before is None or after is None:
            return
        delta = (before - after) if lower else (after - before)
        print(f"  {label} {delta:+.4f} ({100 * delta / before:.2f}% relative, positive = better)")

    print("\n--- Stage gains ---")
    _delta_line("init -> refinement promoted:", init0, ref1)
    _delta_line("refinement -> tuned bake:   ", ref1, tune_b)
    _delta_line("init -> submission:         ", init0, sub)
    if analysis.extras.get("contract_violations") is not None:
        print(f"  contract violations in log: {analysis.extras['contract_violations']}")
        print(f"  context window errors:      {analysis.extras['context_errors']}")

    print("\n--- Code execution time (summed from final_state exec results) ---")
    total_exec = 0.0
    for stage in ["initialization", "refinement", "tuning", "ensemble", "submission", "other"]:
        st = analysis.stage_timing.get(stage)
        if not st or st.exec_runs == 0:
            continue
        total_exec += st.exec_seconds
        print(f"  {stage:14s} {st.exec_seconds:8.1f}s  ({st.exec_runs} runs)")
    print(f"  {'TOTAL (exec only)':14s} {total_exec:8.1f}s")

    if analysis.extras.get("log_wall_seconds") is not None:
        print("\n--- Wall clock (from ADK log) ---")
        print(f"  started:  {analysis.extras.get('log_started')}")
        print(f"  ended:    {analysis.extras.get('log_ended')}")
        print(f"  duration: {analysis.extras.get('log_wall_seconds'):.0f}s")
        print(f"  exit:     {analysis.extras.get('log_exit_code')}")

    print("\n--- Debug / loop pressure (from log agent lines) ---")
    for stage in ["initialization", "refinement", "tuning", "ensemble", "submission"]:
        total = analysis.log_stage_counts.get(stage, 0)
        dbg = analysis.log_debug_counts.get(stage, 0)
        bugs = analysis.bug_summary_nonempty.get(stage, 0)
        print(f"  {stage:14s} log_events={total:3d}  debug/bug_lines={dbg:3d}  nonempty_bug_summaries={bugs}")

    print("\n--- DataOps adherence (workspace scripts) ---")
    for rel, checks in sorted(analysis.dataops_by_file.items()):
        hits = sum(checks.values())
        print(f"  {rel}: {hits}/{len(checks)} checks")
        for label, ok in checks.items():
            if not ok and rel in ("1/train_tune_search.py", "1/train1.py", "ensemble/final_solution.py"):
                print(f"    - missing: {label}")

    print("\n--- Key artifacts ---")
    if analysis.workspace_dir:
        for rel in SCRIPT_CANDIDATES:
            p = analysis.workspace_dir / rel
            if p.is_file():
                print(f"  {rel} ({p.stat().st_size} bytes)")

    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Run bundle dir: project root, workspace/<task>, or dir containing final_state.json",
    )
    parser.add_argument("--task-id", type=str, default=None, help="Solution id (default: 1)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    analysis = analyze_run(args.path, task_id=args.task_id)
    if args.json:
        payload = {
            "row": to_row(analysis),
            "scores": analysis.scores,
            "extras": analysis.extras,
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        _print_report(analysis)


if __name__ == "__main__":
    main()
