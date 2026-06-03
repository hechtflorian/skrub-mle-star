#!/usr/bin/env python3
"""Compares two MLE-STAR run artifacts for DataOps adherence and quality."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

ANCHORS = [
    r"skrub\.var\(",
    r"skrub\.X\(",
    r"skrub\.y\(",
    r"\.skb\.mark_as_X\(",
    r"\.skb\.mark_as_y\(",
    r"\.skb\.apply\(",
    r"choose_from\(",
    r"choose_",
]


@dataclass
class RunMetrics:
    label: str
    model: str
    score: float | None
    retry_proxy: int
    anchor_hits: int
    anchor_total: int
    has_sklearn_pipeline: bool


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_code(state: dict) -> str:
    for key in (
        "submission_code",
        "ensemble_code_1",
        "ensemble_code_0",
        "train_code_1_1",
        "train_code_0_1",
    ):
        value = state.get(key, "")
        if value:
            return value
    return ""


def _pick_score(state: dict) -> float | None:
    for key in (
        "submission_code_exec_result",
        "ensemble_code_exec_result_1",
        "ensemble_code_exec_result_0",
    ):
        value = state.get(key, {})
        if isinstance(value, dict) and "score" in value:
            return value["score"]
    return state.get("best_score_1")


def _retry_proxy(state: dict) -> int:
    return len(
        [
            k
            for k, v in state.items()
            if "bug_summary" in k and isinstance(v, str) and v.strip()
        ]
    )


def _compute_metrics(label: str, state: dict) -> RunMetrics:
    code = _pick_code(state)
    anchor_hits = sum(bool(re.search(p, code)) for p in ANCHORS)
    return RunMetrics(
        label=label,
        model=state.get("agent_model", "unknown"),
        score=_pick_score(state),
        retry_proxy=_retry_proxy(state),
        anchor_hits=anchor_hits,
        anchor_total=len(ANCHORS),
        has_sklearn_pipeline=("Pipeline(" in code or "sklearn.pipeline" in code),
    )


def _token_estimate(path: Path) -> float:
    return len(path.read_text(encoding="utf-8")) / 4.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--skill-md", type=Path, required=True)
    parser.add_argument("--skill-ref-dir", type=Path, required=True)
    args = parser.parse_args()

    run_a = _compute_metrics(args.label_a, _load_state(args.run_a))
    run_b = _compute_metrics(args.label_b, _load_state(args.run_b))
    l2_tokens = _token_estimate(args.skill_md)
    l3_tokens = sum(_token_estimate(p) for p in args.skill_ref_dir.glob("*.md"))

    print(
        json.dumps(
            {
                "run_a": run_a.__dict__,
                "run_b": run_b.__dict__,
                "skill_context_token_estimate": {
                    "l2_skill_md": round(l2_tokens, 1),
                    "l3_reference_bundle": round(l3_tokens, 1),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
