"""Tests for ensemble plan refine guards (tool-only turns, missing scores)."""

from types import SimpleNamespace
from unittest.mock import patch

from machine_learning_engineering.sub_agents.ensemble import agent as ensemble_agent


class _Ctx:
    def __init__(self, state: dict):
        self.state = state


def test_refinement_instruction_skips_unscored_plans():
    ctx = _Ctx(
        {
            "num_solutions": 2,
            "outer_loop_round": 1,
            "num_top_plans": 3,
            "lower": False,
            "ensemble_plans": ["good plan", ""],
            "train_code_1_1": "code1",
            "train_code_1_2": "code2",
            "ensemble_code_exec_result_0": {"score": 0.81, "returncode": 0},
        }
    )
    instruction = ensemble_agent.get_ensemble_plan_refinement_instruction(ctx)
    assert "## Plan: good plan" in instruction
    assert "0.81000" in instruction
    assert "## Plan: \n" not in instruction


def test_refinement_instruction_handles_no_scored_plans():
    ctx = _Ctx(
        {
            "num_solutions": 2,
            "outer_loop_round": 1,
            "lower": True,
            "ensemble_plans": ["plan without score"],
            "train_code_1_1": "code1",
            "train_code_1_2": "code2",
        }
    )
    instruction = ensemble_agent.get_ensemble_plan_refinement_instruction(ctx)
    assert "# Ensemble plans you have tried" in instruction
    assert "## Score:" not in instruction


@patch(
    "machine_learning_engineering.sub_agents.ensemble.agent.common_util.get_text_from_response",
    return_value="",
)
def test_get_refined_ensemble_plan_skips_empty_response(_mock_text):
    state = {"ensemble_plans": ["initial plan"]}
    ctx = SimpleNamespace(state=state)
    ensemble_agent.get_refined_ensemble_plan(ctx, llm_response=None)
    assert state["ensemble_plans"] == ["initial plan"]


@patch(
    "machine_learning_engineering.sub_agents.ensemble.agent.common_util.get_text_from_response",
    return_value="   ",
)
def test_get_init_ensemble_plan_skips_whitespace_only(_mock_text):
    state = {}
    ctx = SimpleNamespace(state=state)
    ensemble_agent.get_init_ensemble_plan(ctx, llm_response=None)
    assert "ensemble_plans" not in state
