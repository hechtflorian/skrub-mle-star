"""Tests for optional tuning stage skip / failure gates."""

from machine_learning_engineering.sub_agents.tuning import agent as tuning_agent


def test_mark_tune_skipped_plan_sets_state():
    state = {}
    tuning_agent.mark_tune_skipped_plan(state, "1", "flat ablation")
    assert state["tune_stage_status_1"] == "skipped_plan"
    assert state["tune_skip_reason_1"] == "flat ablation"
    assert state["tune_winner_source_1"] == "structural"
    assert state["tune_param_source_1"] == "skipped"
    assert state["train_code_tune_search_exec_result_1"]["skipped_plan"] is True


def test_mark_tune_search_failed_sets_state():
    state = {}
    tuning_agent.mark_tune_search_failed(state, "1", "NumericChoice error")
    assert state["tune_stage_status_1"] == "failed_search"
    assert "NumericChoice" in state["tune_skip_reason_1"]
    assert state["train_code_tune_exec_result_1"]["skipped_search_failed"] is True


def test_search_succeeded_requires_score_and_params():
    state = {
        "train_code_tune_search_exec_result_1": {
            "returncode": 0,
            "score": 0.8,
        },
        "tune_best_params_1": {"lr": 0.03},
    }
    assert tuning_agent._search_succeeded(state, "1")

    state["train_code_tune_search_exec_result_1"]["skipped_plan"] = True
    assert not tuning_agent._search_succeeded(state, "1")
