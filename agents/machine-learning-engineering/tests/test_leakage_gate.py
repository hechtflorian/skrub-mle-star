"""Tests for data leakage checker tool-only turn guards."""

from types import SimpleNamespace
from unittest.mock import patch

from machine_learning_engineering.shared_libraries import check_leakage_util


class _State(dict):
    def get(self, key, default=None):
        return super().get(key, default)


@patch(
    "machine_learning_engineering.shared_libraries.check_leakage_util.common_util.get_text_from_response",
    return_value="",
)
def test_update_extract_status_handles_empty_tool_only_response(_mock_text):
    state = _State({"init_code_1_1": "x = 1"})
    ctx = SimpleNamespace(agent_name="model_eval_check_leakage_agent_1_1", state=state)
    check_leakage_util.update_extract_status(ctx, llm_response=None, prefix="model_eval")
    assert state["model_eval_extract_status_1_1"] is False
    assert state["model_eval_leakage_status_1_1"] == ""
    assert state["model_eval_leakage_block_1_1"] == ""


def test_update_extract_status_parses_no_leakage_json():
    state = _State({"init_code_1_1": "data_train = skrub.var('data', train_part)"})
    ctx = SimpleNamespace(agent_name="model_eval_check_leakage_agent_1_1", state=state)
    response = '[{"leakage_status": "No Data Leakage", "code_block": ""}]'
    with patch(
        "machine_learning_engineering.shared_libraries.check_leakage_util.common_util.get_text_from_response",
        return_value=response,
    ):
        check_leakage_util.update_extract_status(
            ctx, llm_response=None, prefix="model_eval"
        )
    assert state["model_eval_extract_status_1_1"] is True
    assert state["model_eval_leakage_status_1_1"] == "No Data Leakage"


@patch(
    "machine_learning_engineering.shared_libraries.check_leakage_util.common_util.get_text_from_response",
    return_value="",
)
def test_replace_leakage_code_skips_empty_response(_mock_text):
    state = _State({"init_code_1_1": "original"})
    ctx = SimpleNamespace(agent_name="model_eval_refine_leakage_agent_1_1", state=state)
    check_leakage_util.replace_leakage_code(ctx, llm_response=None, prefix="model_eval")
    assert state["init_code_1_1"] == "original"
