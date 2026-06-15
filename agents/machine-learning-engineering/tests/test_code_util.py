"""Tests for code_util tuning param mapping."""

from machine_learning_engineering.shared_libraries import code_util


def test_map_tuning_best_params_preserves_human_named_keys():
    raw = {"max_depth": 5, "learning_rate": 0.08}
    plan = {
        "tunable_params": [
            {"name": "max_depth", "kind": "choose_int", "low": 3, "high": 6},
            {"name": "learning_rate", "kind": "choose_float", "low": 0.03, "high": 0.12},
        ]
    }
    assert code_util.map_tuning_best_params(raw, plan) == raw


def test_map_tuning_best_params_matches_by_value_not_data_op_order():
    """Spaceship Titanic run1: data_op index order != plan param order."""
    raw = {"data_op__0": 0.08276096024435112, "data_op__1": 5}
    plan = {
        "tunable_params": [
            {"name": "max_depth", "kind": "choose_int", "low": 3, "high": 6, "default": 3},
            {
                "name": "learning_rate",
                "kind": "choose_float",
                "low": 0.03,
                "high": 0.12,
                "default": 0.06,
                "log": True,
            },
        ]
    }
    mapped = code_util.map_tuning_best_params(raw, plan)
    assert mapped == {"max_depth": 5, "learning_rate": 0.08276096024435112}


def test_map_tuning_best_params_choose_from_variant_key():
    raw = {"data_op__0": "d8_lr0.05"}
    plan = {
        "tunable_params": [
            {
                "name": "model_variant",
                "kind": "choose_from",
                "outcomes": ["d7_lr0.03", "d8_lr0.05"],
            }
        ]
    }
    assert code_util.map_tuning_best_params(raw, plan) == {
        "model_variant": "d8_lr0.05"
    }
