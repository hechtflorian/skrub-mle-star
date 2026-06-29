"""Tests for code_util tuning param mapping and backbone drift gates."""

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


def test_retriever_estimator_classes():
    assert code_util.retriever_estimator_classes(
        "LightGBM (LGBMClassifier)"
    ) == {"LGBMClassifier"}
    assert code_util.retriever_estimator_classes(
        "CatBoostClassifier"
    ) == {"CatBoostClassifier"}


def test_backbone_drift_rejects_swap():
    required = {"CatBoostClassifier"}
    err = code_util.backbone_drift_violation(
        required,
        "RandomForestClassifier()",
        label="retriever: CatBoostClassifier",
    )
    assert err is not None
    assert "CatBoostClassifier" in err
    assert "RandomForestClassifier" in err


def test_tune_backbone_drift_passes_matching_backbone():
    structural = "LGBMClassifier()\nCatBoostClassifier()"
    tune = "from lightgbm import LGBMClassifier\nLGBMClassifier()\nCatBoostClassifier()"
    assert code_util.backbone_drift_violation(
        {"LGBMClassifier", "CatBoostClassifier"},
        tune,
        label="structural solution",
    ) is None


def test_tune_backbone_drift_rejects_estimator_swap():
    structural = "from lightgbm import LGBMClassifier\nLGBMClassifier()"
    tune = "from sklearn.ensemble import RandomForestClassifier\nRandomForestClassifier()"
    err = code_util.backbone_drift_violation(
        {"LGBMClassifier"},
        tune,
        label="structural solution",
    )
    assert err is not None
    assert "LGBMClassifier" in err
    assert "RandomForestClassifier" in err


def test_tune_search_contract_requires_search_block():
    code = "print('hello')"
    err = code_util.tune_search_contract_violation(code)
    assert err is not None
    assert "choose_*" in err
    assert "make_randomized_search" in err


def test_should_enforce_backbone_drift_init_and_tune_only():
    assert code_util.should_enforce_backbone_drift("model_eval_agent_1_1")
    assert code_util.should_enforce_backbone_drift(
        "model_eval_debug_agent_1_1"
    )
    assert code_util.should_enforce_backbone_drift("tune_implement_agent_1")
    assert not code_util.should_enforce_backbone_drift("plan_implement_agent_1")
    assert not code_util.should_enforce_backbone_drift("merger_agent_1_1")
