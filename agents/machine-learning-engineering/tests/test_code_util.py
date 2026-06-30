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


def test_backbone_drift_accepts_lightgbm_name_alias():
    """Retriever may say LightGBMClassifier; code correctly uses LGBMClassifier."""
    err = code_util.backbone_drift_violation(
        {"LightGBMClassifier"},
        "from lightgbm import LGBMClassifier\nmodel = LGBMClassifier()",
        label="retriever: LightGBMClassifier",
    )
    assert err is None


def test_backbone_drift_accepts_import_alias_both_names_in_code():
    code = (
        "from lightgbm import LGBMClassifier as LightGBMClassifier\n"
        "model = LightGBMClassifier()"
    )
    assert code_util.backbone_drift_violation(
        {"LightGBMClassifier"},
        code,
        label="retriever: LightGBMClassifier",
    ) is None


def test_canonical_estimator_set_merges_aliases():
    assert code_util.canonical_estimator_set(
        {"LightGBMClassifier", "LGBMClassifier"}
    ) == {"LGBMClassifier"}


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


def test_resolve_backbone_required_refinement_from_structural_code():
    class State:
        def __init__(self):
            self.data = {
                "refine_step_1": 0,
                "train_code_0_1": "from catboost import CatBoostClassifier\nCatBoostClassifier()",
            }

        def get(self, key, default=None):
            return self.data.get(key, default)

    class Ctx:
        state = State()

    required, label = code_util.resolve_backbone_required(
        Ctx(), "plan_implement_debug_agent_1", "0_0_1"
    )
    assert required == {"CatBoostClassifier"}
    assert "refine step" in label
    

def test_ablation_contract_requires_print_template():
    assert code_util.ablation_contract_violation("print('hello')") is not None
    assert "Ablation" in code_util.ablation_contract_violation("print('hello')")


def test_ablation_contract_passes_with_print_template():
    code = "print(f'Ablation[{name}] accuracy: {score}')"
    assert code_util.ablation_contract_violation(code) is None


def test_ablation_backbone_requires_overlap_with_input():
    required = {"CatBoostClassifier"}
    err = code_util.ablation_backbone_violation(
        required,
        "from sklearn.ensemble import RandomForestClassifier\n"
        "RandomForestClassifier()",
        label="input solution at refine step 0",
    )
    assert err is not None
    assert "CatBoostClassifier" in err
    assert "no overlap" in err


def test_ablation_backbone_passes_when_input_estimator_present():
    code = (
        "from catboost import CatBoostClassifier\n"
        "CatBoostClassifier()\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "RandomForestClassifier()"
    )
    assert code_util.ablation_backbone_violation(
        {"CatBoostClassifier"},
        code,
        label="input solution at refine step 0",
    ) is None


def test_ablation_backbone_skipped_when_anchor_empty():
    assert code_util.ablation_backbone_violation(set(), "RandomForestClassifier()", label="") is None


def test_resolve_backbone_required_ablation_from_input_solution():
    class State:
        def __init__(self):
            self.data = {
                "refine_step_1": 0,
                "train_code_0_1": "CatBoostClassifier()",
            }

        def get(self, key, default=None):
            return self.data.get(key, default)

    class Ctx:
        state = State()

    required, label = code_util.resolve_backbone_required(
        Ctx(), "ablation_agent_1", "0_1"
    )
    assert required == {"CatBoostClassifier"}
    assert "input solution" in label
