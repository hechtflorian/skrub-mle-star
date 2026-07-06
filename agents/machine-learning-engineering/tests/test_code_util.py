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
    err = code_util.backbone_violation(
        required,
        "RandomForestClassifier()",
        label="retriever: CatBoostClassifier",
        mode="strict",
    )
    assert err is not None
    assert "CatBoostClassifier" in err
    assert "RandomForestClassifier" in err


def test_backbone_drift_accepts_lightgbm_name_alias():
    """Retriever may say LightGBMClassifier; code correctly uses LGBMClassifier."""
    err = code_util.backbone_violation(
        {"LightGBMClassifier"},
        "from lightgbm import LGBMClassifier\nmodel = LGBMClassifier()",
        label="retriever: LightGBMClassifier",
        mode="strict",
    )
    assert err is None


def test_backbone_drift_accepts_import_alias_both_names_in_code():
    code = (
        "from lightgbm import LGBMClassifier as LightGBMClassifier\n"
        "model = LightGBMClassifier()"
    )
    assert code_util.backbone_violation(
        {"LightGBMClassifier"},
        code,
        label="retriever: LightGBMClassifier",
        mode="strict",
    ) is None


def test_canonical_estimator_set_merges_aliases():
    assert code_util.canonical_estimator_set(
        {"LightGBMClassifier", "LGBMClassifier"}
    ) == {"LGBMClassifier"}


def test_backbone_overlap_passes_when_all_estimators_present():
    tune = "from lightgbm import LGBMClassifier\nLGBMClassifier()\nCatBoostClassifier()"
    assert code_util.backbone_violation(
        {"LGBMClassifier", "CatBoostClassifier"},
        tune,
        label="structural solution",
        mode="overlap",
    ) is None


def test_backbone_overlap_passes_when_one_ensemble_leg_kept():
    tune = (
        "from lightgbm import LGBMClassifier\n"
        "LGBMClassifier(learning_rate=skrub.choose_float(0.01, 0.1, name='lr'))"
    )
    assert code_util.backbone_violation(
        {"LGBMClassifier", "CatBoostClassifier"},
        tune,
        label="structural solution",
        mode="overlap",
    ) is None


def test_backbone_overlap_rejects_full_estimator_swap():
    tune = "from sklearn.ensemble import RandomForestClassifier\nRandomForestClassifier()"
    err = code_util.backbone_violation(
        {"LGBMClassifier"},
        tune,
        label="structural solution",
        mode="overlap",
    )
    assert err is not None
    assert "LGBMClassifier" in err
    assert "RandomForestClassifier" in err
    assert "no overlap" in err


def test_map_tuning_best_params_fixes_swapped_human_named_values():
    raw = {"n_estimators": 0, "learning_rate": 1600.0}
    plan = {
        "tunable_params": [
            {"name": "n_estimators", "kind": "choose_int", "low": 800, "high": 1600, "default": 1200},
            {"name": "learning_rate", "kind": "choose_float", "low": 0.01, "high": 0.04, "default": 0.02},
        ]
    }
    mapped = code_util.map_tuning_best_params(raw, plan)
    assert mapped == {"n_estimators": 1600, "learning_rate": 0.02}


def test_tune_search_contract_requires_search_block():
    code = "print('hello')"
    err = code_util.tune_search_contract_violation(code)
    assert err is not None
    assert "choose_*" in err
    assert "make_randomized_search" in err


def test_backbone_check_mode():
    assert code_util.backbone_check_mode("model_eval_agent_1_1") == "strict"
    assert code_util.backbone_check_mode("tune_implement_agent_1") == "overlap"
    assert code_util.backbone_check_mode("ablation_agent_1") == "overlap"
    assert code_util.backbone_check_mode("plan_implement_agent_1") is None


def test_should_snapshot_debug_anchor_init_and_tune_only():
    assert code_util.should_snapshot_debug_anchor("model_eval_agent_1_1")
    assert code_util.should_snapshot_debug_anchor(
        "model_eval_debug_agent_1_1"
    )
    assert code_util.should_snapshot_debug_anchor("tune_implement_agent_1")
    assert not code_util.should_snapshot_debug_anchor("plan_implement_agent_1")
    assert not code_util.should_snapshot_debug_anchor("merger_agent_1_1")
    assert not code_util.should_snapshot_debug_anchor("ablation_agent_1")


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


def test_backbone_overlap_requires_shared_estimator():
    required = {"CatBoostClassifier"}
    err = code_util.backbone_violation(
        required,
        "from sklearn.ensemble import RandomForestClassifier\n"
        "RandomForestClassifier()",
        label="input solution at refine step 0",
        mode="overlap",
    )
    assert err is not None
    assert "CatBoostClassifier" in err
    assert "no overlap" in err


def test_backbone_overlap_passes_when_input_estimator_present():
    code = (
        "from catboost import CatBoostClassifier\n"
        "CatBoostClassifier()\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "RandomForestClassifier()"
    )
    assert code_util.backbone_violation(
        {"CatBoostClassifier"},
        code,
        label="input solution at refine step 0",
        mode="overlap",
    ) is None


def test_backbone_overlap_skipped_when_anchor_empty():
    assert code_util.backbone_violation(
        set(), "RandomForestClassifier()", label="", mode="overlap"
    ) is None


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


def test_submission_export_contract_requires_submission_csv():
    err = code_util.submission_export_contract_violation(
        "test_df = 1\nprint('Final Validation Performance: 0.5')"
    )
    assert err is not None
    assert "submission.csv" in err


def test_submission_export_contract_requires_test_data():
    err = code_util.submission_export_contract_violation(
        "train_df = 1\nsubmission.to_csv('submission.csv')"
    )
    assert err is not None
    assert "test" in err.lower()


def test_submission_export_contract_passes_minimal_valid_script():
    code = """
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
data_full = skrub.var('data', train_df)
test_pred = learner.predict({'data': test_df})
submission.to_csv('./final/submission.csv')
"""
    assert code_util.submission_export_contract_violation(code) is None
