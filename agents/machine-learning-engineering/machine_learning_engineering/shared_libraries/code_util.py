"""Code related utility functions."""

import json
import os
import re
import subprocess
import time
from typing import Any

from google.adk.agents import callback_context as callback_context_module

_ESTIMATOR_CLASS_RE = re.compile(r"\b([A-Z]\w*(?:Regressor|Classifier))\b")


def _estimator_classes(code: str) -> set[str]:
    return set(_ESTIMATOR_CLASS_RE.findall(code))


def retriever_estimator_classes(model_name: str) -> set[str]:
    """Parse estimator class names from model retriever output."""
    if not model_name:
        return set()
    return set(_ESTIMATOR_CLASS_RE.findall(model_name))


def should_enforce_backbone_drift(agent_name: str) -> bool:
    """Stages where debug must not swap the backbone estimator set."""
    return agent_name.startswith("model_eval") or agent_name.startswith(
        "tune_implement"
    )


def debug_anchor_code_key(suffix: str) -> str:
    return f"debug_anchor_code_{suffix}"


def resolve_backbone_required(
    callback_context: callback_context_module.CallbackContext,
    agent_name: str,
    suffix: str,
) -> tuple[set[str], str]:
    """Return (required_estimator_classes, short_label) for drift checks."""
    if agent_name.startswith("model_eval"):
        task_id = agent_name.split("_")[-2]
        model_id = agent_name.split("_")[-1]
        model_info = callback_context.state.get(
            f"init_{task_id}_model_{model_id}",
            {},
        )
        required = retriever_estimator_classes(
            model_info.get("model_name", "")
        )
        if required:
            return required, f"retriever: {model_info.get('model_name', '')}"
        anchor = callback_context.state.get(debug_anchor_code_key(suffix), "")
        if anchor.strip():
            return _estimator_classes(anchor), "first failing init script"
        return set(), ""
    if agent_name.startswith("tune_implement"):
        task_id = agent_name.split("_")[-1]
        outer_loop_round = callback_context.state.get("outer_loop_round", 1)
        structural = callback_context.state.get(
            f"train_code_{outer_loop_round}_{task_id}",
            "",
        )
        if structural.strip():
            return _estimator_classes(structural), "structural solution"
        return set(), ""
    return set(), ""


def backbone_drift_violation(
    required: set[str],
    candidate_code: str,
    *,
    label: str,
) -> str | None:
    """Return an error if candidate code uses a different estimator set."""
    if not required:
        return None
    candidate = _estimator_classes(candidate_code)
    if candidate == required:
        return None
    return (
        "Backbone drift: keep estimator classes "
        f"{sorted(required)} ({label}); got {sorted(candidate)}. "
        "Fix the reported error without swapping model families or "
        "simplifying to a different pipeline."
    )


def tune_search_contract_violation(code: str) -> str | None:
    """Tune scripts must keep in-graph search, not fake tuning or debug shortcuts."""
    missing: list[str] = []
    if "choose_" not in code:
        missing.append("choose_*")
    if "make_randomized_search" not in code:
        missing.append("make_randomized_search")
    if "search.fit" not in code:
        missing.append("search.fit")
    if "TUNING_BEST_PARAMS" not in code:
        missing.append("TUNING_BEST_PARAMS print")
    if not missing:
        return None
    return (
        "Tuning search contract: missing "
        + ", ".join(missing)
        + ". Fix the error without removing the search block or choose_* nodes."
    )


def maybe_set_debug_anchor(
    callback_context: callback_context_module.CallbackContext,
    agent_name: str,
    suffix: str,
    code: str,
) -> None:
    """Snapshot the first failing script for init fallback anchoring."""
    if "debug_agent" in agent_name or not should_enforce_backbone_drift(agent_name):
        return
    key = debug_anchor_code_key(suffix)
    if not callback_context.state.get(key) and code.strip():
        callback_context.state[key] = code


def preexec_code_failure(
    callback_context: callback_context_module.CallbackContext,
    agent_name: str,
    suffix: str,
    raw_code: str,
) -> dict[str, Any] | None:
    """Compile + backbone/search gates before subprocess (cheap, explicit stderr)."""
    if not raw_code.strip():
        return None
    try:
        compile(raw_code, f"<{agent_name}>", "exec")
    except SyntaxError as exc:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Syntax error: {exc}",
            "execution_time": 0.0,
        }
    if should_enforce_backbone_drift(agent_name):
        required, label = resolve_backbone_required(
            callback_context, agent_name, suffix
        )
        drift = backbone_drift_violation(required, raw_code, label=label)
        if drift:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": drift,
                "execution_time": 0.0,
            }
    if agent_name.startswith("tune_implement"):
        contract = tune_search_contract_violation(raw_code)
        if contract:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": contract,
                "execution_time": 0.0,
            }
    return None


class Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_python_code(
    code_text: str,
    run_cwd: str,
    py_filepath: str,
    exec_timeout: int,
) -> dict[str, Any]:
    start_time = time.time()
    output_filepath = os.path.join(run_cwd, py_filepath)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(code_text)
    try:
        result = subprocess.run(
            ["python", py_filepath],
            check=False,
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=exec_timeout,
        )
    except Exception as e:
        result = Result(returncode=1, stdout="", stderr=str(e))
    end_time = time.time()
    execution_time = end_time - start_time
    result_dict = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time": execution_time,
    }
    return result_dict


def truncate_for_state(text: str, head: int = 2000, tail: int = 4000) -> str:
    """Bound subprocess output before storing it in state.

    Keeps head + tail (metric lines and tracebacks are at the end); the
    middle (e.g. unbounded CatBoost/LightGBM training logs) is elided.
    Callers must extract scores/params from the full text before truncating.
    """
    if len(text) <= head + tail + 200:
        return text
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n... [{omitted} characters of output omitted] ...\n"
        + text[-tail:]
    )


def normalize_tuning_best_params(params: dict) -> dict:
    """Convert numpy scalars to native Python types for JSON-safe state storage."""
    normalized: dict = {}
    for key, value in params.items():
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass
        normalized[key] = value
    return normalized


def extract_tuning_best_params(text: str) -> dict | None:
    """Parse TUNING_BEST_PARAMS JSON line from tune_implement stdout."""
    for line in text.splitlines():
        if line.startswith("TUNING_BEST_PARAMS:"):
            json_part = line.split(":", 1)[1].strip()
            try:
                parsed = json.loads(json_part)
                if isinstance(parsed, dict):
                    return normalize_tuning_best_params(parsed)
            except json.JSONDecodeError:
                return None
    return None


def _data_op_sort_key(key: str) -> tuple[int, str]:
    if key.startswith("data_op__"):
        suffix = key.split("__", 1)[-1]
        if suffix.isdigit():
            return (int(suffix), key)
    return (10**9, key)


def _normalize_param_value(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    return value


def _param_match_cost(value: Any, spec: dict) -> float:
    """Return match cost; lower is better, inf means no match."""
    value = _normalize_param_value(value)
    kind = spec.get("kind", "")

    if kind == "choose_int":
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            return 1.0
        try:
            iv = int(round(float(value)))
        except (TypeError, ValueError):
            return float("inf")
        if low <= iv <= high:
            return 0.0
        if iv < low:
            return float(low - iv)
        return float(iv - high)

    if kind == "choose_float":
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            return 1.0
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return float("inf")
        if low <= fv <= high:
            return 0.0
        denom = max(abs(low), abs(high), 1e-9)
        if fv < low:
            return (low - fv) / denom
        return (fv - high) / denom

    if kind == "choose_from":
        outcomes = spec.get("outcomes") or spec.get("choices") or []
        if isinstance(outcomes, dict):
            keys = list(outcomes.keys())
        elif isinstance(outcomes, list):
            keys = outcomes
        else:
            keys = []
        if value in keys or str(value) in keys:
            return 0.0
        return 1.0

    return 1.0


def _map_params_by_value_match(values: list[Any], specs: list[dict]) -> dict | None:
    """Greedy min-cost assignment of search values to plan param names."""
    if len(values) != len(specs):
        return None
    pairs: list[tuple[float, int, int]] = []
    for value_idx, value in enumerate(values):
        for spec_idx, spec in enumerate(specs):
            cost = _param_match_cost(value, spec)
            pairs.append((cost, value_idx, spec_idx))
    pairs.sort()

    used_values: set[int] = set()
    used_specs: set[int] = set()
    mapped: dict = {}
    for cost, value_idx, spec_idx in pairs:
        if cost == float("inf"):
            continue
        if value_idx in used_values or spec_idx in used_specs:
            continue
        name = specs[spec_idx].get("name")
        if not name:
            continue
        mapped[name] = values[value_idx]
        used_values.add(value_idx)
        used_specs.add(spec_idx)

    if len(mapped) == len(specs):
        return mapped
    return None


def map_tuning_best_params(raw: dict, tune_plan: dict) -> dict:
    """Map skrub search keys (e.g. data_op__0) to plan param names."""
    if not raw:
        return raw
    tunable = tune_plan.get("tunable_params") or []
    specs = [p for p in tunable if isinstance(p, dict) and p.get("name")]
    names = [p["name"] for p in specs]
    if not names:
        return raw
    raw = normalize_tuning_best_params(raw)
    if set(raw.keys()) == set(names):
        return raw

    ordered_values = [
        _normalize_param_value(value)
        for _, value in sorted(raw.items(), key=lambda item: _data_op_sort_key(item[0]))
    ]
    if len(names) != len(ordered_values):
        return raw

    if any(key.startswith("data_op__") for key in raw.keys()) or set(raw.keys()) != set(
        names
    ):
        mapped = _map_params_by_value_match(ordered_values, specs)
        if mapped:
            return normalize_tuning_best_params(mapped)

    return normalize_tuning_best_params(
        {name: ordered_values[i] for i, name in enumerate(names)}
    )


def code_contains_tuning_placeholders(raw_code: str) -> bool:
    """Return True if code still has choose_* or search calls."""
    return (
        "choose_" in raw_code
        or "make_randomized_search" in raw_code
        or "make_grid_search" in raw_code
    )


def extract_performance_from_text(text: str) -> float | None:
    """Extracts the final validation performance score from the text."""
    lines = text.splitlines()
    performance_value = None
    for line in lines:
        if "Final Validation Performance" in line:
            try:
                parts = line.split(":")
                # score_str = line.split("Final Validation Performance:")[-1].strip()
                score_str = parts[-1].strip()
                performance_value = float(score_str)
            except ValueError:
                pass
    return performance_value


def get_name_with_prefix_and_suffix(
    base_name: str,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Gets the name with the specified prefix and suffix."""
    new_name = base_name
    if prefix:
        new_name = prefix + "_" + new_name
    if suffix:
        new_name = new_name + "_" + suffix
    return new_name


def get_updated_suffix(
    callback_context: callback_context_module.CallbackContext,
) -> str:
    """Gets the suffix string."""
    agent_name = callback_context.agent_name
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        suffix = f"{task_id}_{model_id}"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        suffix = f"{task_id}_{reference_idx}"
    elif agent_name.startswith("check_data_use"):
        task_id = agent_name.split("_")[-1]
        suffix = f"{task_id}"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        suffix = f"{step}_{task_id}"
    elif agent_name.startswith("plan_implement"):
        task_id = callback_context.agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
        suffix = f"{inner_iter}_{step}_{task_id}"
    elif agent_name.startswith("tune_implement") or agent_name.startswith("tune_bake"):
        task_id = agent_name.split("_")[-1]
        suffix = f"{task_id}"
    elif agent_name.startswith("ensemble_plan_implement"):
        ensemble_iter = callback_context.state.get("ensemble_iter", 0)
        suffix = f"{ensemble_iter}"
    elif agent_name.startswith("submission"):
        suffix = ""
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return suffix


def get_code_state_key(
    agent_name: str,
    suffix: str,
) -> str:
    """Gets the state key for the code."""
    if agent_name.startswith("model_eval"):
        key = f"init_code_{suffix}"
    elif agent_name.startswith("merger"):
        key = f"merger_code_{suffix}"
    elif agent_name.startswith("check_data_use"):
        key = f"train_code_0_{suffix}"
    elif agent_name.startswith("ablation"):
        key = f"ablation_code_{suffix}"
    elif agent_name.startswith("plan_implement"):
        key = f"train_code_improve_{suffix}"
    elif agent_name.startswith("tune_implement"):
        key = f"train_code_tune_search_{suffix}"
    elif agent_name.startswith("tune_bake"):
        key = f"train_code_tune_{suffix}"
    elif agent_name.startswith("ensemble_plan_implement"):
        key = f"ensemble_code_{suffix}"
    elif agent_name.startswith("submission"):
        key = "submission_code"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return key


def get_code_execution_result_state_key(
    agent_name: str,
    suffix: str,
) -> str:
    """Gets the state key for the code execution result."""
    if agent_name.startswith("model_eval"):
        key = f"init_code_exec_result_{suffix}"
    elif agent_name.startswith("merger"):
        key = f"merger_code_exec_result_{suffix}"
    elif agent_name.startswith("check_data_use"):
        key = f"train_code_exec_result_0_{suffix}"
    elif agent_name.startswith("ablation"):
        key = f"ablation_code_exec_result_{suffix}"
    elif agent_name.startswith("plan_implement"):
        key = f"train_code_improve_exec_result_{suffix}"
    elif agent_name.startswith("tune_implement"):
        key = f"train_code_tune_search_exec_result_{suffix}"
    elif agent_name.startswith("tune_bake"):
        key = f"train_code_tune_exec_result_{suffix}"
    elif agent_name.startswith("ensemble_plan_implement"):
        key = f"ensemble_code_exec_result_{suffix}"
    elif agent_name.startswith("submission"):
        key = "submission_code_exec_result"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return key


def get_run_code_condition(
    agent_name: str,
    raw_code: str,
) -> bool:
    """Gets the condition for running the code. Will ignore (empty) tool-call results."""
    if agent_name.startswith("ensemble_plan_implement"):
        # Tool calls may be returned before code; only run real Python code.
        if not raw_code.strip():
            return False
        try:
            compile(raw_code, "<ensemble_plan_implement>", "exec")
        except SyntaxError:
            return False
        if "debug_agent" not in agent_name:
            return True
        if (
            "Final Validation Performance" in raw_code
            and "exit()" not in raw_code
        ):
            #if code_contains_tuning_placeholders(raw_code):
                #return False
            return True
    elif agent_name.startswith("ablation"):
        # With tool-enabled ablation agents, responses can contain tool/prose output which will be empty.
        # Run only when we have non-empty, syntactically valid Python code.
        if not raw_code.strip():
            return False
        try:
            compile(raw_code, "<ablation>", "exec")
        except SyntaxError:
            return False
        if "debug_agent" not in agent_name:
            return True
        if "exit()" not in raw_code:
            return True
    elif agent_name.startswith("plan_implement"):
        # Tool calls may be returned before code; only run real Python code.
        if not raw_code.strip():
            return False
        try:
            compile(raw_code, "<plan_implement>", "exec")
        except SyntaxError:
            return False
        #if code_contains_tuning_placeholders(raw_code):
            #return False
        if "debug_agent" not in agent_name:
            return True
        if "exit()" not in raw_code:
            return True
    elif agent_name.startswith("tune_implement"):
        if not raw_code.strip():
            return False
        try:
            compile(raw_code, "<tune_implement>", "exec")
        except SyntaxError:
            return False
        if "TUNING_BEST_PARAMS" not in raw_code:
            return False
        if "choose_" not in raw_code:
            return False
        if "make_randomized_search" not in raw_code:
            return False
        if "search.fit" not in raw_code:
            return False
        if "debug_agent" not in agent_name:
            return True
        if "exit()" not in raw_code:
            return True
    elif agent_name.startswith("tune_bake"):
        if not raw_code.strip():
            return False
        try:
            compile(raw_code, "<tune_bake>", "exec")
        except SyntaxError:
            return False
        if code_contains_tuning_placeholders(raw_code):
            return False
        if "debug_agent" not in agent_name:
            return True
        if (
            "Final Validation Performance" in raw_code
            and "exit()" not in raw_code
        ):
            return True
    elif agent_name.startswith("submission"):
        if (
            "debug_agent" not in agent_name
            and "exit()" not in raw_code
            and "submission.csv" in raw_code
        ):
            #if code_contains_tuning_placeholders(raw_code):
                #return False
            return True
        if "debug_agent" in agent_name and "exit()" not in raw_code:
            #if code_contains_tuning_placeholders(raw_code):
                #return False
            return True
    elif (
        "Final Validation Performance" in raw_code and "exit()" not in raw_code
    ):
        return True
    return False


def evaluate_code(
    callback_context: callback_context_module.CallbackContext,
) -> None:
    """Evaluates the given code."""
    lower = callback_context.state.get("lower", True)
    exec_timeout = callback_context.state.get("exec_timeout", 1800)
    agent_name = callback_context.agent_name
    suffix = get_updated_suffix(callback_context=callback_context)
    code_state_key = get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    raw_code = callback_context.state.get(code_state_key, "")
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        py_filepath = f"init_code_{model_id}.py"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        py_filepath = f"train0_{reference_idx}.py"
    elif agent_name.startswith("check_data_use"):
        task_id = agent_name.split("_")[-1]
        py_filepath = "train0.py"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        py_filepath = f"ablation_{step}.py"
    elif agent_name.startswith("plan_implement"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
        py_filepath = f"train{step}_improve{inner_iter}.py"
    elif agent_name.startswith("tune_implement"):
        task_id = agent_name.split("_")[-1]
        py_filepath = "train_tune_search.py"
    elif agent_name.startswith("tune_bake"):
        task_id = agent_name.split("_")[-1]
        py_filepath = "train_tune_baked.py"
    elif agent_name.startswith("ensemble_plan_implement"):
        task_id = "ensemble"
        py_filepath = f"ensemble{suffix}.py"
    elif agent_name.startswith("submission"):
        task_id = "ensemble"
        py_filepath = "final_solution.py"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    if not raw_code.strip():
        result_dict = {}
    else:
        preexec = preexec_code_failure(
            callback_context, agent_name, suffix, raw_code
        )
        if preexec is not None:
            result_dict = preexec
        elif get_run_code_condition(
            agent_name=agent_name,
            raw_code=raw_code,
        ):
            workspace_dir = callback_context.state.get("workspace_dir", "")
            task_name = callback_context.state.get("task_name", "")
            run_cwd = os.path.join(workspace_dir, task_name, task_id)
            result_dict = run_python_code(
                code_text=raw_code,
                run_cwd=run_cwd,
                py_filepath=py_filepath,
                exec_timeout=exec_timeout,
            )
            if result_dict.get("returncode", 1) != 0:
                maybe_set_debug_anchor(
                    callback_context, agent_name, suffix, raw_code
                )
            if agent_name.startswith("ablation"):
                if result_dict["returncode"] == 0:
                    ablation_result = result_dict.get("stdout", "None")
                    if ablation_result.count("Ablation[") < 2:
                        result_dict["returncode"] = 1
                        result_dict["stderr"] = (
                            result_dict.get("stderr", "")
                            + "\nAblation contract violation: stdout must contain "
                            "at least 2 lines in the format "
                            "'Ablation[<variant_name>] <metric>: <value>' "
                            "(baseline + at least one ablated variant). Keep the "
                            "existing variants and backbone; only fix the error."
                        )
                        ablation_result = "None"
                else:
                    ablation_result = "None"
                result_dict["ablation_result"] = ablation_result
            else:
                if result_dict.get("returncode", 1) == 0:
                    try:
                        score = extract_performance_from_text(
                            result_dict.get("stdout", "")
                        )
                        score = float(score)
                    except Exception:
                        score = 1e9 if lower else 0
                    if agent_name.startswith("tune_implement"):
                        task_id = agent_name.split("_")[-1]
                        best_params = extract_tuning_best_params(
                            result_dict.get("stdout", "")
                        )
                        if best_params:
                            tune_plan = callback_context.state.get(
                                f"tune_plan_{task_id}", {}
                            )
                            callback_context.state[
                                f"tune_best_params_{task_id}"
                            ] = map_tuning_best_params(best_params, tune_plan)
                        else:
                            result_dict["returncode"] = 1
                            result_dict["stderr"] = (
                                result_dict.get("stderr", "")
                                + "\nTuning contract violation: script exited 0 "
                                "but no non-empty TUNING_BEST_PARAMS JSON line "
                                "was found in stdout. Print exactly: "
                                'print("TUNING_BEST_PARAMS:", '
                                "json.dumps(best_params, default=str)) "
                                "with the best parameters from the executed "
                                "search."
                            )
                else:
                    score = 1e9 if lower else 0
                result_dict["score"] = score
            for key in ("stdout", "stderr", "ablation_result"):
                if isinstance(result_dict.get(key), str):
                    result_dict[key] = truncate_for_state(result_dict[key])
        else:
            result_dict = {}
    code_execution_result_state_key = get_code_execution_result_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    callback_context.state[code_execution_result_state_key] = result_dict
