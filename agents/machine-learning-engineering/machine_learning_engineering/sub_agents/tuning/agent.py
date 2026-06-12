"""Terminal tuning agent for Machine Learning Engineering."""

import json
import os

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_request as llm_request_module
from google.adk.models import llm_response as llm_response_module
from google.genai import types

from machine_learning_engineering.shared_libraries import (
    code_util,
    common_util,
    config,
    debug_util,
    skill_tool_util,
)
from machine_learning_engineering.sub_agents.tuning import prompt


def _outer_loop_round(state) -> int:
    return state.get("outer_loop_round", config.CONFIG.outer_loop_round)


def _last_refine_step(state, outer_loop_round: int) -> int:
    return outer_loop_round - 1


def _build_plan_summary(
    context: callback_context_module.ReadonlyContext,
    task_id: str,
    step: int,
) -> str:
    """Summarize inner-loop plan outcomes from the last refinement outer step."""
    lower = context.state.get("lower", True)
    prev_exec_result = context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    prev_plans = context.state.get(f"refine_plans_{step}_{task_id}", [])
    summary = ""
    for inner_iter, curr_plan in enumerate(prev_plans):
        exec_result = context.state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}",
            {},
        )
        if "score" not in prev_exec_result or "score" not in exec_result:
            continue
        if lower:
            improvement = prev_exec_result["score"] - exec_result["score"]
        else:
            improvement = exec_result["score"] - prev_exec_result["score"]
        summary += f"## Plan: {curr_plan}\n"
        summary += f"## Score delta: {improvement:.5f}\n\n"
    if not summary:
        summary = "No structural plan improvements recorded."
    return summary


def promote_tuning_winner(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Promote tuned code into canonical train_code keys if it beats structural."""
    task_id = callback_context.agent_name.split("_")[-1]
    outer_loop_round = _outer_loop_round(callback_context.state)
    lower = callback_context.state.get("lower", True)
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name, task_id)

    structural_result = callback_context.state.get(
        f"train_code_exec_result_{outer_loop_round}_{task_id}", {}
    )
    structural_score = structural_result.get("score")
    tuned_result = callback_context.state.get(
        f"train_code_tune_exec_result_{task_id}", {}
    )
    tuned_code = callback_context.state.get(f"train_code_tune_{task_id}", "")
    winner_source = "structural"

    if (
        structural_score is not None
        and tuned_result.get("returncode", 1) == 0
        and "score" in tuned_result
        and tuned_code
        and not code_util.code_contains_tuning_placeholders(tuned_code)
    ):
        tuned_score = tuned_result["score"]
        if lower:
            tuned_improves = structural_score - tuned_score > 0
        else:
            tuned_improves = tuned_score - structural_score > 0
        if tuned_improves:
            callback_context.state[
                f"train_code_{outer_loop_round}_{task_id}"
            ] = tuned_code
            callback_context.state[
                f"train_code_exec_result_{outer_loop_round}_{task_id}"
            ] = tuned_result
            output_filepath = os.path.join(
                run_cwd, f"train{outer_loop_round}_tuned.py"
            )
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(tuned_code)
            winner_source = "tuned"

    callback_context.state[f"tune_winner_source_{task_id}"] = winner_source
    return None


def get_tune_plan_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets tune plan agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    outer_loop_round = _outer_loop_round(context.state)
    step = _last_refine_step(context.state, outer_loop_round)
    code = context.state.get(f"train_code_{outer_loop_round}_{task_id}", "")
    ablation_results = context.state.get(
        f"ablation_summary_{step}_{task_id}", ""
    )
    plan_summary = _build_plan_summary(context, task_id, step)
    return prompt.TUNE_PLAN_INSTR.format(
        code=code,
        ablation_results=ablation_results,
        plan_summary=plan_summary,
        n_iter=config.CONFIG.tuning_n_iter,
    )


def get_tune_implement_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets tune implement agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    outer_loop_round = _outer_loop_round(context.state)
    code = context.state.get(f"train_code_{outer_loop_round}_{task_id}", "")
    tune_plan = context.state.get(f"tune_plan_{task_id}", {})
    return prompt.TUNE_IMPLEMENT_INSTR.format(
        code=code,
        tune_plan=json.dumps(tune_plan, indent=2),
        n_iter=config.CONFIG.tuning_n_iter,
        n_jobs=config.CONFIG.tuning_n_jobs,
    )


def get_tune_bake_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets tune bake agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    outer_loop_round = _outer_loop_round(context.state)
    code = context.state.get(f"train_code_{outer_loop_round}_{task_id}", "")
    tune_plan = context.state.get(f"tune_plan_{task_id}", {})
    best_params = context.state.get(f"tune_best_params_{task_id}", {})
    return prompt.TUNE_BAKE_INSTR.format(
        code=code,
        tune_plan=json.dumps(tune_plan, indent=2),
        best_params_json=json.dumps(best_params, indent=2),
    )


def get_tune_plan(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Parse tune plan JSON from agent response."""
    response_text = common_util.get_text_from_response(llm_response)
    task_id = callback_context.agent_name.split("_")[-1]
    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}") + 1
    try:
        plan = json.loads(response_text[start_idx:end_idx])
    except (json.JSONDecodeError, ValueError):
        plan = {}
    callback_context.state[f"tune_plan_{task_id}"] = plan
    return None


def check_tune_plan_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if tune plan is finished."""
    task_id = callback_context.agent_name.split("_")[-1]
    plan = callback_context.state.get(f"tune_plan_{task_id}", {})
    if plan and plan.get("focus_block"):
        return llm_response_module.LlmResponse()
    return None


def check_tune_implement_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if tune implement is finished."""
    task_id = callback_context.agent_name.split("_")[-1]
    result_dict = callback_context.state.get(
        f"train_code_tune_search_exec_result_{task_id}", {}
    )
    if not result_dict:
        return None

    best_params = callback_context.state.get(f"tune_best_params_{task_id}", {})
    compliant = (
        result_dict.get("returncode", 1) == 0
        and "score" in result_dict
        and bool(best_params)
    )
    callback_context.state[
        f"tune_implement_skip_data_leakage_check_{task_id}"
    ] = compliant
    # Fail-fast: one implement attempt per rollback round, then debug.
    return llm_response_module.LlmResponse()


def check_tune_bake_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if tune bake finished with fixed-parameter code."""
    task_id = callback_context.agent_name.split("_")[-1]
    result_dict = callback_context.state.get(
        f"train_code_tune_exec_result_{task_id}", {}
    )
    baked_code = callback_context.state.get(f"train_code_tune_{task_id}", "")
    best_params = callback_context.state.get(f"tune_best_params_{task_id}", {})
    callback_context.state[f"tune_bake_skip_data_leakage_check_{task_id}"] = True
    if (
        result_dict.get("returncode", 1) == 0
        #and "score" in result_dict
        #and baked_code
        #and best_params
        and not code_util.code_contains_tuning_placeholders(baked_code)
    ):
        return llm_response_module.LlmResponse()
    callback_context.state[f"tune_bake_skip_data_leakage_check_{task_id}"] = False
    return None


tuning_parallel_sub_agents = []
for k in range(config.CONFIG.num_solutions):
    tune_plan_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"tune_plan_agent_{k + 1}",
        description="Plan choose_* tuning on one focus block.",
        instruction=get_tune_plan_agent_instruction,
        tools=[skill_tool_util.get_skill_toolset()],
        before_model_callback=check_tune_plan_finish,
        after_model_callback=get_tune_plan,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.get_compatible_temperature(
                config.CONFIG.agent_model, 1.0
            ),
        ),
        include_contents="none",
    )
    tune_plan_loop_agent = agents.LoopAgent(
        name=f"tune_plan_loop_agent_{k + 1}",
        description="Generate choose_* tuning plan until valid.",
        sub_agents=[tune_plan_agent],
        max_iterations=config.CONFIG.max_retry,
    )
    tune_implement_agent = debug_util.get_run_and_debug_agent(
        prefix="tune_implement",
        suffix=f"{k + 1}",
        agent_description="Run bounded choose_* search on holdout",
        instruction_func=get_tune_implement_agent_instruction,
        before_model_callback=check_tune_implement_finish,
        tools=[skill_tool_util.get_skill_toolset()],
    )
    tune_bake_agent = debug_util.get_run_and_debug_agent(
        prefix="tune_bake",
        suffix=f"{k + 1}",
        agent_description="Bake tuned params into fixed-parameter code",
        instruction_func=get_tune_bake_agent_instruction,
        before_model_callback=check_tune_bake_finish,
        tools=[skill_tool_util.get_skill_toolset()],
    )
    tune_task_agent = agents.SequentialAgent(
        name=f"tune_task_agent_{k + 1}",
        description="Plan, search, bake, and promote tuning.",
        sub_agents=[
            tune_plan_loop_agent,
            tune_implement_agent,
            tune_bake_agent,
        ],
        after_agent_callback=promote_tuning_winner,
    )
    tuning_parallel_sub_agents.append(tune_task_agent)

tuning_agent = agents.ParallelAgent(
    name="tuning_agent",
    description="Run choose_* tuning once after refinement.",
    sub_agents=tuning_parallel_sub_agents,
)
