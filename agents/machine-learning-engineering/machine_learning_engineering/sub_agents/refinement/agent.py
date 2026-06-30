"""Refinement agent for Machine Learning Engineering."""

import functools
import json
import os

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_request as llm_request_module
from google.adk.models import llm_response as llm_response_module
from google.genai import types

from machine_learning_engineering.shared_libraries import (
    check_leakage_util,
    common_util,
    config,
    debug_util,
    skill_tool_util,
    table_report_util,
)
from machine_learning_engineering.sub_agents.refinement import prompt


def update_inner_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Updates inner loop states."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"inner_iter_{task_id}"] += 1
    return None


def update_outer_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Updates outer loop states."""
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    lower = callback_context.state.get("lower", True)
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    prev_solution = callback_context.state.get(
        f"train_code_{step}_{task_id}", ""
    )
    prev_exec_result = callback_context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    inner_loop_round = callback_context.state.get("inner_loop_round", config.CONFIG.inner_loop_round)
    improvements: list[float] = []
    improvement_indices: list[int] = []
    # init_plan_implement -> improve_0; each refine_inner_loop iter -> improve_1..N - ensure 1..N also read for promotion logic
    for inner_iter in range(1 + inner_loop_round):
        exec_result = callback_context.state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}",
            {},
        )
        if "score" not in prev_exec_result or "score" not in exec_result:
            continue
        if lower:
            improvement = prev_exec_result["score"] - exec_result["score"]
        else:
            improvement = exec_result["score"] - prev_exec_result["score"]
        improvements.append(improvement)
        improvement_indices.append(inner_iter)
    best_solution = prev_solution
    best_exec_result = prev_exec_result
    if improvements:
        best_improvement = max(improvements)
        if best_improvement > 0.0:
            best_pos = improvements.index(best_improvement)
            best_idx = improvement_indices[best_pos]
            best_solution = callback_context.state.get(
                f"train_code_improve_{best_idx}_{step}_{task_id}", ""
            )
            best_exec_result = callback_context.state.get(
                f"train_code_improve_exec_result_{best_idx}_{step}_{task_id}",
                {},
            )
    output_filepath = os.path.join(run_cwd, f"train{step + 1}.py")
    callback_context.state[f"train_code_{step + 1}_{task_id}"] = best_solution
    callback_context.state[
        f"train_code_exec_result_{step + 1}_{task_id}"
    ] = best_exec_result
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(best_solution)
    ablation_results = callback_context.state.get(
        f"ablation_summary_{step}_{task_id}", ""
    )
    code_block = callback_context.state.get(
        f"refine_code_block_{step}_{task_id}", ""
    )
    callback_context.state[f"prev_ablations_{task_id}"].append(ablation_results)
    callback_context.state[f"prev_code_blocks_{task_id}"].append(code_block)
    callback_context.state[f"refine_step_{task_id}"] += 1
    return None


def init_inner_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Initializes inner loop states."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"inner_iter_{task_id}"] = 0
    return None


def init_outer_loop_states(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Initializes outer loop states."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"refine_step_{task_id}"] = 0
    callback_context.state[f"prev_ablations_{task_id}"] = []
    callback_context.state[f"prev_code_blocks_{task_id}"] = []
    profile_key = table_report_util.profile_state_key(task_id)
    if not config.CONFIG.table_report_enabled:
        callback_context.state[profile_key] = ""
        return None
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name, task_id)
    train_path = os.path.join(run_cwd, "input", "train.csv")
    task_workspace = os.path.join(workspace_dir, task_name)
    if not os.path.exists(train_path):
        callback_context.state[profile_key] = ""
        return None
    try:
        report_dict = table_report_util.load_table_report_dict(train_path)
        target_col = table_report_util.extract_target_from_code(
            callback_context.state.get(f"train_code_0_{task_id}", "")
        )
        callback_context.state[profile_key] = (
            table_report_util.format_ablation_profile(
                report_dict,
                target_col=target_col,
            )
        )
        os.makedirs(task_workspace, exist_ok=True)
        report_path = os.path.join(task_workspace, "table_report.json")
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report_dict, report_file, indent=2)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        callback_context.state[profile_key] = ""
    return None


def get_ablation_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ablation agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    prev_ablations = context.state.get(f"prev_ablations_{task_id}", [])
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    prev_ablations_str = ""
    for i, ablation_result in enumerate(prev_ablations):
        prev_ablations_str += f"## Previous ablation study result {i + 1}\n"
        prev_ablations_str += f"{ablation_result}\n\n"
    data_profile = table_report_util.get_profile_from_state(
        context.state, task_id
    )
    if prev_ablations_str:
        instruction = prompt.ABLATION_SEQ_INSTR.format(
            code=code,
            prev_ablations=prev_ablations_str,
            data_profile=data_profile,
        )
    else:
        instruction = prompt.ABLATION_INSTR.format(
            code=code,
            data_profile=data_profile,
        )
    return instruction


def get_ablation_summary_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ablation summary agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"ablation_code_{step}_{task_id}", "")
    result_dict = context.state.get(
        f"ablation_code_exec_result_{step}_{task_id}", {}
    )
    return prompt.SUMMARIZE_ABLATION_INSTR.format(
        code=code,
        result=result_dict["ablation_result"],
    )


def get_init_plan_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the initial plan agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code = context.state.get(f"train_code_{step}_{task_id}", "")
    ablation_results = context.state.get(
        f"ablation_summary_{step}_{task_id}", ""
    )
    prev_code_blocks = context.state.get(f"prev_code_blocks_{task_id}", [])
    data_profile = table_report_util.get_profile_from_state(
        context.state, task_id
    )
    if not prev_code_blocks:
        instruction = prompt.EXTRACT_BLOCK_AND_PLAN_INSTR.format(
            code=code,
            ablation_results=ablation_results,
            data_profile=data_profile,
        )
    else:
        instruction = prompt.EXTRACT_BLOCK_AND_PLAN_SEQ_INSTR.format(
            code=code,
            ablation_results=ablation_results,
            prev_code_blocks=prev_code_blocks,
            data_profile=data_profile,
        )
    return instruction


def get_plan_refinement_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets plan refinement instruction."""
    lower = context.state.get("lower", True)
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code_block = context.state.get(f"refine_code_block_{step}_{task_id}", "")
    prev_plans = context.state.get(f"refine_plans_{step}_{task_id}", [])
    prev_exec_result = context.state.get(
        f"train_code_exec_result_{step}_{task_id}", {}
    )
    score_plan_time_list = []
    for inner_iter, curr_plan in enumerate(prev_plans):
        exec_result = context.state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}", {}
        )
        # new check: if no improvements, don't advance loop and keep previous solution
        if "score" not in prev_exec_result or "score" not in exec_result:
            continue
        execution_time = exec_result.get("execution_time", 0.0)
        if lower:
            improvement = prev_exec_result["score"] - exec_result["score"]
        else:
            improvement = exec_result["score"] - prev_exec_result["score"]
        score_plan_time_list.append(
            (improvement, curr_plan, execution_time)
        )
    num_top_plans = context.state.get("num_top_plans", 3)
    score_plan_time_list.sort(key=lambda x: x[0], reverse=True)
    prev_plan_summary = ""
    selected_score_plan_time_list = score_plan_time_list[:num_top_plans]
    for score, curr_plan, execution_time in selected_score_plan_time_list:
        prev_plan_summary += f"## Plan: {curr_plan}\n"
        prev_plan_summary += (
            f"## Execution time after implement: {execution_time}s\n"
        )
        prev_plan_summary += f"## Score: {score:.5f}\n\n"
    data_profile = table_report_util.get_profile_from_state(
        context.state, task_id
    )
    return prompt.PLAN_REFINEMENT_INSTR.format(
        code_block=code_block,
        prev_plan_summary=prev_plan_summary,
        data_profile=data_profile,
    )


def get_plan_implement_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the plan implement agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    step = context.state.get(f"refine_step_{task_id}", 0)
    code_block = context.state.get(f"refine_code_block_{step}_{task_id}", "")
    plan = context.state.get(f"refine_plans_{step}_{task_id}", [""])[-1]
    return prompt.IMPLEMENT_PLAN_INSTR.format(
        code_block=code_block,
        plan=plan,
    )


def check_ablation_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the ablation study is finished."""
    task_id = callback_context.agent_name.split("_")[-1]
    callback_context.state[f"ablation_skip_data_leakage_check_{task_id}"] = True
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    result_dict = callback_context.state.get(
        f"ablation_code_exec_result_{step}_{task_id}", {}
    )
    if result_dict.get("returncode", 1) == 0:
        return llm_response_module.LlmResponse()
    callback_context.state[f"ablation_skip_data_leakage_check_{task_id}"] = (
        False
    )
    return None


def check_init_plan_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the initial plan is finished."""
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    code = callback_context.state.get(f"train_code_{step}_{task_id}", "")
    code_block = callback_context.state.get(
        f"refine_code_block_{step}_{task_id}", ""
    )
    status = code and code_block and (code_block in code)
    if status:
        return llm_response_module.LlmResponse()
    return None


def check_plan_implement_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if the plan implement is finished."""
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
    suffix = f"{inner_iter}_{step}_{task_id}"
    code_block = callback_context.state.get(
        f"refine_code_block_{step}_{task_id}", ""
    )
    if not code_block.strip():
        # No valid extracted block from init_plan: the block merge cannot
        # work, so skip all implement LLM calls; outer loop keeps the
        # previous solution.
        callback_context.state[
            f"plan_implement_skip_data_leakage_check_{suffix}"
        ] = True
        return llm_response_module.LlmResponse()
    result_dict = callback_context.state.get(
        f"train_code_improve_exec_result_{suffix}", {}
    )
    improved_code = callback_context.state.get(
        f"train_code_improve_{suffix}", ""
    )
    prev_code = callback_context.state.get(
        f"train_code_{step}_{task_id}", ""
    )
    callback_context.state[
        f"plan_implement_skip_data_leakage_check_{suffix}"
    ] = True
    # new tool-call only check: finish only when exec succeeds, code non-empty, and differs from prev code
    if (
        result_dict.get("returncode", 1) == 0
        and "score" in result_dict
        and improved_code.strip()
        and improved_code != prev_code
    ):
        return llm_response_module.LlmResponse()
    callback_context.state[
        f"plan_implement_skip_data_leakage_check_{suffix}"
    ] = False
    return None


def get_ablation_summary(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Gets the ablation summary from the response."""
    response_text = common_util.get_text_from_response(llm_response)
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    callback_context.state[f"ablation_summary_{step}_{task_id}"] = response_text
    return None


def get_plan_and_code_block(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Gets the plan and code block from the response."""
    response_text = common_util.get_text_from_response(llm_response)
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    start_idx = response_text.find("[")
    end_idx = response_text.rfind("]") + 1
    try:
        result = json.loads(response_text[start_idx:end_idx])[0]
        plan = result["plan"]
        code_block = (
            result["code_block"].replace("```python", "").replace("```", "")
        )
    except Exception:
        plan = ""
        code_block = ""
    callback_context.state[f"refine_plans_{step}_{task_id}"] = [plan]
    callback_context.state[f"refine_code_block_{step}_{task_id}"] = code_block
    return None


def get_refined_plan(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> llm_response_module.LlmResponse | None:
    """Gets the refined plan from the response."""
    response_text = common_util.get_text_from_response(llm_response)
    if not response_text.strip():
        return None
    task_id = callback_context.agent_name.split("_")[-1]
    step = callback_context.state.get(f"refine_step_{task_id}", 0)
    callback_context.state[f"refine_plans_{step}_{task_id}"].append(
        response_text
    )
    return None


use_data_leakage_checker = config.CONFIG.use_data_leakage_checker
refinement_parallel_sub_agents = []
for k in range(config.CONFIG.num_solutions):
    ablation_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"ablation_agent_{k + 1}",
        description="Perform ablation studies to improve the solution.",
        instruction=get_ablation_agent_instruction,
        tools=[skill_tool_util.get_skill_toolset()],    # this may overwrite python script output and gets evaluated to 0 instead of returning actual code. didnt have tools before, hence it may fail due to overwrite (or expeciting code first)
        before_model_callback=check_ablation_finish,    # will return 0 if no code to evaluate
        after_model_callback=functools.partial(
            debug_util.get_code_from_response,  # fails if model returns only tool output (no ablation code, only 'load_skills()' used). Empty code evaluated and returns 0.
            do_eval=not use_data_leakage_checker,
        ),
        generate_content_config=types.GenerateContentConfig(
            temperature=config.get_compatible_temperature(
                config.CONFIG.agent_model, 1.0
            ),
        ),
        include_contents="none",
    )
    ablation_sequential_sub_agents = [ablation_agent]
    if use_data_leakage_checker:
        data_leakage_checker_agent = (
            check_leakage_util.get_data_leakage_checker_agent(
                prefix="ablation",
                suffix=f"{k + 1}",
            )
        )
        ablation_sequential_sub_agents.append(data_leakage_checker_agent)
        additional_agent_description = (
            " and check if there are data leakage issues"
        )
    else:
        additional_agent_description = ""
    ablation_sequential_agent = agents.SequentialAgent(
        name=f"ablation_sequential_agent_{k + 1}",
        description=f"Perform ablation studies{additional_agent_description}.",
        sub_agents=ablation_sequential_sub_agents,
    )
    debug_inner_loop_agent = debug_util.get_debug_inner_loop_agent(
        prefix="ablation",
        suffix=f"{k + 1}",
    )
    ablation_and_debug_loop_agent = agents.LoopAgent(
        name=f"ablation_and_debug_loop_agent_{k + 1}",
        description="Perform ablation studies and debug the code until it succeeds.",
        sub_agents=[
            ablation_sequential_agent,
            debug_inner_loop_agent,
        ],
        max_iterations=config.CONFIG.max_rollback_round,
    )
    ablation_summary_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"ablation_summary_agent_{k + 1}",
        description="Summarize the ablation study results.",
        instruction=get_ablation_summary_agent_instruction,
        after_model_callback=get_ablation_summary,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.get_compatible_temperature(
                config.CONFIG.agent_model, 0.0
            ),
        ),
        include_contents="none",
    )
    init_plan_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"init_plan_agent_{k + 1}",
        description="Generate an initial plan and a code block.",
        instruction=get_init_plan_agent_instruction,
        tools=[skill_tool_util.get_skill_toolset()],
        before_model_callback=check_init_plan_finish,
        after_model_callback=get_plan_and_code_block,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.get_compatible_temperature(
                config.CONFIG.agent_model, 1.0
            ),
        ),
        include_contents="none",
    )
    init_plan_loop_agent = agents.LoopAgent(
        name=f"init_plan_loop_agent_{k + 1}",
        description=(
            "Generate an initial plan and a code block until the code block is valid."
        ),
        sub_agents=[init_plan_agent],
        before_agent_callback=init_inner_loop_states,
        max_iterations=config.CONFIG.max_retry,
    )
    init_plan_implement_agent = debug_util.get_run_and_debug_agent(
        prefix="plan_implement_initial",
        suffix=f"{k + 1}",
        agent_description="Implement the initial plan to generate a solution.",
        instruction_func=get_plan_implement_agent_instruction,
        before_model_callback=check_plan_implement_finish,
        tools=[skill_tool_util.get_skill_toolset()],
    )
    plan_refine_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=f"plan_refine_agent_{k + 1}",
        description="Refine the plan.",
        instruction=get_plan_refinement_instruction,
        tools=[skill_tool_util.get_skill_toolset()],
        after_model_callback=get_refined_plan,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.get_compatible_temperature(
                config.CONFIG.agent_model, 1.0
            ),
        ),
        include_contents="none",
    )
    plan_implement_agent = debug_util.get_run_and_debug_agent(
        prefix="plan_implement",
        suffix=f"{k + 1}",
        agent_description="Implement the plan to generate a solution.",
        instruction_func=get_plan_implement_agent_instruction,
        before_model_callback=check_plan_implement_finish,
        tools=[skill_tool_util.get_skill_toolset()],
    )
    plan_refine_and_implement_agent = agents.SequentialAgent(
        name=f"plan_refine_and_implement_agent_{k + 1}",
        description="Refine the plan and then implement it.",
        sub_agents=[
            plan_refine_agent,
            plan_implement_agent,
        ],
        after_agent_callback=update_inner_loop_states,
    )
    refine_inner_loop_agent = agents.LoopAgent(
        name=f"refine_inner_loop_agent_{k + 1}",
        description="Refine the given solution.",
        sub_agents=[plan_refine_and_implement_agent],
        before_agent_callback=update_inner_loop_states,
        max_iterations=config.CONFIG.inner_loop_round,
    )
    ablation_and_refine_agent = agents.SequentialAgent(
        name=f"ablation_and_refine_agent_{k + 1}",
        description="Perform ablation study and refine the code.",
        sub_agents=[
            ablation_and_debug_loop_agent,
            ablation_summary_agent,
            init_plan_loop_agent,
            init_plan_implement_agent,
            refine_inner_loop_agent,
        ],
        after_agent_callback=update_outer_loop_states,
    )
    ablation_and_refine_loop_agent = agents.LoopAgent(
        name=f"ablation_and_refine_loop_agent_{k + 1}",
        description="Perform ablation study and refine the code for multiple rounds.",
        sub_agents=[ablation_and_refine_agent],
        before_agent_callback=init_outer_loop_states,
        max_iterations=config.CONFIG.outer_loop_round,
    )
    refinement_parallel_sub_agents.append(ablation_and_refine_loop_agent)
refinement_agent = agents.ParallelAgent(
    name="refinement_agent",
    description="Refine each solution by performing ablation studies.",
    sub_agents=refinement_parallel_sub_agents,
    before_agent_callback=None,
)
