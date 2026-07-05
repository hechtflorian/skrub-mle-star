"""Utility functions for debug agents."""

import functools

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_request as llm_request_module
#from google.adk.tools.google_search_tool import google_search
from google.adk.models import llm_response as llm_response_module
from google.genai import types

from machine_learning_engineering.shared_libraries import (
    check_leakage_util,
    code_util,
    common_util,
    config,
    debug_prompt,
    skill_tool_util,
)


def check_rollback(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Checks if rollback is needed and updates related states."""
    agent_name = callback_context.agent_name
    if agent_name.startswith("ablation"):
        return None
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    code_execution_result_state_key = (
        code_util.get_code_execution_result_state_key(
            agent_name=agent_name,
            suffix=suffix,
        )
    )
    result_dict = callback_context.state.get(
        code_execution_result_state_key, {}
    )
    if result_dict.get("returncode", 1) == 1:
        # Rollback is needed
        callback_context.state[code_execution_result_state_key] = {}
    return None


def get_bug_summary(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
    prefix: str,
) -> llm_response_module.LlmResponse | None:
    """Gets the bug summary."""
    response_text = common_util.get_text_from_response(llm_response)
    clean_bug = response_text.replace("```", "")
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    key_name = code_util.get_name_with_prefix_and_suffix(
        base_name="bug_summary",
        prefix=prefix,
        suffix=suffix,
    )
    callback_context.state[key_name] = clean_bug
    return None


def skip_bug_summary(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
    prefix: str,
) -> llm_response_module.LlmResponse | None:
    """Skips the bug summary if there are no bugs."""
    agent_name = callback_context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    code_execution_result_state_key = (
        code_util.get_code_execution_result_state_key(
            agent_name=agent_name,
            suffix=suffix,
        )
    )
    result_dict = callback_context.state.get(
        code_execution_result_state_key, {}
    )
    if result_dict and result_dict.get("returncode", 1) == 0:
        key_name = code_util.get_name_with_prefix_and_suffix(
            base_name="bug_summary",
            prefix=prefix,
            suffix=suffix,
        )
        callback_context.state[key_name] = ""
        return llm_response_module.LlmResponse()
    return None


def check_bug_existence(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> llm_response_module.LlmResponse | None:
    """Checks if a bug exists in the code."""
    agent_name = callback_context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    code_execution_result_state_key = (
        code_util.get_code_execution_result_state_key(
            agent_name=agent_name,
            suffix=suffix,
        )
    )
    result_dict = callback_context.state.get(
        code_execution_result_state_key, {}
    )
    if result_dict and result_dict.get("returncode", 1) == 0:
        return llm_response_module.LlmResponse()
    return None


def get_bug_summary_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the bug summary agent instruction."""
    agent_name = context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=context)
    code_execution_result_state_key = (
        code_util.get_code_execution_result_state_key(
            agent_name=agent_name,
            suffix=suffix,
        )
    )
    result_dict = context.state.get(code_execution_result_state_key, {})
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        filename = f"init_code_{model_id}.py"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        filename = f"train0_{reference_idx}.py"
    elif agent_name.startswith("check_data_use"):
        filename = "train0.py"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = context.state.get(f"refine_step_{task_id}", 0)
        filename = f"ablation_{step}.py"
    elif agent_name.startswith("plan_implement"):
        task_id = context.agent_name.split("_")[-1]
        step = context.state.get(f"refine_step_{task_id}", 0)
        inner_iter = context.state.get(f"inner_iter_{task_id}", 0)
        filename = f"train{step}_improve{inner_iter}.py"
    elif agent_name.startswith("tune_implement"):
        filename = "train_tune_search.py"
    elif agent_name.startswith("tune_bake"):
        filename = "train_tune_baked.py"
    elif agent_name.startswith("ensemble_plan_implement"):
        filename = f"ensemble{suffix}.py"
    elif agent_name.startswith("submission"):
        filename = "final_solution.py"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    bug = result_dict.get("stderr", "")
    return debug_prompt.BUG_SUMMARY_INSTR.format(
        bug=bug,
        filename=filename,
    )


def _get_backbone_contract(
    context: callback_context_module.ReadonlyContext,
    agent_name: str,
    suffix: str,
) -> str:
    """Deterministic anti-drift context from retriever/structural anchor (for debug agent)."""
    mode = code_util.backbone_check_mode(agent_name)
    is_plan = agent_name.startswith("plan_implement")
    is_ablation = agent_name.startswith("ablation")
    if mode is None and not is_plan and not is_ablation:
        return ""

    required, label = code_util.resolve_backbone_required(
        context, agent_name, suffix
    )
    if not required:
        if is_plan:
            return (
                "\n# Backbone contract\n"
                "- Preserve the structural solution's model family and feature blocks; "
                "do not swap backbone to silence errors unless truly unfixable "
                "without a class change.\n"
            )
        if is_ablation:
            return (
                "\n# Backbone contract\n"
                "- Baseline ablation must keep the input solution's model family.\n"
            )
        return ""

    required_canon = sorted(code_util.canonical_estimator_set(required))
    if is_plan:
        guidance = (
            "Fix the error in place — do not swap model family to silence "
            "errors unless the bug is truly unfixable without a class change."
        )
    elif mode == "overlap":
        guidance = (
            "Keep at least one anchor estimator; do not swap to a "
            "different model family."
        )
    else:
        guidance = "Fix the error without swapping to a different model family."

    return (
        "\n# Backbone contract\n"
        f"- Anchor estimators ({label}): {required_canon}. {guidance}\n"
    )


def get_debug_agent_instruction(
    context: callback_context_module.ReadonlyContext,
    prefix: str,
) -> str:
    """Gets the debug agent instruction."""
    task_description = context.state.get("task_description", "")
    agent_name = context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=context)
    key_name = code_util.get_name_with_prefix_and_suffix(
        base_name="bug_summary",
        prefix=prefix,
        suffix=suffix,
    )
    bug = context.state.get(key_name, "")
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = context.state.get(code_state_key, "")
    return debug_prompt.BUG_REFINE_INSTR.format(
        task_description=task_description,
        code=code,
        bug=bug,
        backbone_contract=_get_backbone_contract(context, agent_name, suffix),
    )


def get_code_from_response(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
    do_eval: bool = True,
) -> llm_response_module.LlmResponse | None:
    """Gets the code from the response."""
    response_text = common_util.get_text_from_response(llm_response)
    code = response_text.replace("```python", "").replace("```", "")
    agent_name = callback_context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    if agent_name.startswith("check_data_use"):
        if "All the provided information is used" in code:
            callback_context.state[f"check_data_use_finish_{suffix}"] = True
        check_data_use_finish = callback_context.state.get(
            f"check_data_use_finish_{suffix}", False
        )
        if not check_data_use_finish:
            new_code = code
        else:
            return None
    elif agent_name.startswith("plan_implement"):
        if "debug" not in agent_name:
            task_id = callback_context.agent_name.split("_")[-1]
            step = callback_context.state.get(f"refine_step_{task_id}", 0)
            code_block = callback_context.state.get(
                f"refine_code_block_{step}_{task_id}", ""
            )
            prev_code = callback_context.state.get(
                f"train_code_{step}_{task_id}", ""
            )
            if not code.strip():    # return early if no code (tool-call only)
                return None
            if not code_block.strip():
                # Empty extracted block (init_plan never produced a valid
                # one): str.replace("") would insert the new code between
                # every character of prev_code (multi-MB string -> context
                # window explosion). Treat as no-op.
                return None
            new_code = prev_code.replace(code_block, code)
            if new_code == prev_code:    # no change, return early (tool-call only)
                return None
        else:
            new_code = code
    else:
        new_code = code
    callback_context.state[code_state_key] = new_code
    if do_eval:
        code_util.evaluate_code(callback_context=callback_context)
    return None


def get_debug_inner_loop_agent(
    prefix: str,
    suffix: str,
) -> agents.LoopAgent:
    """Gets the debug_inner_loop_agent."""
    bug_summary_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="bug_summary_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Summarize the bug of the code.",
        instruction=get_bug_summary_agent_instruction,
        before_model_callback=functools.partial(
            skip_bug_summary,
            prefix=prefix,
        ),
        after_model_callback=functools.partial(
            get_bug_summary,
            prefix=prefix,
        ),
        generate_content_config=types.GenerateContentConfig(
            temperature=config.get_compatible_temperature(
                config.CONFIG.agent_model, 0.0
            ),
        ),
        include_contents="none",
    )
    debug_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="debug_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Debug the given code.",
        instruction=functools.partial(
            get_debug_agent_instruction,
            prefix=prefix,
        ),
        # Attach skrub DataOps SkillToolset + model-aware search tools.
        tools=skill_tool_util.get_skill_and_search_tools(
            config.CONFIG.agent_model
        ),
        before_model_callback=check_bug_existence,
        after_model_callback=get_code_from_response,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.get_compatible_temperature(
                config.CONFIG.agent_model, 1.0
            ),
        ),
        include_contents="none",
    )
    debug_loop_agent = agents.LoopAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="debug_loop_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Debug the given code until it succeeds.",
        sub_agents=[debug_agent],
        max_iterations=config.CONFIG.max_debug_round,
    )
    debug_inner_loop_agent = agents.LoopAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="debug_inner_loop_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Debug the given code until it succeeds.",
        sub_agents=[
            bug_summary_agent,
            debug_loop_agent,
        ],
        max_iterations=config.CONFIG.max_debug_round,
    )
    return debug_inner_loop_agent


def get_run_and_debug_agent(
    prefix: str,
    suffix: str,
    agent_description: str,
    instruction_func: agents.llm_agent.InstructionProvider,
    before_model_callback: agents.llm_agent.BeforeModelCallback | None,
    tools: list | None = None,
) -> agents.LoopAgent:
    """Gets the run and debug agent."""
    if prefix.startswith("ensemble_plan_implement") or prefix.startswith("submission"):
        use_data_leakage_checker = False
    else:
        use_data_leakage_checker = config.CONFIG.use_data_leakage_checker
    run_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description=f"{agent_description}.",
        instruction=instruction_func,
        before_model_callback=before_model_callback,
        tools=tools or [],
        after_model_callback=functools.partial(
            get_code_from_response,
            do_eval=not use_data_leakage_checker,
        ),
        generate_content_config=types.GenerateContentConfig(
            temperature=config.get_compatible_temperature(
                config.CONFIG.agent_model, 1.0
            ),
        ),
        include_contents="none",
    )
    run_sequential_sub_agents = [run_agent]
    if use_data_leakage_checker:
        data_leakage_checker_agent = (
            check_leakage_util.get_data_leakage_checker_agent(
                prefix=prefix,
                suffix=suffix,
            )
        )
        run_sequential_sub_agents.append(data_leakage_checker_agent)
        additional_agent_description = (
            " and check if there are data leakage issues"
        )
    else:
        additional_agent_description = ""
    run_sequential_agent = agents.SequentialAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="sequential_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description=f"{agent_description}{additional_agent_description}.",
        sub_agents=run_sequential_sub_agents,
    )
    run_loop_agent = agents.LoopAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="loop_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description=(f"{agent_description} until it succeeds."),
        sub_agents=[run_sequential_agent],
        max_iterations=config.CONFIG.max_retry,
    )
    debug_inner_loop_agent = get_debug_inner_loop_agent(
        prefix=prefix,
        suffix=suffix,
    )
    run_and_debug_sequential_agent = agents.SequentialAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="and_debug_sequential_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description=f"{agent_description} and debug the code.",
        sub_agents=[
            run_loop_agent,
            debug_inner_loop_agent,
        ],
        after_agent_callback=check_rollback,
    )
    run_and_debug_loop_agent = agents.LoopAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="and_debug_loop_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description=f"{agent_description} and debug the code until it succeeds.",
        sub_agents=[run_and_debug_sequential_agent],
        max_iterations=config.CONFIG.max_rollback_round,
    )
    return run_and_debug_loop_agent
