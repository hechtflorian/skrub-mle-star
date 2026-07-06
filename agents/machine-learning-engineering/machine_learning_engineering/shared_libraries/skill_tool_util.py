"""Utilities to expose file-based ADK skills as agent tools."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from google.adk.skills import load_skill_from_dir
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools.tool_context import ToolContext

from machine_learning_engineering.shared_libraries import search_tool_util

SKILL_DIR = (
    Path(__file__).resolve().parent.parent / "skills" / "skrub-dataops-pipeline"
)

SKILL_LOG_TOOLS = frozenset(
    {"list_skills", "load_skill", "load_skill_resource", "run_skill_script"}
)


# --- Helper functions for skill tool logging ---
def _skill_logging_enabled() -> bool:
    return os.environ.get("MLE_STAR_LOG_SKILL_TOOLS", "1") != "0"


def _format_skill_call(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "list_skills":
        return "list_skills()"
    if tool_name == "load_skill":
        return f'load_skill(name="{args.get("name", "?")}")'
    if tool_name == "load_skill_resource":
        return (
            f'load_skill_resource(skill_name="{args.get("skill_name", "?")}", '
            f'path="{args.get("path", "?")}")'
        )
    if tool_name == "run_skill_script":
        script_path = args.get("script_path", "?")
        return (
            f'run_skill_script(skill_name="{args.get("skill_name", "?")}", '
            f'script_path="{script_path}")'
        )
    return tool_name


def _log_skill_tool_call(agent_name: str, tool_name: str, args: dict[str, Any]) -> None:
    if not _skill_logging_enabled():
        return
    print(
        f"[{agent_name}:skill] {_format_skill_call(tool_name, args)}",
        flush=True,
    )


def _track_loaded_skill_resource(
    tool_context: ToolContext,
    args: dict[str, Any],
) -> None:
    skill_name = args.get("skill_name")
    resource_path = args.get("path")
    if not skill_name or not resource_path:
        return
    state_key = f"_adk_loaded_skill_resources_{tool_context.agent_name}"
    loaded_resources = list(tool_context.state.get(state_key, []))
    entry = {"skill_name": skill_name, "path": resource_path}
    if entry not in loaded_resources:
        loaded_resources.append(entry)
        tool_context.state[state_key] = loaded_resources


def _wrap_skill_tool_logging(tool: BaseTool) -> BaseTool:
    if tool.name not in SKILL_LOG_TOOLS:
        return tool

    original_run_async = tool.run_async

    async def logged_run_async(
        *,
        args: dict[str, Any],
        tool_context: ToolContext,
        **kwargs: Any,
    ) -> Any:
        call_args = args or {}
        _log_skill_tool_call(tool_context.agent_name, tool.name, call_args)
        if tool.name == "load_skill_resource":
            _track_loaded_skill_resource(tool_context, call_args)
        return await original_run_async(
            args=call_args,
            tool_context=tool_context,
            **kwargs,
        )

    tool.run_async = logged_run_async
    return tool


# --- Skill tool implementation for agents using adk native skill ---
@lru_cache(maxsize=1)
def get_skill_toolset() -> SkillToolset:
    """Builds and caches the native SkillToolset for skrub DataOps."""
    skrub_skill = load_skill_from_dir(SKILL_DIR)
    toolset = SkillToolset(skills=[skrub_skill])
    toolset._tools = [_wrap_skill_tool_logging(tool) for tool in toolset._tools]
    return toolset


def get_skill_and_search_tools(model_name: str) -> list:
    """Returns native SkillToolset, then model-compatible search tools."""
    return [
        get_skill_toolset(),
        *search_tool_util.get_search_tools(model_name),
    ]
