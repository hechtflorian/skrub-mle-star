"""Utilities to expose file-based ADK skills as agent tools."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

from machine_learning_engineering.shared_libraries import search_tool_util

SKILL_DIR = (
    Path(__file__).resolve().parent.parent / "skills" / "skrub-dataops-pipeline"
)


@lru_cache(maxsize=1)
def get_skill_toolset() -> SkillToolset:
    """Builds and caches the native SkillToolset for skrub DataOps."""
    skrub_skill = load_skill_from_dir(SKILL_DIR)
    return SkillToolset(skills=[skrub_skill])


def get_skill_and_search_tools(model_name: str) -> list:
    """Returns native SkillToolset, then model-compatible search tools."""
    return [
        get_skill_toolset(),
        *search_tool_util.get_search_tools(model_name),
    ]
