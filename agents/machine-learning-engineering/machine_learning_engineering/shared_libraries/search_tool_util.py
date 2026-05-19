"""
Search tool selection utilities for model-compatible web search. Used in both initialization and debug agents.

This module solves the Gemini vs non-Gemini search-tool compatibility:

- If model is Gemini (`is_gemini_model(model_name)`): use ADK `google_search`.
- Otherwise: use a custom DuckDuckGo function tool (`ddg_web_search`) via `duckduckgo_search`.

So:

- `ROOT_AGENT_MODEL=gemini-*` -> native `google_search`
- `ROOT_AGENT_MODEL=openai/...` or other non-Gemini -> DDG tool
"""

from __future__ import annotations

from google.adk.tools import FunctionTool
from google.adk.tools.google_search_tool import google_search
from google.adk.utils.model_name_utils import is_gemini_model


# Added for ChatAI search compatibility: DDG-backed search for non-Gemini models.
def ddg_web_search(query: str, max_results: int = 5) -> str:
    """Searches the web with DuckDuckGo and returns compact citation text."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No search results were found."
        formatted = []
        for idx, item in enumerate(results, 1):
            title = item.get("title", "Untitled")
            url = item.get("href", "")
            snippet = item.get("body", "")
            formatted.append(
                f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}".strip()
            )
        return "\n\n".join(formatted)
    except Exception as exc:
        # Added for ChatAI search compatibility: keep tool failures non-fatal.
        return f"Web search failed: {exc}"


ddg_search_tool = FunctionTool(func=ddg_web_search)


def get_search_tools(model_name: str) -> list:
    """Returns model-compatible search tools for ADK agents."""
    # Replaced google_search for non-Gemini compatibility.
    if is_gemini_model(model_name):
        return [google_search]
    return [ddg_search_tool]
