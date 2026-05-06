"""OpenAI-compat web search via DuckDuckGo (no API key).

Used when agents run through LiteLLM / custom api_base instead of Gemini
native grounding.
"""

from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool


def web_search(query: str, max_results: int = 8) -> str:
    """Search the web for ML methods, repos, docs, or error explanations.

    Call this before answering when up-to-date or external facts are needed.
    Use short, concrete queries.

    Args:
        query: What to search for.
        max_results: Number of result snippets (capped internally).

    Returns:
        Concatenated titles, snippets, and URLs from search hits.
    """
    from duckduckgo_search import DDGS

    q = (query or "").strip()
    if not q:
        return "Error: empty search query."

    n = max(1, min(int(max_results), 12))

    lines: list[str] = []
    try:
        with DDGS() as ddgs:
            for i, hit in enumerate(ddgs.text(q, max_results=n)):
                title = hit.get("title", "").strip()
                body = hit.get("body", "").strip()
                href = hit.get("href", "").strip()
                lines.append(f"[{i + 1}] {title}\n{body}\n{href}")
    except Exception as exc:
        return f"web_search failed: {exc!s}"

    if not lines:
        return "No web results returned; try a different query."

    return "\n\n".join(lines)


compat_web_search_tool = FunctionTool(web_search)
