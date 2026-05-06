"""Tests for OpenAI-compat vs Gemini search tool routing."""

import pytest


def test_search_tools_compat_when_api_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.com/v1")
    monkeypatch.setenv("ROOT_AGENT_MODEL", "gemini-x")
    from machine_learning_engineering.shared_libraries import config

    tools = config.get_search_tools()
    assert len(tools) == 1
    assert getattr(tools[0], "name", None) == "web_search"


def test_search_tools_compat_when_openai_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "openai/some-model")
    from machine_learning_engineering.shared_libraries import config

    tools = config.get_search_tools()
    assert len(tools) == 1
    assert tools[0].name == "web_search"


def test_search_tools_native_gemini(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("ROOT_AGENT_MODEL", "gemini-2.0-flash-001")
    from machine_learning_engineering.shared_libraries import config

    tools = config.get_search_tools()
    assert len(tools) == 1
    assert getattr(tools[0], "name", None) == "google_search"
