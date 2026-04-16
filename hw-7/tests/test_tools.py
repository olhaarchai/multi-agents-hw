"""
Tool Correctness tests -- ToolCorrectnessMetric.

Verifies that agents call the right tools for given inputs:
1. Planner calls web_search and/or knowledge_search
2. Researcher calls at least 2 search tools
3. Supervisor calls save_report after getting APPROVE from Critic
"""
import uuid
import pytest
from unittest.mock import patch, MagicMock

from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from tests.claude_judge import claude_judge
from tests.conftest import make_mock_retriever
from tests.helpers import extract_tool_calls_from_state, get_actual_output, get_tool_names_called

# Metric (module-level)
tool_metric = ToolCorrectnessMetric(
    threshold=0.5,
    model=claude_judge,
)


def build_tool_calls(state: dict) -> list[ToolCall]:
    """Convert raw state records into deepeval ToolCall objects."""
    records = extract_tool_calls_from_state(state)
    return [
        ToolCall(
            name=r["name"],
            input_parameters=r["input_parameters"],
            output=r["output"],
        )
        for r in records
    ]


# ── Test 1: Planner uses search tools ─────────────────────────────────────────
@pytest.mark.slow
def test_planner_uses_search_tools(planner_agent):
    """
    Planner should call knowledge_search and/or web_search
    when decomposing a research request.
    """
    query = "Compare naive RAG vs sentence-window retrieval"

    with patch("tools.get_retriever", return_value=make_mock_retriever()):
        state = planner_agent.invoke({"messages": [("user", query)]})

    actual_tool_calls = build_tool_calls(state)
    tool_names = [tc.name for tc in actual_tool_calls]

    # Planner must call at least one search tool
    assert any(n in ("knowledge_search", "web_search") for n in tool_names), (
        f"Planner must call knowledge_search or web_search; called: {tool_names}"
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=get_actual_output(state),
        tools_called=actual_tool_calls,
        expected_tools=[ToolCall(name="knowledge_search")],
    )
    assert_test(test_case, [tool_metric])


# ── Test 2: Researcher uses multiple tools ────────────────────────────────────
@pytest.mark.slow
def test_researcher_uses_multiple_tools(researcher_agent):
    """
    Researcher should call web_search at least twice per the RESEARCHER_PROMPT
    instruction ('Always do at least 2 web searches').
    """
    query = "Explain hybrid retrieval combining BM25 and vector search"

    with patch("tools.get_retriever", return_value=make_mock_retriever()):
        state = researcher_agent.invoke({"messages": [("user", query)]})

    actual_tool_calls = build_tool_calls(state)
    tool_names = [tc.name for tc in actual_tool_calls]

    web_calls = [n for n in tool_names if n == "web_search"]
    assert len(web_calls) >= 2, (
        f"Researcher must call web_search at least 2 times; got: {tool_names}"
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=get_actual_output(state),
        tools_called=actual_tool_calls,
        expected_tools=[
            ToolCall(name="web_search"),
            ToolCall(name="web_search"),
        ],
    )
    assert_test(test_case, [tool_metric])


# ── Test 3: Supervisor calls save_report ──────────────────────────────────────
@pytest.mark.slow
def test_supervisor_calls_save_report(supervisor_agent):
    """
    Full supervisor pipeline: after Critic APPROVEs, Supervisor must call save_report.

    tools.interrupt is patched to auto-approve so no HITL blocking occurs.
    builtins.open and os.makedirs are patched to prevent file creation.
    """
    query = "What is FAISS and why is it used in RAG systems?"
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    with patch("tools.interrupt", return_value={"action": "approve"}), \
         patch("os.makedirs"), \
         patch("builtins.open", MagicMock()):
        state = supervisor_agent.invoke(
            {"messages": [("user", query)]},
            config=config,
        )

    tool_names = get_tool_names_called(state)
    assert "save_report" in tool_names, (
        f"Supervisor must call save_report after APPROVE; called: {tool_names}"
    )

    actual_tool_calls = build_tool_calls(state)
    test_case = LLMTestCase(
        input=query,
        actual_output=get_actual_output(state),
        tools_called=actual_tool_calls,
        expected_tools=[
            ToolCall(name="plan"),
            ToolCall(name="research"),
            ToolCall(name="save_report"),
        ],
    )
    assert_test(test_case, [tool_metric])
