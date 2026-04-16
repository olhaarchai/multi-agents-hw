"""
Planner Agent tests — GEval "Plan Quality" metric.

Tests that the Planner produces well-formed ResearchPlan objects:
- Specific, actionable search_queries (not vague)
- sources_to_check relevant for the topic
- output_format matching the request
- goal is clear and scoped
"""
import pytest
from unittest.mock import patch

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.claude_judge import claude_judge
from tests.conftest import make_mock_retriever
from tests.helpers import get_actual_output

# ── Metric (module-level — created once) ──────────────────────────────────────
plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that 'search_queries' contains at least 2 specific queries directly related to the input topic (not vague like 'find information')",
        "Check that 'sources_to_check' includes 'knowledge_base' for AI/RAG/LLM topics, 'web' for current events or broad topics",
        "Check that 'output_format' describes a concrete format (comparison table, structured report, pros/cons list)",
        "Check that 'goal' is a clear, scoped statement of what will be answered",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=claude_judge,
    threshold=0.7,
)


# ── Helper ─────────────────────────────────────────────────────────────────────
def run_planner(agent, query: str) -> str:
    """Invoke planner with mocked knowledge_search (no FAISS required)."""
    with patch("tools.get_retriever", return_value=make_mock_retriever()):
        state = agent.invoke({"messages": [("user", query)]})
    return get_actual_output(state)


# ── Happy Path Tests ───────────────────────────────────────────────────────────
@pytest.mark.component
@pytest.mark.slow
def test_plan_quality_rag_comparison(planner_agent):
    """Planner produces a specific plan for RAG comparison query."""
    query = "Compare naive RAG vs sentence-window retrieval"
    actual = run_planner(planner_agent, query)

    test_case = LLMTestCase(input=query, actual_output=actual)
    assert_test(test_case, [plan_quality])


