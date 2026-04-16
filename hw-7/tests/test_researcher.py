"""
Researcher Agent tests -- GEval "Groundedness" metric.

Tests that the Researcher's output claims are grounded in actual
retrieval context (what knowledge_search / web_search returned),
not hallucinated.
"""
import pytest
from unittest.mock import patch

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.claude_judge import claude_judge
from tests.conftest import make_mock_retriever
from tests.helpers import get_actual_output, extract_retrieval_context

# Metric (module-level -- created once)
groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every specific factual claim from 'actual output'",
        "For each claim, check if it can be directly supported or inferred from 'retrieval context'",
        "Claims not traceable to retrieval context count as ungrounded, even if generally true",
        "Score = grounded_claims / total_claims (0.0 to 1.0)",
        "If retrieval context is empty, score conservatively",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model=claude_judge,
    threshold=0.6,
)


def run_researcher(agent, query: str) -> tuple[str, list[str]]:
    """Invoke researcher and return (actual_output, retrieval_context)."""
    with patch("tools.get_retriever", return_value=make_mock_retriever()):
        state = agent.invoke({"messages": [("user", query)]})
    actual_output = get_actual_output(state)
    retrieval_context = extract_retrieval_context(state)
    return actual_output, retrieval_context


@pytest.mark.component
@pytest.mark.slow
def test_research_grounded_rag_comparison(researcher_agent):
    """Researcher's RAG comparison report is grounded in retrieved documents."""
    query = "Compare naive RAG vs sentence-window retrieval approaches"
    actual, context = run_researcher(researcher_agent, query)
    test_case = LLMTestCase(input=query, actual_output=actual, retrieval_context=context)
    assert_test(test_case, [groundedness])


