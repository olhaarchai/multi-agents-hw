"""
Critic Agent tests -- GEval "Critique Quality" metric.

Tests that the Critic produces specific, actionable critiques:
- APPROVE verdict: gaps list empty or minor, no revision_requests
- REVISE verdict: at least one actionable revision_request
- Booleans (is_fresh, is_complete, is_well_structured) consistent with verdict
"""
import json
import pytest
from unittest.mock import patch

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.claude_judge import claude_judge
from tests.conftest import make_mock_retriever
from tests.helpers import get_actual_output

# Metric (module-level)
critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies SPECIFIC issues, not vague complaints (e.g. 'missing 2024 benchmark data' is specific; 'needs more info' is not)",
        "If verdict is REVISE: check that revision_requests are actionable -- a researcher can directly act on each one",
        "If verdict is APPROVE: check that gaps list is empty or contains only minor items, and revision_requests is empty",
        "If verdict is REVISE: there must be at least one revision_request",
        "Check that is_fresh, is_complete, is_well_structured booleans are consistent with the verdict",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=claude_judge,
    threshold=0.7,
)

# Pre-written research findings used as inputs to the Critic
GOOD_RESEARCH = """
# RAG Comparison: Naive vs Sentence-Window Retrieval

## Naive RAG
- Splits documents into fixed 500-token chunks
- Retrieves top-k by cosine similarity
- Simple to implement, works for basic Q&A

## Sentence-Window Retrieval
- Splits on sentence boundaries
- Returns +/-3 sentence context window around the matched sentence
- Achieves 22% improvement in Answer Relevance over naive approach (LlamaIndex 2024 benchmark)
- Better for context-heavy domains

## Recommendation
Sentence-window is preferred when answers require surrounding context.
Naive RAG is sufficient for simple, self-contained factual queries.

## Sources
- https://blog.llamaindex.ai/rag-retrieval-strategies-2024
- Local knowledge base: retrieval-augmented-generation.pdf, p.12-15
"""

POOR_RESEARCH = """
RAG is important for AI applications. There are different approaches.
Some methods work better than others depending on the situation.
You should choose the right approach for your use case.
Consider various factors when making your decision.
"""


def run_critic(agent, findings: str) -> str:
    """Invoke critic with mocked search tools."""
    with patch("tools.get_retriever", return_value=make_mock_retriever()):
        state = agent.invoke({"messages": [("user", findings)]})
    return get_actual_output(state)


@pytest.mark.component
@pytest.mark.slow
def test_critic_approves_good_research(critic_agent):
    """Critic should APPROVE well-structured, cited research."""
    actual = run_critic(critic_agent, GOOD_RESEARCH)
    test_case = LLMTestCase(input=GOOD_RESEARCH, actual_output=actual)
    assert_test(test_case, [critique_quality])


@pytest.mark.component
@pytest.mark.slow
def test_critic_revises_poor_research(critic_agent):
    """Critic should REVISE vague, uncited, non-specific research."""
    actual = run_critic(critic_agent, POOR_RESEARCH)
    test_case = LLMTestCase(input=POOR_RESEARCH, actual_output=actual)
    assert_test(test_case, [critique_quality])


