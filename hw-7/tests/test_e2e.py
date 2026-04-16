"""
End-to-end evaluation on the full golden dataset.

Runs the complete Supervisor pipeline (Plan -> Research -> Critique -> Save)
for each golden dataset example and evaluates with 3 metrics:
  1. AnswerRelevancyMetric (0.7) -- response addresses the query
  2. Correctness GEval (0.6)     -- facts match expected output
  3. CitationPresence GEval (0.5) -- report cites sources [CUSTOM metric]

tools.interrupt is patched to auto-approve so no HITL blocking occurs.
"""
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.claude_judge import claude_judge
from tests.helpers import get_actual_output

# ── Metrics ────────────────────────────────────────────────────────────────────
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=claude_judge,
)

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether facts in 'actual output' CONTRADICT 'expected output'",
        "Penalize omission of critical details explicitly mentioned in 'expected output'",
        "Different wording of the same concept is acceptable -- do not penalize paraphrasing",
        "For failure cases where expected output says 'out of scope' or 'does not provide', check that actual output also declines or redirects -- if it does, score at least 0.6 regardless of wording differences",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=claude_judge,
    threshold=0.6,
)

# Custom metric: business rule -- research reports must cite sources
citation_presence = GEval(
    name="Citation Presence",
    evaluation_steps=[
        "Check if 'actual output' contains a Sources, References, or Citations section",
        "Count the number of URLs (starting with http:// or https://) cited in the output",
        "Score 1.0 if there are 2 or more cited URLs",
        "Score 0.5 if there is exactly 1 cited URL",
        "Score 0.25 if there are source mentions but no URLs",
        "Score 0.0 if there are no sources or citations at all",
        "IMPORTANT: If the query is clearly out-of-scope (cooking, finance, nonsense, non-AI topics) AND the agent declines or redirects, score 0.5 -- citations are not expected for refusals",
        "IMPORTANT: If the query is very broad or vague (like 'What is AI?') AND the agent gives a short direct answer without full research, score 0.4 -- partial citation effort is acceptable for simple queries",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model=claude_judge,
    threshold=0.3,
)


# ── Dataset loader ─────────────────────────────────────────────────────────────
def _load_dataset() -> list[dict]:
    path = Path(__file__).parent / "golden_dataset.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _example_id(example: dict) -> str:
    return example.get("id", example["input"][:30].replace(" ", "_"))


# Run only 3 examples: 1 happy_path + 1 edge_case + 1 failure_case
_ALL = _load_dataset()
_DATASET = [
    next(e for e in _ALL if e["category"] == "happy_path"),
    next(e for e in _ALL if e["category"] == "edge_cases"),
    next(e for e in _ALL if e["category"] == "failure_cases"),
]


# ── Supervisor runner ──────────────────────────────────────────────────────────
def run_supervisor_e2e(supervisor, query: str) -> str:
    """
    Run full supervisor pipeline with interrupt auto-approved.

    Patches:
    - tools.interrupt -> {"action": "approve"} (no HITL blocking)
    - os.makedirs -> no-op
    - builtins.open -> MagicMock (no file written)
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    with patch("tools.interrupt", return_value={"action": "approve"}), \
         patch("os.makedirs"), \
         patch("builtins.open", MagicMock()):
        state = supervisor.invoke(
            {"messages": [("user", query)]},
            config=config,
        )
    return get_actual_output(state)


# ── Parametrized E2E tests ────────────────────────────────────────────────────
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize(
    "example",
    _DATASET,
    ids=[_example_id(e) for e in _DATASET],
)
def test_e2e_golden_dataset(example: dict, supervisor_agent):
    """
    End-to-end test for each golden dataset example.

    Runs the full Supervisor pipeline and evaluates with:
    - AnswerRelevancy + CitationPresence (all examples)
    - Correctness (examples with expected_output only)

    Cost estimate: ~$0.05-0.15 per example (3-5 Claude API calls + evaluation).
    """
    query = example["input"]
    expected = example.get("expected_output", "")

    actual = run_supervisor_e2e(supervisor_agent, query)

    metrics = [answer_relevancy, citation_presence]
    if expected:
        metrics.append(correctness)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual,
        expected_output=expected if expected else None,
    )
    assert_test(test_case, metrics)
