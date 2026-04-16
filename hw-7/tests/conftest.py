"""
Shared fixtures and configuration for the hw-6 test suite.

sys.path is managed via pytest.ini (pythonpath = .) — hw-6/ is on path
before any test file loads.

.env is loaded using the absolute path relative to this conftest.py,
so tests work regardless of the working directory.

After every test session, results are saved to output/eval_results.json
via the pytest_sessionfinish hook.
"""
import datetime
import json
import os
import uuid
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document

# ── Environment ────────────────────────────────────────────────────────────────
_HW6_DIR = Path(__file__).parent.parent.resolve()
load_dotenv(_HW6_DIR / ".env", override=False)

if not os.environ.get("ANTHROPIC_API_KEY"):
    warnings.warn(
        "ANTHROPIC_API_KEY is not set. Agent calls and ClaudeJudge will fail. "
        "Add ANTHROPIC_API_KEY to hw-6/.env",
        UserWarning,
        stacklevel=1,
    )


# ── Pytest markers ─────────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that make real LLM API calls")
    config.addinivalue_line("markers", "e2e: marks end-to-end supervisor pipeline tests")
    config.addinivalue_line("markers", "component: marks component-level agent tests")


# ── Golden Dataset ─────────────────────────────────────────────────────────────
_GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"


def load_golden_dataset(category: str | None = None) -> list[dict]:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        examples = json.load(f)
    if category:
        examples = [e for e in examples if e["category"] == category]
    return examples


@pytest.fixture(scope="session")
def golden_dataset() -> list[dict]:
    return load_golden_dataset()


@pytest.fixture(scope="session")
def happy_path_examples() -> list[dict]:
    return load_golden_dataset("happy_path")


# ── Mock Retriever ─────────────────────────────────────────────────────────────
_MOCK_DOCS = [
    "Naive RAG splits documents into fixed-size chunks and uses cosine similarity for retrieval.",
    "Sentence-window retrieval splits on sentence boundaries and returns surrounding context window of ±3 sentences.",
    "BM25 is a lexical sparse retrieval method using term frequency and inverse document frequency (IDF).",
    "FAISS (Facebook AI Similarity Search) enables efficient approximate nearest neighbor search for dense vectors.",
    "Hybrid retrieval combines dense (vector) and sparse (BM25) scores, typically via Reciprocal Rank Fusion (RRF).",
    "Parent-child retrieval uses small child chunks for search but returns the larger parent context for generation.",
    "LangChain ReAct agents interleave reasoning steps (Thought) with tool calls (Action) iteratively until done.",
    "RAG evaluation metrics: Faithfulness (no hallucination), Answer Relevancy, Context Recall, Context Precision.",
    "Cross-encoder rerankers score each (query, document) pair independently to improve retrieval precision.",
    "Contextual compression in LangChain reduces retrieved documents to only the relevant portions.",
]


def make_mock_retriever(doc_texts: list[str] | None = None) -> MagicMock:
    """
    Build a mock LangChain retriever returning synthetic RAG/LLM documents.

    Patch target: patch('tools.get_retriever', return_value=make_mock_retriever())

    Note: patch 'tools.get_retriever' (not 'retriever.get_retriever') because
    knowledge_search() uses the name imported into tools.py's namespace.
    """
    texts = doc_texts or _MOCK_DOCS
    docs = [
        Document(
            page_content=text,
            metadata={"source": f"mock_doc_{i}.pdf", "page": i},
        )
        for i, text in enumerate(texts)
    ]
    mock = MagicMock()
    mock.invoke.return_value = docs
    return mock


@pytest.fixture
def mock_retriever() -> MagicMock:
    return make_mock_retriever()


# ── Agent Fixtures (session-scoped — pay import cost once) ────────────────────
@pytest.fixture(scope="session")
def planner_agent():
    from agents.planner import planner_agent as _agent
    return _agent


@pytest.fixture(scope="session")
def researcher_agent():
    from agents.research import researcher_agent as _agent
    return _agent


@pytest.fixture(scope="session")
def critic_agent():
    from agents.critic import critic_agent as _agent
    return _agent


@pytest.fixture(scope="session")
def supervisor_agent():
    from supervisor import supervisor as _agent
    return _agent


# ── Thread config (function-scoped — fresh UUID per test) ─────────────────────
@pytest.fixture
def thread_config() -> dict:
    """Fresh thread_id for each supervisor test (InMemorySaver requires it)."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


# ── Save eval results to output/eval_results.json after session ───────────────
def pytest_sessionfinish(session, exitstatus):
    """
    After all tests complete, collect deepeval results from the global
    test run manager and save them to output/eval_results.json.

    File structure:
    {
      "timestamp": "2026-04-16T14:00:00",
      "summary": {"total": 30, "passed": 22, "failed": 8, "pass_rate": 0.73},
      "results": [
        {
          "test_name": "test_plan_quality_rag_comparison",
          "input": "...",
          "actual_output": "...",
          "success": true,
          "metrics": [{"name": "Plan Quality", "score": 0.85, "passed": true, "threshold": 0.7}]
        }
      ]
    }
    """
    try:
        from deepeval.test_run import global_test_run_manager

        test_run = global_test_run_manager.test_run
        if test_run is None or not test_run.test_cases:
            return

        results = []
        for tc in test_run.test_cases:
            metrics = []
            if tc.metrics_data:
                for m in tc.metrics_data:
                    metrics.append({
                        "name": m.name,
                        "score": round(m.score, 4) if m.score is not None else None,
                        "threshold": m.threshold,
                        "passed": m.success,
                        "reason": m.reason,
                    })

            results.append({
                "test_name": tc.name,
                "input": tc.input,
                "actual_output": (tc.actual_output or "")[:300],
                "success": tc.success,
                "metrics": metrics,
            })

        total = len(results)
        passed = sum(1 for r in results if r["success"] is True)
        failed = total - passed

        output = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / total, 3) if total else 0,
            },
            "results": results,
        }

        out_dir = _HW6_DIR / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "eval_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n📊 Eval results saved → {out_path}")
        print(f"   Total: {total} | Passed: {passed} | Failed: {failed} | Pass rate: {output['summary']['pass_rate']:.0%}")

    except Exception as e:
        # Never crash the test session because of result saving
        print(f"\n⚠️  Could not save eval results: {e}")
