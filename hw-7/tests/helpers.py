"""
Helpers for extracting structured data from LangGraph agent states.

Used across all test files to:
- Extract tool call records for ToolCorrectnessMetric
- Extract retrieval context for Groundedness metric
- Extract the final text output for LLMTestCase.actual_output
"""
import json

from langchain_core.messages import AIMessage, ToolMessage


# Tools whose outputs count as retrieval context for groundedness evaluation
RETRIEVAL_TOOL_NAMES = {"knowledge_search", "web_search", "read_url"}


def extract_tool_calls_from_state(state: dict) -> list[dict]:
    """
    Extract tool call records from a LangGraph agent state.

    Matches AIMessage.tool_calls (id, name, args) with the corresponding
    ToolMessage responses (tool_call_id, content).

    Returns list of dicts:
        [{"name": "web_search", "input_parameters": {...}, "output": "..."}]
    """
    messages = state.get("messages", [])

    # Collect pending tool calls keyed by tool_call_id
    pending: dict[str, tuple[str, dict]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                pending[tc["id"]] = (tc["name"], tc.get("args", {}))

    # Match ToolMessage responses to pending calls
    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None)
            if call_id and call_id in pending:
                name, args = pending[call_id]
                content = msg.content
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                result.append({
                    "name": name,
                    "input_parameters": args,
                    "output": content[:500],  # truncate for readability
                })
    return result


def extract_retrieval_context(state: dict) -> list[str]:
    """
    Extract retrieval context from search/retrieval ToolMessage outputs.

    Collects outputs from knowledge_search, web_search, and read_url tools.
    Used to populate LLMTestCase.retrieval_context for the Groundedness metric.
    """
    context = []
    for msg in state.get("messages", []):
        if (
            isinstance(msg, ToolMessage)
            and getattr(msg, "name", "") in RETRIEVAL_TOOL_NAMES
        ):
            content = msg.content
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)
            if isinstance(content, str) and content.strip():
                context.append(content)
    return context


def get_actual_output(state: dict) -> str:
    """
    Extract the final text output from an agent state.

    - Planner / Critic: serializes structured_response as JSON
      (so the evaluator can inspect all fields like search_queries, verdict, etc.)
    - Researcher / Supervisor: returns the last non-tool-calling AIMessage text.
    """
    structured = state.get("structured_response")
    if structured is not None:
        return json.dumps(structured.model_dump(), indent=2, ensure_ascii=False)

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
                if text.strip():
                    return text
    return ""


def get_tool_names_called(state: dict) -> list[str]:
    """Return ordered list of tool names called in the agent state."""
    names = []
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                names.append(tc["name"])
    return names
