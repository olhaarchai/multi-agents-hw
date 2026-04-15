from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from agents.planner import planner_agent
from agents.research import researcher_agent
from agents.critic import critic_agent
from config import settings, SUPERVISOR_PROMPT
from tools import save_report  # HITL tool


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _stream_agent(agent, request: str) -> dict:
    """Stream an agent in values mode — one LLM pass. Prints intermediate tool calls
    with 2-space indent. Returns the final state dict (contains 'messages' and optionally
    'structured_response')."""
    final_state: dict = {}
    seen_ids: set = set()
    for state in agent.stream(
        {"messages": [("user", request)]},
        stream_mode="values",
    ):
        final_state = state
        for msg in state.get("messages", []):
            mid = getattr(msg, "id", None) or id(msg)
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("args", {})
                    args_str = ", ".join(
                        f'{k}="{str(v)[:80]}"' for k, v in args.items()
                    )
                    print(f"  🔧 {name}({args_str})")
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "")
                content = _extract_text(getattr(msg, "content", ""))
                if tool_name == "knowledge_search":
                    count = len([x for x in content.split("\n\n") if x.strip()])
                    print(f"  📎 [{count} documents found]")
                elif tool_name == "web_search":
                    count = content.count("URL:")
                    print(f"  📎 [{count} results found]")
                elif tool_name == "read_url":
                    print(f"  📎 [{len(content)} chars]")
    return final_state


def _final_ai_text(state: dict) -> str:
    """Extract last AI message text from state (for Research agent)."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = _extract_text(msg.content)
            if text.strip():
                return text
    return ""


# ── Tool wrappers for sub-agents (single-pass via stream_mode="values") ───────

@tool
def plan(request: str) -> str:
    """Decompose the user research request into a structured ResearchPlan using the Planner Agent."""
    state = _stream_agent(planner_agent, request)
    structured = state.get("structured_response")
    if structured is not None:
        return structured.model_dump_json(indent=2)
    return _final_ai_text(state) or "Planner returned no output."


@tool
def research(request: str) -> str:
    """Execute research following the given plan/request using the Research Agent."""
    state = _stream_agent(researcher_agent, request)
    return _final_ai_text(state) or "Research returned no output."


@tool
def critique(findings: str) -> str:
    """Evaluate the research findings using the Critic Agent. Returns structured CritiqueResult."""
    state = _stream_agent(critic_agent, findings)
    structured = state.get("structured_response")
    if structured is not None:
        return structured.model_dump_json(indent=2)
    return _final_ai_text(state) or "Critic returned no output."


# ── Supervisor agent ──────────────────────────────────────────────────────────

def build_supervisor():
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        max_tokens=8192,
    )
    return create_react_agent(
        model=llm,
        tools=[plan, research, critique, save_report],
        prompt=SUPERVISOR_PROMPT,
        checkpointer=InMemorySaver(),
    )


supervisor = build_supervisor()
