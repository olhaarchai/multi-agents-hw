import fastmcp
from acp_sdk.client import Client as ACPClient
from acp_sdk.models import Message, MessagePart
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from config import settings, SUPERVISOR_PROMPT, ACP_SERVER_URL, REPORT_MCP_URL


# ── ACP delegation tools (async) ──────────────────────────────────────────────

async def _acp_call(agent_name: str, text: str) -> str:
    async with ACPClient(base_url=ACP_SERVER_URL, headers={"Content-Type": "application/json"}) as client:
        run = await client.run_sync(
            agent=agent_name,
            input=[Message(role="user", parts=[MessagePart(content=text)])],
        )
        if not run.output:
            return f"{agent_name} returned no output."
        out = run.output[0]
        if getattr(out, "parts", None):
            return "\n".join(str(p.content) for p in out.parts if getattr(p, "content", None))
        return str(out)


@tool
async def delegate_to_planner(request: str) -> str:
    """Decompose the user research request into a structured ResearchPlan via the Planner Agent (ACP)."""
    return await _acp_call("planner", request)


@tool
async def delegate_to_researcher(plan: str) -> str:
    """Execute research following the given plan via the Researcher Agent (ACP)."""
    return await _acp_call("researcher", plan)


@tool
async def delegate_to_critic(findings: str) -> str:
    """Evaluate research findings via the Critic Agent (ACP). Returns structured CritiqueResult."""
    return await _acp_call("critic", findings)


# ── save_report tool (delegates to ReportMCP) ─────────────────────────────────

class SaveReportInput(BaseModel):
    filename: str = Field(description="Filename for the report, e.g. 'rag_comparison.md'")
    content: str = Field(description="Full Markdown content of the report")


@tool("save_report", args_schema=SaveReportInput)
async def save_report(filename: str, content: str) -> str:
    """Save the research report to disk via ReportMCP. HITL-protected by the Supervisor middleware."""
    async with fastmcp.Client(REPORT_MCP_URL) as mcp_client:
        result = await mcp_client.call_tool(
            "save_report", {"filename": filename, "content": content}
        )
        if result.content:
            return result.content[0].text
        return "Report saved."


# ── Supervisor with HumanInTheLoopMiddleware ──────────────────────────────────

def build_supervisor():
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        max_tokens=8192,
    )
    hitl = HumanInTheLoopMiddleware(
        interrupt_on={
            "save_report": {
                "allowed_decisions": ["approve", "edit", "reject"],
            }
        }
    )
    return create_agent(
        model=llm,
        tools=[delegate_to_planner, delegate_to_researcher, delegate_to_critic, save_report],
        system_prompt=SUPERVISOR_PROMPT,
        middleware=[hitl],
        checkpointer=InMemorySaver(),
    )


supervisor = build_supervisor()
