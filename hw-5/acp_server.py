from collections.abc import AsyncGenerator

import fastmcp
from langchain_core.messages import AIMessage
from acp_sdk.server import Server, Context
from acp_sdk.models import Message, MessagePart

from config import SEARCH_MCP_URL
from mcp_utils import mcp_tools_to_langchain
from agents.planner import build_planner
from agents.research import build_researcher
from agents.critic import build_critic

server = Server()


def _extract_user_text(input: list[Message]) -> str:
    if not input:
        return ""
    msg = input[0]
    if getattr(msg, "parts", None):
        return "\n".join(str(p.content) for p in msg.parts if getattr(p, "content", None))
    return str(msg)


def _extract_ai_text(state: dict) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


@server.agent()
async def planner(input: list[Message], context: Context) -> AsyncGenerator[Message, None]:
    request = _extract_user_text(input)
    async with fastmcp.Client(SEARCH_MCP_URL) as mcp_client:
        tools = await mcp_tools_to_langchain(mcp_client)
        agent = build_planner(tools)
        state = await agent.ainvoke({"messages": [("user", request)]})
    structured = state.get("structured_response")
    content = (
        structured.model_dump_json(indent=2)
        if structured
        else (_extract_ai_text(state) or "Planner returned no output.")
    )
    yield Message(role="agent", parts=[MessagePart(content=content)])


@server.agent()
async def researcher(input: list[Message], context: Context) -> AsyncGenerator[Message, None]:
    request = _extract_user_text(input)
    async with fastmcp.Client(SEARCH_MCP_URL) as mcp_client:
        tools = await mcp_tools_to_langchain(mcp_client)
        agent = build_researcher(tools)
        state = await agent.ainvoke({"messages": [("user", request)]})
    content = _extract_ai_text(state) or "Researcher returned no output."
    yield Message(role="agent", parts=[MessagePart(content=content)])


@server.agent()
async def critic(input: list[Message], context: Context) -> AsyncGenerator[Message, None]:
    findings = _extract_user_text(input)
    async with fastmcp.Client(SEARCH_MCP_URL) as mcp_client:
        tools = await mcp_tools_to_langchain(mcp_client)
        agent = build_critic(tools)
        state = await agent.ainvoke({"messages": [("user", findings)]})
    structured = state.get("structured_response")
    content = (
        structured.model_dump_json(indent=2)
        if structured
        else (_extract_ai_text(state) or "Critic returned no output.")
    )
    yield Message(role="agent", parts=[MessagePart(content=content)])


if __name__ == "__main__":
    try:
        server.run(host="0.0.0.0", port=8903, self_registration=False)
    except TypeError:
        server.run(host="0.0.0.0", port=8903)
