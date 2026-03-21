from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config import settings, SYSTEM_PROMPT
from state_store import set_last_text
from tools import web_search, read_url, write_report


# --- LLM ---
llm = ChatAnthropic(
    model=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
    max_tokens=8192,
)

tools = [web_search, read_url, write_report]
llm_with_tools = llm.bind_tools(tools)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""


# --- Nodes ---
def call_model(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    # Save any text content so write_report can use it
    text = _extract_text(response.content)
    if text.strip():
        set_last_text(text)

    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


# --- Graph ---
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)
