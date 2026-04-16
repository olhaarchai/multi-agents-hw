from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from config import settings, PLANNER_PROMPT
from schemas import ResearchPlan


def build_planner(tools: list):
    """Build the Planner agent with the given (already-converted) LangChain tools."""
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        max_tokens=4096,
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=PLANNER_PROMPT,
        response_format=ResearchPlan,
    )
