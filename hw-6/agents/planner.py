from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from config import settings, PLANNER_PROMPT
from schemas import ResearchPlan
from tools import web_search, knowledge_search


def build_planner_agent():
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key.get_secret_value(),
        max_tokens=4096,
    )
    return create_agent(
        model=llm,
        tools=[web_search, knowledge_search],
        system_prompt=PLANNER_PROMPT,
        response_format=ResearchPlan,
    )


planner_agent = build_planner_agent()
