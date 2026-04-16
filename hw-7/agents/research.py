from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from config import settings, RESEARCHER_PROMPT
from tools import web_search, read_url, knowledge_search


def build_researcher_agent():
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key.get_secret_value(),
        max_tokens=8192,
    )
    return create_agent(
        model=llm,
        tools=[web_search, read_url, knowledge_search],
        system_prompt=RESEARCHER_PROMPT,
    )


researcher_agent = build_researcher_agent().with_config({"recursion_limit": 6})
