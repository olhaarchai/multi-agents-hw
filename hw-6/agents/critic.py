from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

from config import settings, CRITIC_PROMPT
from schemas import CritiqueResult
from tools import web_search, read_url, knowledge_search


def build_critic_agent():
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key.get_secret_value(),
        max_tokens=4096,
    )
    return create_agent(
        model=llm,
        tools=[web_search, read_url, knowledge_search],
        system_prompt=CRITIC_PROMPT,
        response_format=CritiqueResult,
    )


critic_agent = build_critic_agent().with_config({"recursion_limit": 10})
