from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from config import settings, CRITIC_PROMPT
from schemas import CritiqueResult
from tools import web_search, read_url, knowledge_search


def build_critic_agent():
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        max_tokens=4096,
    )
    return create_react_agent(
        model=llm,
        tools=[web_search, read_url, knowledge_search],
        prompt=CRITIC_PROMPT,
        response_format=CritiqueResult,
    )


critic_agent = build_critic_agent()
