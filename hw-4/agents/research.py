from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from config import settings, RESEARCHER_PROMPT
from tools import web_search, read_url, knowledge_search


def build_researcher_agent():
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        max_tokens=8192,
    )
    return create_react_agent(
        model=llm,
        tools=[web_search, read_url, knowledge_search],
        prompt=RESEARCHER_PROMPT,
    )


researcher_agent = build_researcher_agent()
