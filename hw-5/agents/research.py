from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from config import settings, RESEARCHER_PROMPT


def build_researcher(tools: list):
    """Build the Researcher agent. No structured response — returns free-form Markdown."""
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        max_tokens=8192,
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=RESEARCHER_PROMPT,
    )
