from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from config import settings, CRITIC_PROMPT
from schemas import CritiqueResult


def build_critic(tools: list):
    """Build the Critic agent with structured CritiqueResult output."""
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        max_tokens=4096,
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=CRITIC_PROMPT,
        response_format=CritiqueResult,
    )
