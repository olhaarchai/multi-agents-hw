"""
Custom DeepEval judge using Anthropic Claude.

DeepEval natively supports only OpenAI. This wrapper implements
DeepEvalBaseLLM so all GEval / ToolCorrectness / AnswerRelevancy
metrics can use Claude as the evaluator model.

Usage in metrics:
    from tests.claude_judge import claude_judge
    metric = GEval(..., model=claude_judge)
"""
import os

from deepeval.models.base_model import DeepEvalBaseLLM
from anthropic import Anthropic


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "claude-haiku-4-5-20251001"):
        self.model_name = model_name
        self._client = None

    def load_model(self):
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = Anthropic(api_key=api_key)
        return self._client

    def generate(self, prompt: str, *args, **kwargs) -> str:
        client = self.load_model()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


# Singleton — import and reuse across all test files
claude_judge = ClaudeJudge()
