from dotenv import load_dotenv
load_dotenv()

from pydantic import SecretStr
from pydantic_settings import BaseSettings
from langfuse import get_client


class Settings(BaseSettings):
    anthropic_api_key: SecretStr
    model_name: str

    # Web search
    max_search_results: int = 5
    max_url_content_length: int = 5000

    # RAG
    embedding_model: str = "intfloat/multilingual-e5-small"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_n: int = 3

    # Agent
    output_dir: str = "output"
    max_iterations: int = 10

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    model_config = {"env_file": ".env"}


settings = Settings()

# ── Langfuse client (singleton) ─────────────────────────────────────────────
langfuse = get_client()


# ── Prompt loading from Langfuse Prompt Management ──────────────────────────

def load_prompt(name: str, **variables: str) -> str:
    """Load a prompt from Langfuse by name (label=production) and compile with variables."""
    prompt = langfuse.get_prompt(name, label="production")
    return prompt.compile(**variables)


PLANNER_PROMPT = load_prompt("planner_system")
RESEARCHER_PROMPT = load_prompt("researcher_system")
CRITIC_PROMPT = load_prompt("critic_system")
SUPERVISOR_PROMPT = load_prompt("supervisor_system")
