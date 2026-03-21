"""Shared in-memory store for passing data between agent nodes and tools."""

_store: dict = {"last_ai_text": ""}


def set_last_text(text: str) -> None:
    _store["last_ai_text"] = text


def get_last_text() -> str:
    return _store["last_ai_text"]
