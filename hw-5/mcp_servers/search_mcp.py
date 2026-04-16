import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import trafilatura
from ddgs import DDGS
from fastmcp import FastMCP

from config import settings
from retriever import get_retriever

mcp = FastMCP("SearchMCP")
_retriever = None


def _get_retriever_cached():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


@mcp.tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns results with title, URL, and snippet."""
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   URL: {r.get('href', '')}")
            lines.append(f"   Snippet: {r.get('body', '')}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


@mcp.tool
def read_url(url: str) -> str:
    """Fetch and extract main text content from a URL. Returns up to ~5000 characters."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: Could not fetch URL: {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return f"Error: Could not extract text from: {url}"
        return text[: settings.max_url_content_length]
    except Exception as e:
        return f"Error reading URL {url}: {e}"


@mcp.tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base (RAG hybrid: FAISS + BM25 + reranker)."""
    try:
        retriever = _get_retriever_cached()
        docs = retriever.invoke(query)
        if not docs:
            return "No results found in knowledge base."
        lines = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            label = f"{os.path.basename(source)}, p.{page}" if page != "" else os.path.basename(source)
            lines.append(f"{i}. [{label}]\n   {doc.page_content[:300]}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Knowledge search error: {e}"


@mcp.resource("resource://knowledge-base-stats")
def kb_stats() -> str:
    """Number of indexed chunks and index directory."""
    import json
    chunks_path = os.path.join(settings.index_dir, "bm25_chunks.json")
    if not os.path.exists(chunks_path):
        return "Knowledge base not indexed yet. Run: python ingest.py"
    with open(chunks_path) as f:
        data = json.load(f)
    return f"Knowledge base: {len(data)} chunks indexed at '{settings.index_dir}'"


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8901)
