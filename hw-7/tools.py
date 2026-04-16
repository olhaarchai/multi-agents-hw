import os
import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from config import settings
from retriever import get_retriever


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns a list of results with title, URL, and snippet."""
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


@tool
def read_url(url: str) -> str:
    """Fetch and extract the main text content from a URL. Returns up to 5000 characters."""
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


class SaveReportInput(BaseModel):
    filename: str = Field(description="Filename for the report, e.g. 'rag_comparison.md'")
    content: str = Field(description="Full Markdown content of the report")


@tool("save_report", args_schema=SaveReportInput)
def save_report(filename: str, content: str) -> str:
    """Save the research report to a file. Requires user approval before saving."""
    filename = str(filename).strip().replace("/", "_").replace("\\", "_")
    if not filename.endswith(".md"):
        filename += ".md"

    decision = interrupt({
        "filename": filename,
        "content": content,
    })

    action = decision.get("action", "reject")

    if action == "approve":
        output_dir = str(settings.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report saved to {path}"
    elif action == "edit":
        feedback = decision.get("feedback", "")
        return f"User requested edits: {feedback}. Please revise the report and call save_report again with the updated content."
    else:
        return "User rejected saving the report. Do not retry."


@tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base. Use for questions about RAG, LLMs, LangChain, or other ingested documents."""
    try:
        retriever = get_retriever()
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
