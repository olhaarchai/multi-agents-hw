import os
import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import settings
from state_store import get_last_text


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


class WriteReportInput(BaseModel):
    filename: str = Field(description="Filename for the report, e.g. 'rag_comparison.md'")


@tool("write_report", args_schema=WriteReportInput)
def write_report(filename: str) -> str:
    """Save the research report you just wrote to a file. Call this after writing the complete report in your response."""
    try:
        content = get_last_text()
        if not content:
            return "Error: No report text found. Write the complete report in your response first, then call this tool."
        filename = str(filename).strip().replace("/", "_").replace("\\", "_")
        if not filename.endswith(".md"):
            filename += ".md"
        output_dir = str(settings.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error saving report: {e}"
