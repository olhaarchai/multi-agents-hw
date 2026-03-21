import os

import trafilatura
from ddgs import DDGS

from config import settings


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns title, URL, and snippet for each result."""
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


def write_report(filename: str, content: str) -> str:
    """Save the research report to a Markdown file."""
    try:
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


TOOLS_SCHEMA = [
    {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo. Returns a list of results with title, URL, and snippet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_url",
        "description": "Fetch and extract the main text content from a URL. Returns up to 5000 characters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "write_report",
        "description": "Save the research report to a Markdown file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename for the report, e.g. 'rag_comparison.md'",
                },
                "content": {
                    "type": "string",
                    "description": "Full Markdown content of the report",
                },
            },
            "required": ["filename", "content"],
        },
    },
]
