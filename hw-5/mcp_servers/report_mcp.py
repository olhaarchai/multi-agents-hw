import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastmcp import FastMCP
from config import settings

mcp = FastMCP("ReportMCP")


@mcp.tool
def save_report(filename: str, content: str) -> str:
    """Write the report file to disk. Caller is responsible for prior user approval."""
    filename = str(filename).strip().replace("/", "_").replace("\\", "_")
    if not filename.endswith(".md"):
        filename += ".md"
    output_dir = str(settings.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to {path}"


@mcp.resource("resource://output-dir")
def output_dir_info() -> str:
    """Path to the output directory and list of saved reports."""
    output_dir = str(settings.output_dir)
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith(".md")]
        return f"Output dir: {output_dir} | Reports: {files}"
    return f"Output dir: {output_dir} (does not exist yet)"


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8902)
