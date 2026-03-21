import anthropic

from config import settings, SYSTEM_PROMPT
from tools import web_search, read_url, write_report, TOOLS_SCHEMA

client = anthropic.Anthropic(api_key=settings.api_key.get_secret_value())

TOOL_FNS = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 80:
            v_str = v_str[:80] + "..."
        parts.append(f'{k}="{v_str}"')
    return ", ".join(parts)


def _run_tool(name: str, args: dict) -> str:
    fn = TOOL_FNS.get(name)
    if not fn:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**args)
    except Exception as e:
        return f"Tool error: {e}"


class ResearchAgent:
    def __init__(self):
        self.messages: list = []  # manual conversation memory

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(settings.max_iterations):
            response = client.messages.create(
                model=settings.model_name,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=TOOLS_SCHEMA,
                messages=self.messages,
            )

            if response.stop_reason == "end_turn":
                text = _extract_text(response.content)
                self.messages.append({"role": "assistant", "content": response.content})
                return text

            if response.stop_reason == "tool_use":
                self.messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"\n🔧 Tool call: {block.name}({_fmt_args(block.input)})")
                        result = _run_tool(block.name, block.input)
                        preview = str(result)[:200]
                        print(f"📎 Result: {preview}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                self.messages.append({"role": "user", "content": tool_results})
                continue

            # unexpected stop reason
            break

        return "Error: max iterations reached without a final response."


agent = ResearchAgent()
