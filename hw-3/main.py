from agent import agent


def _extract_text(content) -> str:
    """Handle both str and list-of-blocks content (Anthropic format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            for chunk in agent.stream(
                {"messages": [("user", user_input)]},
                config={"configurable": {"thread_id": "session_1"}},
            ):
                # Agent node: показуємо tool calls що плануються
                if "agent" in chunk:
                    for msg in chunk["agent"].get("messages", []):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                args = tc.get("args", {})
                                args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
                                print(f"\n🔧 Tool call: {tc['name']}({args_str})")
                        elif hasattr(msg, "content"):
                            text = _extract_text(msg.content)
                            if text.strip():
                                print(f"\nAgent: {text}")

                # Tools node: показуємо результат (включно з помилками)
                if "tools" in chunk:
                    for msg in chunk["tools"].get("messages", []):
                        tool_name = getattr(msg, "name", "tool")
                        result = _extract_text(getattr(msg, "content", ""))
                        if result.startswith("Error") or result.startswith("Search error") or result.startswith("Knowledge search error"):
                            print(f"📎 Result: ERROR — {result[:200]}")
                        elif tool_name == "knowledge_search":
                            lines = result.strip().split("\n\n")
                            print(f"📎 Result: [{len(lines)} documents found]")
                            for line in lines:
                                # формат: "N. [source]\n   text"
                                parts = line.split("\n", 1)
                                header = parts[0].strip()
                                body = parts[1].strip() if len(parts) > 1 else ""
                                # витягуємо [source] з header
                                src = header[header.find("[")+1:header.find("]")] if "[" in header else header
                                print(f"   - [{src}] {body[:80]}")
                        elif tool_name == "web_search":
                            count = result.count("\nURL:")
                            print(f"📎 Result: Found {count} results...")
                        elif tool_name == "read_url":
                            print(f"📎 Result: [{len(result)} chars] {result[:80].strip()}...")
                        else:
                            print(f"📎 Result: {result[:100]}")

        except KeyboardInterrupt:
            print("\n[interrupted]")


if __name__ == "__main__":
    main()
