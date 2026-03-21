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
                                first_arg = next(iter(args.values()), "") if args else ""
                                print(f"  -> {tc['name']}({str(first_arg)[:70]})")
                        elif hasattr(msg, "content"):
                            text = _extract_text(msg.content)
                            if text.strip():
                                print(f"\nAgent: {text}")

                # Tools node: показуємо результат (включно з помилками)
                if "tools" in chunk:
                    for msg in chunk["tools"].get("messages", []):
                        tool_name = getattr(msg, "name", "tool")
                        result = _extract_text(getattr(msg, "content", ""))
                        if result.startswith("Error") or result.startswith("Search error"):
                            print(f"  [{tool_name}] ERROR: {result[:300]}")
                        else:
                            print(f"  [{tool_name}] ok ({len(result)} chars)")

        except KeyboardInterrupt:
            print("\n[interrupted]")


if __name__ == "__main__":
    main()
