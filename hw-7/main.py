import json
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from supervisor import supervisor

# Langfuse
langfuse = get_client()
_langfuse_handler = CallbackHandler()

# Tracks how many times research/critique were called (for round labels)
_research_round = 0
_critique_round = 0


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _pretty_json(raw: str, model_name: str) -> str:
    """Try to parse JSON and return a pretty model-like representation."""
    try:
        data = json.loads(raw)
        lines = [f"  📎 {model_name}("]
        for k, v in data.items():
            lines.append(f"       {k}={json.dumps(v, ensure_ascii=False)},")
        lines.append("     )")
        return "\n".join(lines)
    except Exception:
        return f"  📎 {raw[:300]}"


def _print_supervisor_chunk(chunk: dict) -> None:
    """Print a stream chunk from the Supervisor graph."""
    global _research_round, _critique_round

    for node_name, node_output in chunk.items():
        if node_name == "__interrupt__":
            continue
        messages = node_output.get("messages", []) if isinstance(node_output, dict) else []

        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "tool")
                        args = tc.get("args", {})
                        # Section header
                        if name == "plan":
                            print(f"\n[Supervisor → Planner]")
                        elif name == "research":
                            _research_round += 1
                            print(f"\n[Supervisor → Researcher]  (round {_research_round})")
                        elif name == "critique":
                            _critique_round += 1
                            print(f"\n[Supervisor → Critic]")
                        elif name == "save_report":
                            print(f"\n[Supervisor → save_report]")
                        # Tool call line
                        if name == "save_report":
                            fname = args.get("filename", "")
                            content_len = len(args.get("content", ""))
                            print(f'🔧 {name}(filename="{fname}", content="# ...{content_len} chars")')
                        else:
                            first_val = next(iter(args.values()), "") if args else ""
                            print(f'🔧 {name}("{str(first_val)[:80]}")')
                else:
                    text = _extract_text(msg.content)
                    if text.strip():
                        print(f"\nAgent: {text}")

            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                result = _extract_text(getattr(msg, "content", ""))
                if tool_name == "plan":
                    print(_pretty_json(result, "ResearchPlan"))
                elif tool_name == "critique":
                    print(_pretty_json(result, "CritiqueResult"))
                elif tool_name == "research":
                    # Internal calls were already printed by supervisor.py streaming
                    pass
                elif tool_name == "save_report":
                    # Result after resume — success or feedback message
                    if result.strip():
                        print(f"  📎 {result[:200]}")
                else:
                    print(f"  📎 [{tool_name}] {result[:100]}")


def _handle_interrupt(interrupt_value: dict, config: RunnableConfig) -> None:
    """Show proposed report and ask approve/edit/reject. Loops until resolved."""
    filename = interrupt_value.get("filename", "report.md")
    content = interrupt_value.get("content", "")

    while True:
        print("\n" + "=" * 60)
        print("⏸️  ACTION REQUIRES APPROVAL")
        print("=" * 60)
        print(f'    Tool:  save_report')
        print(f'    Args:  {{"filename": "{filename}", "content": "{content[:60]}..."}}')
        print("=" * 60)

        raw = input("\n  👉 approve / edit / reject: ").strip().lower()

        if raw == "approve":
            for chunk in supervisor.stream(
                Command(resume={"action": "approve"}),
                config=config,
                stream_mode="updates",
            ):
                if "__interrupt__" in chunk:
                    continue
                _print_supervisor_chunk(chunk)
            print(f"\n  ✅ Approved! Report saved to output/{filename}")
            return

        elif raw == "edit":
            feedback = input("  ✏️  Your feedback: ").strip()
            interrupted_again = False
            new_interrupt_value = None
            print("\n[Supervisor revises report based on feedback]")
            for chunk in supervisor.stream(
                Command(resume={"action": "edit", "feedback": feedback}),
                config=config,
                stream_mode="updates",
            ):
                if "__interrupt__" in chunk:
                    interrupted_again = True
                    interrupts = chunk["__interrupt__"]
                    if interrupts:
                        new_interrupt_value = interrupts[0].value
                    break
                _print_supervisor_chunk(chunk)

            if interrupted_again and new_interrupt_value:
                filename = new_interrupt_value.get("filename", filename)
                content = new_interrupt_value.get("content", content)
                interrupt_value = new_interrupt_value
                # Loop continues → show HITL again
            else:
                return  # No new interrupt → done

        elif raw == "reject":
            for chunk in supervisor.stream(
                Command(resume={"action": "reject"}),
                config=config,
                stream_mode="updates",
            ):
                if "__interrupt__" in chunk:
                    continue
                _print_supervisor_chunk(chunk)
            print("\n  ❌ Report rejected. Not saved.")
            return

        else:
            print("  Please type: approve, edit, or reject")


@observe(name="research-system-run")
def _run_query(user_input: str, session_id: str):
    """Run a single user query through the supervisor with Langfuse tracing."""
    global _research_round, _critique_round
    _research_round = 0
    _critique_round = 0

    thread_id = str(uuid.uuid4())
    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=[_langfuse_handler],
    )

    interrupted = False
    interrupt_value = None

    for chunk in supervisor.stream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode="updates",
    ):
        if "__interrupt__" in chunk:
            interrupted = True
            interrupts = chunk["__interrupt__"]
            if interrupts:
                interrupt_value = interrupts[0].value
            break
        _print_supervisor_chunk(chunk)

    if interrupted and interrupt_value:
        _handle_interrupt(interrupt_value, config)


def main():
    print("Multi-Agent Research System (type 'exit' to quit)")
    print("-" * 50)

    session_id = f"session-{uuid.uuid4().hex[:8]}"
    print(f"Langfuse session: {session_id}")

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
            with propagate_attributes(
                session_id=session_id,
                user_id="student",
                tags=["hw-7", "multi-agent", "research"],
            ):
                _run_query(user_input, session_id)
        except KeyboardInterrupt:
            print("\n[interrupted]")

    langfuse.flush()


if __name__ == "__main__":
    main()
