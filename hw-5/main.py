import asyncio
import json
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from supervisor import supervisor

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
                        if name == "delegate_to_planner":
                            print("\n[Supervisor → Planner]")
                        elif name == "delegate_to_researcher":
                            _research_round += 1
                            print(f"\n[Supervisor → Researcher]  (round {_research_round})")
                        elif name == "delegate_to_critic":
                            _critique_round += 1
                            print("\n[Supervisor → Critic]")
                        elif name == "save_report":
                            print("\n[Supervisor → save_report]")
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
                if tool_name == "delegate_to_planner":
                    print(_pretty_json(result, "ResearchPlan"))
                elif tool_name == "delegate_to_critic":
                    print(_pretty_json(result, "CritiqueResult"))
                elif tool_name == "delegate_to_researcher":
                    pass
                elif tool_name == "save_report":
                    if result.strip():
                        print(f"  📎 {result[:200]}")
                else:
                    print(f"  📎 [{tool_name}] {result[:100]}")


def _parse_interrupt_payload(raw) -> tuple[str, str]:
    """Extract filename/content from HumanInTheLoopMiddleware interrupt payload.
    HITLRequest shape:
    {
        "action_requests": [{"name": "save_report", "args": {"filename": ..., "content": ...}}],
        "review_configs": [...]
    }
    """
    if isinstance(raw, dict):
        action_requests = raw.get("action_requests", [])
        if action_requests:
            args = action_requests[0].get("args", {})
            return args.get("filename", "report.md"), args.get("content", "")
    return "report.md", ""


async def _drain(stream) -> tuple[bool, object]:
    """Consume stream until interrupt or end. Returns (interrupted, interrupt_value)."""
    interrupted = False
    interrupt_value = None
    async for chunk in stream:
        if "__interrupt__" in chunk:
            interrupted = True
            interrupts = chunk["__interrupt__"]
            interrupt_value = interrupts[0].value if interrupts else None
            break
        _print_supervisor_chunk(chunk)
    return interrupted, interrupt_value


async def _handle_interrupt(interrupt_value, config: dict) -> None:
    """Show proposed report and ask approve/edit/reject. Loops until resolved."""
    filename, content = _parse_interrupt_payload(interrupt_value)
    while True:
        print("\n" + "=" * 60)
        print("⏸️  ACTION REQUIRES APPROVAL")
        print("=" * 60)
        print(f'    Tool:  save_report')
        print(f'    Args:  {{"filename": "{filename}", "content": "{content[:60]}..."}}')
        print("=" * 60)
        raw = input("\n  👉 approve / edit / reject: ").strip().lower()

        if raw == "approve":
            resume_val = {"decisions": [{"type": "approve"}]}
            interrupted, new_val = await _drain(supervisor.astream(
                Command(resume=resume_val), config=config, stream_mode="updates"
            ))
            if interrupted and new_val:
                filename, content = _parse_interrupt_payload(new_val)
                continue
            print(f"\n  ✅ Approved! Report saved to output/{filename}")
            return

        elif raw == "edit":
            feedback = input("  ✏️  Your feedback: ").strip()
            # Reject with feedback message → LLM sees the message and revises
            resume_val = {"decisions": [{"type": "reject", "message": f"Please revise: {feedback}"}]}
            print("\n[Supervisor revises report based on feedback]")
            interrupted, new_val = await _drain(supervisor.astream(
                Command(resume=resume_val), config=config, stream_mode="updates"
            ))
            if interrupted and new_val:
                filename, content = _parse_interrupt_payload(new_val)
                continue
            return

        elif raw == "reject":
            resume_val = {"decisions": [{"type": "reject"}]}
            await _drain(supervisor.astream(
                Command(resume=resume_val), config=config, stream_mode="updates"
            ))
            print("\n  ❌ Report rejected. Not saved.")
            return

        else:
            print("  Please type: approve, edit, or reject")


async def main():
    global _research_round, _critique_round
    print("Multi-Agent Research System (type 'exit' to quit)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            return

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        _research_round = 0
        _critique_round = 0

        try:
            interrupted, interrupt_value = await _drain(supervisor.astream(
                {"messages": [("user", user_input)]},
                config=config,
                stream_mode="updates",
            ))
            if interrupted and interrupt_value:
                await _handle_interrupt(interrupt_value, config)
        except KeyboardInterrupt:
            print("\n[interrupted]")


if __name__ == "__main__":
    asyncio.run(main())
