"""Message utilities for LangGraph tool loop."""

from langchain_core.messages import AIMessage, ToolMessage

def get_tool_loop_messages(messages: list) -> list:
    """Extract the current agent's tool loop messages (all rounds).

    Walks backward from the end, collecting ToolMessages and
    AIMessages with tool_calls. Stops at an AIMessage without
    tool_calls (previous agent's final response) or any other type.
    """
    loop = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            loop.insert(0, m)
        elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            loop.insert(0, m)
        else:
            break
    return loop


def format_previous_summaries(stage_summaries: list[str]) -> str:
    """Format previous stage summaries as context for the current agent.

    Returns formatted string with all accumulated summaries,
    or empty string if no summaries exist.
    """
    if not stage_summaries:
        return ""

    return "=== 이전 단계 분석 요약 ===\n" + "\n\n".join(stage_summaries) + "\n==="
