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
            loop.append(m)
        elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            loop.append(m)
        else:
            break
    loop.reverse()
    return loop
