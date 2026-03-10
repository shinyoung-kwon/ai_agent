"""Agent B — Network subgraph with tool_call loop."""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts


def build_network_subgraph(tools: list) -> StateGraph:
    """Build Agent B subgraph that analyzes gene regulatory networks."""
    llm = get_llm()
    prompts = get_prompts()
    system_prompt = prompts["network_agent"]["system"]

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    tools_by_name = {t.name: t for t in tools}

    async def llm_call(state: AgentState):
        candidates = state.get("candidates", [])
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"후보 유전자 목록: {', '.join(candidates)}\n"
                "이 유전자들의 레귤론과 TF-Target 네트워크를 조회하세요."
            )),
        ]
        messages.extend(state.get("messages", []))
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(state: AgentState):
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = await tool.ainvoke(tool_call["args"])
            results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        return {"messages": results}

    def should_continue(state: AgentState) -> Literal["tool_node", "extract"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return "extract"

    async def extract_network(state: AgentState):
        """Extract network data from the final LLM response."""
        last_message = state["messages"][-1]
        return {"network_data": {"raw_response": last_message.content}}

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("extract", extract_network)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "extract": "extract",
    })
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("extract", END)

    return graph.compile()
