"""Agent A — Discovery subgraph with tool_call loop."""

import re
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts


def build_discovery_subgraph(tools: list) -> StateGraph:
    """Build Agent A subgraph that discovers biomarker candidates."""
    llm = get_llm()
    prompts = get_prompts()
    system_prompt = prompts["discovery_agent"]["system"]

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    tools_by_name = {t.name: t for t in tools}

    async def llm_call(state: AgentState):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"질환/주제: {state['query']}\n"
                "차등발현유전자(DEG)를 조회하여 바이오마커 후보 목록을 도출하세요."
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

    async def extract_candidates(state: AgentState):
        """Extract candidate gene list from the final LLM response."""
        last_message = state["messages"][-1]
        return {"candidates": _parse_gene_list(last_message.content)}

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("extract", extract_candidates)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "extract": "extract",
    })
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("extract", END)

    return graph.compile()


def _parse_gene_list(text: str) -> list[str]:
    """Parse gene symbols from LLM text output."""
    genes = re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', text)
    stopwords = {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "DEG", "TF", "MCP", "RAG", "GOI"}
    return list(dict.fromkeys(g for g in genes if g not in stopwords))
