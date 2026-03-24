"""Agent A — Discovery subgraph with tool_call loop."""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts
from app.utils.messages import get_tool_loop_messages
from app.services.discovery_service import parse_gene_list
from app.services.summary_service import summarize_discovery


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
        print("\n[Agent A - Discovery] LLM 호출 중...")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"질환/주제: {state['query']}\n"
                "차등발현유전자(DEG)를 조회하여 바이오마커 후보 목록을 도출하세요."
            )),
        ]
        messages.extend(get_tool_loop_messages(state.get("messages", [])))
        response = await llm_with_tools.ainvoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            print(f"[Agent A - Discovery] 도구 호출 요청: {tool_names}")
        return {"messages": [response]}

    async def tool_node(state: AgentState):
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool = tools_by_name.get(tool_call["name"])
            if tool is None:
                results.append(ToolMessage(
                    content=f"Error: unknown tool '{tool_call['name']}'",
                    tool_call_id=tool_call["id"],
                ))
                continue
            print(f"[Agent A - Discovery] 도구 실행: {tool_call['name']}({tool_call['args']})")
            try:
                result = await tool.ainvoke(tool_call["args"])
                results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            except Exception as e:
                results.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"]))
        return {"messages": results}

    def should_continue(state: AgentState) -> Literal["tool_node", "extract"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return "extract"

    async def extract_candidates(state: AgentState):
        """Extract candidate gene list from the final LLM response."""
        last_message = state["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        candidates = parse_gene_list(content)
        summary = summarize_discovery(content, candidates)
        print(f"[Agent A - Discovery] 완료! 후보 유전자: {candidates}")
        print("-" * 50)
        return {
            "candidates": candidates,
            "stage_summaries": [summary],
        }

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
