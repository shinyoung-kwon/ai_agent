"""Agent B — Network subgraph with tool_call loop."""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts
from app.utils.messages import get_tool_loop_messages
from app.graph.schemas import NetworkOutput


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
        print(f"\n[Agent B - Network] LLM 호출 중... (입력 유전자: {candidates})")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"후보 유전자 목록: {', '.join(candidates)}\n"
                "이 유전자들의 레귤론과 TF-Target 네트워크를 조회하여, 조건별 특이적 TF 후보군을 도출하고 선정된 이유를 간략히 설명해주세요."
            )),
        ]
        messages.extend(get_tool_loop_messages(state.get("messages", [])))
        response = await llm_with_tools.ainvoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            print(f"[Agent B - Network] 도구 호출 요청: {tool_names}")
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
            print(f"[Agent B - Network] 도구 실행: {tool_call['name']}({tool_call['args']})")
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

    async def extract_network(state: AgentState):
        """Extract network data via structured output."""
        last_message = state["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        structured_llm = llm.with_structured_output(NetworkOutput)
        try:
            result = await structured_llm.ainvoke([
                HumanMessage(content=content)
            ])
            network_data = result.model_dump()
            interpretation = result.interpretation
        except Exception as e:
            print(f"[Agent B - Network] 구조화 파싱 실패: {e}")
            network_data = {"genes": [], "key_findings": ""}
            interpretation = ""
        print(f"[Agent B - Network] 완료! 유전자 {len(network_data['genes'])}개 추출됨")
        print("-" * 50)
        return {
            "network_data": network_data,
            "interpretations": [f"[Network] {interpretation}"],
        }

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
