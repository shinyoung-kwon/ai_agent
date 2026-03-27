"""Agent D — Validation subgraph with tool_call loop."""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts
from app.utils.messages import get_tool_loop_messages
from app.graph.schemas import ValidationOutput


def build_validation_subgraph(tools: list) -> StateGraph:
    """Build Agent D subgraph that runs in-silico validation."""
    llm = get_llm()
    prompts = get_prompts()
    system_prompt = prompts["validation_agent"]["system"]

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    tools_by_name = {t.name: t for t in tools}

    async def llm_call(state: AgentState):
        candidates = state.get("candidates", [])
        reasoning = state.get("reasoning", "")
        print(f"\n[Validation Agent] LLM 호출 중... (후보: {candidates})")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"후보 유전자: {', '.join(candidates)}\n"
                f"분석 결과: {reasoning}\n\n"
                "위 후보들에 대해 시뮬레이션을 실행하고, "
                "최종 마스터 레귤레이터를 확정하며, "
                "사용한 도구의 결과를 해석해주세요."
            )),
        ]
        messages.extend(get_tool_loop_messages(state.get("messages", [])))
        response = await llm_with_tools.ainvoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            print(f"[Validation Agent] 도구 호출 요청: {tool_names}")
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
            print(f"[Validation Agent] 도구 실행: {tool_call['name']}({tool_call['args']})")
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

    async def extract_validation(state: AgentState):
        """Extract validation results via structured output."""
        last_message = state["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        structured_llm = llm.with_structured_output(ValidationOutput)
        try:
            result = await structured_llm.ainvoke([
                SystemMessage(content=(
                    "You are the Validation agent. Extract confirmed biomarkers and provide "
                    "interpretation of the tool results used during analysis."
                )),
                HumanMessage(content=content)
            ])
            validation_results = result.model_dump()
            interpretation = result.interpretation
        except Exception as e:
            print(f"[Validation Agent] 구조화 파싱 실패: {e}")
            validation_results = {"confirmed_biomarkers": [], "summary": ""}
            interpretation = ""
        print(f"[Validation Agent] 완료! 확정 바이오마커 {len(validation_results['confirmed_biomarkers'])}개 추출됨")
        print("-" * 50)
        return {
            "validation_results": validation_results,
            "interpretations": [f"[Validation] {interpretation}"],
        }

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("extract", extract_validation)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "extract": "extract",
    })
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("extract", END)

    return graph.compile()
