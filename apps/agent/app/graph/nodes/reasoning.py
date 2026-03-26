"""Agent C — Reasoning subgraph (mock, no tools). RAG will be added later."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts
from app.utils.messages import get_tool_loop_messages
from app.graph.schemas import ReasoningOutput


def build_reasoning_subgraph() -> StateGraph:
    """Build Agent C subgraph that evaluates candidates (mock, no RAG)."""
    llm = get_llm()
    prompts = get_prompts()
    system_prompt = prompts["reasoning_agent"]["system"]

    async def llm_call(state: AgentState):
        candidates = state.get("candidates", [])
        network_data = state.get("network_data", {})
        print(f"\n[Agent C - Reasoning] LLM 호출 중... (후보: {candidates})")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"질환/주제: {state['query']}\n"
                f"후보 유전자: {', '.join(candidates)}\n"
                f"네트워크 데이터: {network_data}\n\n"
                "위 정보를 바탕으로 후보군의 생물학적 타당성을 분석하고 "
                "우선순위를 매겨주세요."
            )),
        ]
        messages.extend(get_tool_loop_messages(state.get("messages", [])))
        structured_llm = llm.with_structured_output(ReasoningOutput)
        try:
            result = await structured_llm.ainvoke(messages)
            reasoning = result.model_dump()
            interpretation = result.interpretation
            summary = ", ".join(
                f"{r.rank}. {r.gene}" for r in result.rankings
            )
        except Exception as e:
            print(f"[Agent C - Reasoning] 구조화 파싱 실패: {e}")
            reasoning = {"rankings": [], "recommendation": ""}
            interpretation = ""
            summary = "파싱 실패"
        print("[Agent C - Reasoning] 완료! 생물학적 타당성 분석 완료")
        print("-" * 50)
        return {
            "messages": [AIMessage(content=summary)],
            "reasoning": reasoning,
            "interpretations": [f"[Reasoning] {interpretation}"],
        }

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)

    graph.add_edge(START, "llm_call")
    graph.add_edge("llm_call", END)

    return graph.compile()
