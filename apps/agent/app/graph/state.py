'''
LangGraph에서 state 공유, 에이전트끼리 주고받는 데이터 구조를 정의

TypedDict로 데이터 구조 설정 — LangGraph 표준 패턴
messages 필드 — Annotated[..., operator.add]로 메시지가 누적되도록 reducer 적용
4단계 파이프라인 반영 — candidates(A), network_data(B), reasoning(C), validation_results(D) 각 에이전트 출력 필드 분리
'''

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State shared across all agents in the pipeline."""

    # User input query (e.g. disease name or research topic)
    query: str

    # Chat message history (accumulates via operator.add)
    messages: Annotated[list[BaseMessage], operator.add]

    # Agent A (Discovery): candidate biomarkers from literature
    candidates: list[str]

    # Agent B (Network): interaction network / pathway data
    network_data: dict

    # Agent C (Reasoning): RAG-based analysis and ranking
    reasoning: str

    # Agent D (Validation): final validated results
    validation_results: list[dict]
