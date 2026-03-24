"""Entrypoint — accept user query, run the pipeline, print results."""

import asyncio

from app.graph.builder import build_graph
from app.utils.config import setup_langsmith


async def run():
    """Build graph, get user input, run pipeline, print results."""
    setup_langsmith()

    print("\n=== AI Biomarker Discovery Agent ===")
    print("파이프라인을 초기화 중...")

    graph, mcp_stacks = await build_graph()

    print("초기화 완료!\n")

    query = input("질환/주제를 입력하세요: ")
    if not query.strip():
        print("입력이 비어있습니다. 종료합니다.")
        return

    print(f"\n'{query}' 에 대한 바이오마커 탐색을 시작합니다...\n")

    initial_state = {
        "query": query,
        "messages": [],
        "candidates": [],
        "network_data": {},
        "reasoning": "",
        "validation_results": [],
        "stage_summaries": [],
    }

    result = await graph.ainvoke(initial_state)

    print("\n=== 파이프라인 완료 ===")
    print(f"\n[Discovery] 후보 유전자: {result.get('candidates', [])}")
    print(f"\n[Network] 네트워크 데이터: {result.get('network_data', {})}")
    print(f"\n[Reasoning] 분석:\n{result.get('reasoning', '')}")
    print(f"\n[Validation] 검증 결과: {result.get('validation_results', [])}")

    # Cleanup MCP connections (suppress anyio cancel scope errors on exit)
    for stack in mcp_stacks:
        if stack:
            try:
                await stack.aclose()
            except BaseException:
                pass


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
