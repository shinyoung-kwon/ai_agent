"""Entrypoint — accept user query, run the pipeline, print results."""

import asyncio

from app.graph.builder import build_graph


async def run():
    """Build graph, get user input, run pipeline, print results."""
    print("=== AI Biomarker Discovery Agent ===")
    print("파이프라인을 초기화 중...")

    graph = await build_graph()

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
    }

    result = await graph.ainvoke(initial_state)

    print("\n=== 파이프라인 완료 ===")
    print(f"\n[Discovery] 후보 유전자: {result.get('candidates', [])}")
    print(f"\n[Network] 네트워크 데이터: {result.get('network_data', {})}")
    print(f"\n[Reasoning] 분석:\n{result.get('reasoning', '')}")
    print(f"\n[Validation] 검증 결과: {result.get('validation_results', [])}")

    # Cleanup MCP sessions
    for session in getattr(graph, "_mcp_sessions", []):
        if session:
            await session.__aexit__(None, None, None)
    for ctx in getattr(graph, "_mcp_contexts", []):
        if ctx:
            await ctx.__aexit__(None, None, None)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
