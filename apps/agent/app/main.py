"""Entrypoint — accept user query, run the pipeline, print results."""

import asyncio
import sys
import os

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
        "reasoning": {},
        "validation_results": {},
        "interpretations": [],
    }

    result = await graph.ainvoke(
        initial_state,
        {"recursion_limit": 50},
    )

    print("\n=== 파이프라인 완료 ===")

    interpretations = result.get("interpretations", [])

    candidates = result.get("candidates", [])
    print(f"\n[Discovery] \n후보 유전자: {', '.join(candidates)}")
    if len(interpretations) > 0:
        print(f" \n해석: {interpretations[0].removeprefix('[Discovery] ')}")

    network = result.get("network_data", {})
    print(f"\n[Network] \n네트워크 유전자: {', '.join(network.get('genes', []))}")
    if len(interpretations) > 1:
        print(f" \n해석: {interpretations[1].removeprefix('[Network] ')}")
    print(f" \n주요 발견: {network.get('key_findings', '')}")

    # reasoning = result.get("reasoning", {})
    # print("\n[Reasoning] \n우선순위:")
    # for r in reasoning.get("rankings", []):
    #     print(f"  {r.get('rank', '?')}. {r.get('gene', '?')} - {r.get('rationale', '')}")
    # if len(interpretations) > 2:
    #     print(f" \n해석: {interpretations[2].removeprefix('[Reasoning] ')}")
    # print(f" \n권장사항: {reasoning.get('recommendation', '')}")

    validation = result.get("validation_results", {})
    confirmed = validation.get("confirmed_biomarkers", [])
    print(f"\n[Validation] \n선정 바이오마커: {', '.join(confirmed)}")
    if len(interpretations) > 3:
        print(f" \n해석: {interpretations[3].removeprefix('[Validation] ')}")
    print(f" \n요약: {validation.get('summary', '')}")

    # Cleanup MCP connections (suppress anyio cancel scope errors on exit)
    stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    for stack in mcp_stacks:
        if stack:
            try:
                await stack.aclose()
            except BaseException:
                pass
    sys.stderr = stderr


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
