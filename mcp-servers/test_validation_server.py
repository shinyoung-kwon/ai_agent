"""
Validation MCP Server (Agent D) — LangGraph 연결 확인용
인실리코 시뮬레이션 결과를 모사한다. 외부 API 호출 없음.

실행: python mcp-servers/validation_server.py
"""

import asyncio
import json
import random

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("validation-server")

# Mock simulation results (deterministic seed for reproducibility)
random.seed(42)


def _mock_simulation(gene: str) -> dict:
    """Generate deterministic mock simulation result for a gene."""
    seed_value = sum(ord(c) for c in gene)
    rng = random.Random(seed_value)
    score = round(rng.uniform(0.3, 0.95), 3)
    return {
        "gene": gene,
        "simulation_score": score,
        "is_master_regulator": score >= 0.7,
        "confidence": "high" if score >= 0.8 else "medium" if score >= 0.6 else "low",
    }


@app.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="ping",
            description="서버 연결 상태를 확인한다.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="run_simulation",
            description="유전자에 대한 인실리코 시뮬레이션을 실행한다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "시뮬레이션할 유전자 심볼",
                    }
                },
                "required": ["gene"],
            },
        ),
        types.Tool(
            name="batch_simulation",
            description="여러 유전자에 대해 일괄 시뮬레이션을 실행하고 결과를 점수 순으로 반환한다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "시뮬레이션할 유전자 목록",
                    }
                },
                "required": ["genes"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "ping":
        return [types.TextContent(type="text", text="pong from validation-server")]

    if name == "run_simulation":
        gene = arguments.get("gene", "").upper()
        result = _mock_simulation(gene)
        return [types.TextContent(type="text", text=json.dumps(result))]

    if name == "batch_simulation":
        genes = [g.upper() for g in arguments.get("genes", [])]
        results = [_mock_simulation(g) for g in genes]
        results.sort(key=lambda x: x["simulation_score"], reverse=True)
        return [types.TextContent(type="text", text=json.dumps(results))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
