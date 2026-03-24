"""
Network MCP Server (Agent B) — LangGraph 연결 확인용
레귤론 분석 및 TF-Target 네트워크 조회를 모사한다. 외부 API 호출 없음.

실행: python mcp-servers/network_server.py
"""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("network-server")

# Mock TF-Target network
MOCK_NETWORK = {
    "TP53": {
        "targets": ["CDKN1A", "BAX", "MDM2", "GADD45A"],
        "regulon_size": 4,
        "regulon_score": 0.85,
    },
    "MYC": {
        "targets": ["CDK4", "CCND1", "LDHA", "PKM"],
        "regulon_size": 4,
        "regulon_score": 0.72,
    },
    "STAT3": {
        "targets": ["BCL2", "VEGFA", "MMP9", "IL6"],
        "regulon_size": 4,
        "regulon_score": 0.68,
    },
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
            name="get_regulon",
            description="전사인자(TF)의 레귤론 정보를 조회한다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tf": {
                        "type": "string",
                        "description": "전사인자 심볼 (예: TP53, MYC)",
                    }
                },
                "required": ["tf"],
            },
        ),
        types.Tool(
            name="get_tf_target_network",
            description="전체 TF-Target 네트워크를 반환한다.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "ping":
        return [types.TextContent(type="text", text="pong from network-server")]

    if name == "get_regulon":
        tf = arguments.get("tf", "").upper()
        data = MOCK_NETWORK.get(tf)
        if data is None:
            return [types.TextContent(type="text", text=f"TF '{tf}' not found in network")]
        result = {"tf": tf, **data}
        return [types.TextContent(type="text", text=json.dumps(result))]

    if name == "get_tf_target_network":
        network = [
            {"tf": tf, **data} for tf, data in MOCK_NETWORK.items()
        ]
        return [types.TextContent(type="text", text=json.dumps(network))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
