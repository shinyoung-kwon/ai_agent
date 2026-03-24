"""
Discovery MCP Server (Agent A) — LangGraph 연결 확인용
유전자 발현 데이터 조회를 모사한다. 외부 API 호출 없음.

실행: python mcp-servers/discovery_server.py
"""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("discovery-server")

# Mock gene expression dataset
MOCK_EXPRESSION = {
    "BRCA1": {"log2fc": 2.3, "pvalue": 0.001, "adj_pvalue": 0.005},
    "TP53": {"log2fc": -1.8, "pvalue": 0.0003, "adj_pvalue": 0.002},
    "EGFR": {"log2fc": 3.1, "pvalue": 0.00005, "adj_pvalue": 0.0004},
    "MYC": {"log2fc": 1.5, "pvalue": 0.02, "adj_pvalue": 0.08},
    "KRAS": {"log2fc": -2.7, "pvalue": 0.0001, "adj_pvalue": 0.001},
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
            name="get_expression",
            description="유전자 이름으로 발현 데이터(log2FC, p-value)를 조회한다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "유전자 심볼 (예: BRCA1, TP53)",
                    }
                },
                "required": ["gene"],
            },
        ),
        types.Tool(
            name="get_deg_list",
            description="DEG(차등발현유전자) 목록을 반환한다. adj_pvalue 기준으로 필터링.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pvalue_cutoff": {
                        "type": "number",
                        "description": "adj_pvalue 임계값 (기본값: 0.05)",
                        "default": 0.05,
                    }
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "ping":
        return [types.TextContent(type="text", text="pong from discovery-server")]

    if name == "get_expression":
        gene = arguments.get("gene", "").upper()
        data = MOCK_EXPRESSION.get(gene)
        if data is None:
            return [types.TextContent(type="text", text=f"Gene '{gene}' not found in dataset")]
        result = {"gene": gene, **data}
        return [types.TextContent(type="text", text=json.dumps(result))]

    if name == "get_deg_list":
        cutoff = arguments.get("pvalue_cutoff", 0.05)
        degs = [
            {"gene": gene, **data}
            for gene, data in MOCK_EXPRESSION.items()
            if data["adj_pvalue"] < cutoff
        ]
        return [types.TextContent(type="text", text=json.dumps(degs))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
