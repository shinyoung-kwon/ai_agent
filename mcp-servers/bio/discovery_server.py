"""
Bio Discovery MCP Server — PyDESeq2 based real DEG analysis.
MCP tool definitions only; actual logic lives in services/deg_analysis.py.

Run: python mcp-servers/bio/discovery_server.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Allow imports from bio/services/
sys.path.insert(0, str(Path(__file__).parent))
from services.deg_analysis import get_gene_expression, run_deg_analysis

app = Server("bio-discovery-server")


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
            description="DEG(차등발현유전자) 목록을 반환한다. padj 기준으로 필터링.",
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
        return [types.TextContent(type="text", text="pong from bio-discovery-server")]

    if name == "get_expression":
        gene = arguments.get("gene", "").upper()
        data = await asyncio.to_thread(get_gene_expression, gene)
        if data is None:
            return [types.TextContent(type="text", text=f"Gene '{gene}' not found")]
        return [types.TextContent(type="text", text=json.dumps(data))]

    if name == "get_deg_list":
        cutoff = arguments.get("pvalue_cutoff", 0.05)
        degs = await asyncio.to_thread(run_deg_analysis, pvalue_cutoff=cutoff)
        return [types.TextContent(type="text", text=json.dumps(degs))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
