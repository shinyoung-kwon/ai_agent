"""
Discovery MCP Server 2 (CellNeighborEX) — LangGraph 연결 확인용
CellNeighborEX 기반 CCI 유전자 식별을 모사한다. 외부 API 호출 없음.

실행: python mcp-servers/test_discovery_server2.py
"""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("cellneighborex-server")

# Mock CCI genes dataset
MOCK_CCI_RESULTS = {
    "Tumor_Immune": {
        "cci_genes": ["EGFR", "CD274", "PDCD1LG2", "CXCL9", "CXCL10"],
        "cell_pair": {"source": "Tumor", "neighbor": "Immune"},
        "n_genes": 5,
    },
    "Tumor_Fibroblast": {
        "cci_genes": ["TGFB1", "FAP", "COL1A1", "MMP2", "BRCA1"],
        "cell_pair": {"source": "Tumor", "neighbor": "Fibroblast"},
        "n_genes": 5,
    },
    "Immune_Stromal": {
        "cci_genes": ["IL6", "STAT3", "TP53", "TNF", "IFNG"],
        "cell_pair": {"source": "Immune", "neighbor": "Stromal"},
        "n_genes": 5,
    },
}


@app.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="ccigenes",
            description="CellNeighborEX 기반 CCI 유전자 목록을 반환한다. 세포 유형 쌍별 상호작용 유전자를 식별한다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cell_pair": {
                        "type": "string",
                        "description": "세포 유형 쌍 (예: Tumor_Immune, Tumor_Fibroblast). 미지정 시 전체 반환.",
                    }
                },
            },
        ),
        types.Tool(
            name="ccipairs",
            description="분석된 세포 유형 상호작용 쌍 목록을 반환한다.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "ccigenes":
        cell_pair = arguments.get("cell_pair")
        if cell_pair:
            data = MOCK_CCI_RESULTS.get(cell_pair)
            if data is None:
                return [types.TextContent(
                    type="text",
                    text=f"Cell pair '{cell_pair}' not found. Available: {list(MOCK_CCI_RESULTS.keys())}",
                )]
            return [types.TextContent(type="text", text=json.dumps(data))]
        # Return all CCI genes
        all_results = list(MOCK_CCI_RESULTS.values())
        return [types.TextContent(type="text", text=json.dumps(all_results))]

    if name == "ccipairs":
        pairs = [
            {"pair": key, **val["cell_pair"]}
            for key, val in MOCK_CCI_RESULTS.items()
        ]
        return [types.TextContent(type="text", text=json.dumps(pairs))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
