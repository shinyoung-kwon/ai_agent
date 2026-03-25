"""Connect to MCP servers and provide LangChain-compatible tools per agent."""

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, create_model

from app.utils.config import get_config, get_project_root

_PROFILES_DIR = Path(__file__).parent / "profiles"


def _load_profile(agent_name: str) -> dict:
    """Load a tool profile YAML for the given agent."""
    path = _PROFILES_DIR / f"{agent_name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_server_command(command_str: str) -> StdioServerParameters:
    """Parse a command string into StdioServerParameters."""
    parts = command_str.split()
    return StdioServerParameters(
        command=parts[0],
        args=parts[1:],
        cwd=str(get_project_root()),
    )


_JSON_TYPE_MAP = {
    "string": (str, ...),
    "integer": (int, ...),
    "number": (float, ...),
    "boolean": (bool, ...),
    "array": (list, ...),
    "object": (dict, ...),
}


def _build_args_schema(tool_name: str, input_schema: dict) -> type[BaseModel]:
    """Build a Pydantic model from MCP tool's inputSchema."""
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    fields: dict[str, Any] = {}
    for field_name, field_info in properties.items():
        json_type = field_info.get("type", "string")
        py_type = _JSON_TYPE_MAP.get(json_type, (str, ...))[0]
        default = ... if field_name in required else None
        fields[field_name] = (py_type, default)
    return create_model(tool_name, **fields)


def _mcp_tool_to_langchain(session: ClientSession, mcp_tool) -> StructuredTool:
    """Wrap a single MCP tool as a LangChain StructuredTool."""
    tool_name = mcp_tool.name
    tool_desc = mcp_tool.description or tool_name
    schema = _build_args_schema(tool_name, mcp_tool.inputSchema or {})

    async def _invoke(**kwargs):
        result = await session.call_tool(tool_name, kwargs)
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=tool_name,
        description=tool_desc,
        args_schema=schema,
    )


async def get_tools_for_agent(agent_name: str) -> tuple[list, AsyncExitStack | None]:
    """
    Connect to the MCP server for the given agent and return LangChain tools.

    Returns:
        (tools, stack)
        For agents with no server (e.g. reasoning), returns ([], None).
        Caller must call `await stack.aclose()` when done.
    """
    profile = _load_profile(agent_name)

    if profile.get("server") is None:
        return [], None

    config = get_config()
    server_name = profile["server"]
    command_str = config["mcp_servers"][server_name]
    server_params = _parse_server_command(command_str)
    allowed_tools = set(profile.get("tools", []))

    # Connect to MCP server using AsyncExitStack
    stack = AsyncExitStack()
    streams = await stack.enter_async_context(stdio_client(server_params))
    session = await stack.enter_async_context(ClientSession(*streams))
    await session.initialize()

    # List tools and filter by profile
    response = await session.list_tools()
    tools = []
    for mcp_tool in response.tools:
        if mcp_tool.name in allowed_tools:
            tools.append(_mcp_tool_to_langchain(session, mcp_tool))

    return tools, stack


