"""Connect to MCP servers and provide LangChain-compatible tools per agent."""

from pathlib import Path

import yaml
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.utils.config import get_config

_PROFILES_DIR = Path(__file__).parent / "profiles"


def _load_profile(agent_name: str) -> dict:
    """Load a tool profile YAML for the given agent."""
    path = _PROFILES_DIR / f"{agent_name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_server_command(command_str: str) -> StdioServerParameters:
    """Parse a command string into StdioServerParameters."""
    parts = command_str.split()
    return StdioServerParameters(command=parts[0], args=parts[1:])


def _mcp_tool_to_langchain(session: ClientSession, mcp_tool) -> StructuredTool:
    """Wrap a single MCP tool as a LangChain StructuredTool."""
    tool_name = mcp_tool.name
    tool_desc = mcp_tool.description or tool_name

    async def _invoke(**kwargs):
        result = await session.call_tool(tool_name, kwargs)
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=tool_name,
        description=tool_desc,
    )


async def get_tools_for_agent(agent_name: str) -> tuple[list, ClientSession | None, object | None]:
    """
    Connect to the MCP server for the given agent and return LangChain tools.

    Returns:
        (tools, session, client_context)
        For agents with no server (e.g. reasoning), returns ([], None, None).
    """
    profile = _load_profile(agent_name)

    if profile.get("server") is None:
        return [], None, None

    config = get_config()
    server_name = profile["server"]
    command_str = config["mcp_servers"][server_name]
    server_params = _parse_server_command(command_str)
    allowed_tools = set(profile.get("tools", []))

    # Connect to MCP server
    client_ctx = stdio_client(server_params)
    streams = await client_ctx.__aenter__()
    session = ClientSession(*streams)
    await session.__aenter__()
    await session.initialize()

    # List tools and filter by profile
    response = await session.list_tools()
    tools = []
    for mcp_tool in response.tools:
        if mcp_tool.name in allowed_tools:
            tools.append(_mcp_tool_to_langchain(session, mcp_tool))

    return tools, session, client_ctx
