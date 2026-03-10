"""Main graph builder — assembles 4 agent subgraphs into a sequential pipeline."""

from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.graph.nodes.discovery import build_discovery_subgraph
from app.graph.nodes.network import build_network_subgraph
from app.graph.nodes.reasoning import build_reasoning_subgraph
from app.graph.nodes.validation import build_validation_subgraph
from app.tools.registry import get_tools_for_agent


async def build_graph():
    """Build and compile the main pipeline graph."""
    # Load tools for each agent from MCP servers
    discovery_tools, disc_session, disc_ctx = await get_tools_for_agent("discovery")
    network_tools, net_session, net_ctx = await get_tools_for_agent("network")
    validation_tools, val_session, val_ctx = await get_tools_for_agent("validation")

    # Build subgraphs
    discovery_graph = build_discovery_subgraph(discovery_tools)
    network_graph = build_network_subgraph(network_tools)
    reasoning_graph = build_reasoning_subgraph()
    validation_graph = build_validation_subgraph(validation_tools)

    # Main graph: sequential pipeline
    main_graph = StateGraph(AgentState)
    main_graph.add_node("discovery", discovery_graph)
    main_graph.add_node("network", network_graph)
    main_graph.add_node("reasoning", reasoning_graph)
    main_graph.add_node("validation", validation_graph)

    main_graph.add_edge(START, "discovery")
    main_graph.add_edge("discovery", "network")
    main_graph.add_edge("network", "reasoning")
    main_graph.add_edge("reasoning", "validation")
    main_graph.add_edge("validation", END)

    compiled = main_graph.compile()

    # Store sessions/contexts for cleanup
    compiled._mcp_sessions = [disc_session, net_session, val_session]
    compiled._mcp_contexts = [disc_ctx, net_ctx, val_ctx]

    return compiled
