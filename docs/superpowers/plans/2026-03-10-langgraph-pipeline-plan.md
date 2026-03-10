# LangGraph 바이오마커 파이프라인 구현 계획서

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 4단계 순차 파이프라인(A→B→C→D)을 LangGraph 서브그래프로 구현하고, MCP 서버 연결이 정상 동작하는지 검증한다.

**Architecture:** 각 에이전트가 독립 서브그래프(tool_call 루프)를 가지며, 메인 그래프가 순차 연결. MCP 도구는 LangChain Tool로 변환하여 LLM에 바인딩.

**Tech Stack:** Python, LangGraph, LangChain, ChatAnthropic, MCP (stdio), PyYAML, python-dotenv

---

## Chunk 1: 기반 설정

### Task 1: 의존성 업데이트 (pyproject.toml)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: pyproject.toml에 필요한 패키지 추가**

```toml
[project]
name = "ai-agent"
version = "0.1.0"
description = "LangGraph 기반 멀티 에이전트 바이오마커 발견 시스템"
requires-python = ">=3.11"
dependencies = [
    # LangGraph / LLM
    "langgraph",
    "langchain",
    "langchain-anthropic",
    "pydantic",
    # MCP servers
    "mcp",
    "httpx",
    # Utils
    "python-dotenv",
    "pyyaml",
    # RAG / Vector search (later)
    "faiss-cpu",
]

[project.optional-dependencies]
dev = [
    "ruff",
    "pytest",
    "pytest-asyncio",
]

[tool.setuptools.packages.find]
where = ["apps/agent"]
```

- [ ] **Step 2: 패키지 설치 확인**

Run: `pip install -e ".[dev]"`
Expected: 정상 설치 완료

- [ ] **Step 3: 커밋**

```bash
git add pyproject.toml
git commit -m "chore: langchain-anthropic, pytest 의존성 추가"
```

---

### Task 2: 설정 파일 작성 (configs/)

**Files:**
- Modify: `configs/dev.yaml`
- Modify: `configs/prompts.yaml`

- [ ] **Step 1: configs/dev.yaml 작성**

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  temperature: 0

mcp_servers:
  discovery: "python mcp-servers/test_discovery_server.py"
  network: "python mcp-servers/test_network_server.py"
  validation: "python mcp-servers/test_validation_server.py"

langsmith:
  project: ai-agent-biomarker
```

- [ ] **Step 2: configs/prompts.yaml 작성**

```yaml
discovery_agent:
  system: |
    당신은 유전자 발현 데이터를 분석하는 Discovery Agent입니다.
    주어진 질환/주제에 대해 차등발현유전자(DEG)를 조회하고,
    바이오마커 후보 유전자 목록을 도출하세요.
    도구를 사용하여 발현 데이터를 조회하고 분석하세요.
    최종 결과는 유전자 심볼 목록으로 반환하세요.

network_agent:
  system: |
    당신은 유전자 네트워크를 분석하는 Network Agent입니다.
    이전 단계에서 발견된 후보 유전자들의 레귤론 정보와
    TF-Target 네트워크를 조회하여 조절 관계를 파악하세요.
    도구를 사용하여 네트워크 데이터를 조회하세요.

reasoning_agent:
  system: |
    당신은 바이오마커 후보를 평가하는 Reasoning Agent입니다.
    이전 단계의 후보 유전자와 네트워크 데이터를 바탕으로
    생물학적 타당성을 분석하고 후보군의 우선순위를 매기세요.
    현재 외부 도구 없이 분석을 수행하세요.

validation_agent:
  system: |
    당신은 인실리코 검증을 수행하는 Validation Agent입니다.
    추천된 후보 유전자들에 대해 시뮬레이션을 실행하고,
    최종 마스터 레귤레이터를 확정하세요.
    도구를 사용하여 시뮬레이션을 실행하세요.
```

- [ ] **Step 3: 커밋**

```bash
git add configs/dev.yaml configs/prompts.yaml
git commit -m "feat: dev.yaml, prompts.yaml 설정 파일 작성"
```

---

### Task 3: 설정 로딩 유틸리티 (utils/config.py)

**Files:**
- Modify: `apps/agent/app/utils/config.py`

- [ ] **Step 1: config.py 구현**

```python
"""Load configuration from configs/dev.yaml and .env."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_PROJECT_ROOT / ".env")


def load_yaml(filename: str) -> dict:
    """Load a YAML file from the configs/ directory."""
    path = _PROJECT_ROOT / "configs" / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config() -> dict:
    """Return the dev.yaml configuration."""
    return load_yaml("dev.yaml")


def get_prompts() -> dict:
    """Return the prompts.yaml configuration."""
    return load_yaml("prompts.yaml")


def get_llm():
    """Create an LLM instance based on dev.yaml config."""
    config = get_config()
    llm_config = config["llm"]
    provider = llm_config.get("provider", "anthropic")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=llm_config["model"],
            temperature=llm_config.get("temperature", 0),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
```

- [ ] **Step 2: 간단한 import 확인**

Run: `python -c "from app.utils.config import get_config, get_prompts, get_llm; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 커밋**

```bash
git add apps/agent/app/utils/config.py
git commit -m "feat: config.py 설정 로딩 및 LLM 팩토리 구현"
```

---

## Chunk 2: MCP 도구 연결

### Task 4: 에이전트별 도구 프로필 (tools/profiles/)

**Files:**
- Create: `apps/agent/app/tools/profiles/discovery.yaml`
- Create: `apps/agent/app/tools/profiles/network.yaml`
- Create: `apps/agent/app/tools/profiles/reasoning.yaml`
- Create: `apps/agent/app/tools/profiles/validation.yaml`

- [ ] **Step 1: 4개 프로필 YAML 생성**

`discovery.yaml`:
```yaml
server: discovery
tools:
  - get_expression
  - get_deg_list
```

`network.yaml`:
```yaml
server: network
tools:
  - get_regulon
  - get_tf_target_network
```

`reasoning.yaml`:
```yaml
server: null
tools: []
```

`validation.yaml`:
```yaml
server: validation
tools:
  - run_simulation
  - batch_simulation
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/tools/profiles/
git commit -m "feat: 에이전트별 도구 프로필 YAML 추가"
```

---

### Task 5: MCP 레지스트리 (tools/registry.py)

**Files:**
- Modify: `apps/agent/app/tools/registry.py`

- [ ] **Step 1: registry.py 구현 — MCP 클라이언트 연결 + LangChain Tool 변환**

```python
"""Connect to MCP servers and provide LangChain-compatible tools per agent."""

import json
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
    """Parse a command string like 'python mcp-servers/foo.py' into StdioServerParameters."""
    parts = command_str.split()
    return StdioServerParameters(command=parts[0], args=parts[1:])


def _mcp_tool_to_langchain(session: ClientSession, mcp_tool) -> StructuredTool:
    """Wrap a single MCP tool as a LangChain StructuredTool."""
    tool_name = mcp_tool.name
    tool_desc = mcp_tool.description or tool_name
    input_schema = mcp_tool.inputSchema

    async def _invoke(**kwargs):
        result = await session.call_tool(tool_name, kwargs)
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts)

    # Build args_schema from inputSchema properties
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    field_definitions = {}
    for prop_name, prop_info in properties.items():
        field_definitions[prop_name] = (
            str if prop_info.get("type") == "string" else float if prop_info.get("type") == "number" else object,
            prop_name if prop_name in required else None,
        )

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=tool_name,
        description=tool_desc,
        # Let LangChain infer schema from the function signature + inputSchema
    )


async def get_tools_for_agent(agent_name: str) -> tuple[list, ClientSession | None, object | None]:
    """
    Connect to the MCP server for the given agent and return LangChain tools.

    Returns:
        (tools, session, client_context) - caller must keep client_context alive.
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
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/tools/registry.py
git commit -m "feat: MCP 레지스트리 — 서버 연결 및 LangChain Tool 변환"
```

---

## Chunk 3: 에이전트 서브그래프

### Task 6: 노드 패키지 초기화 + 공통 서브그래프 팩토리

**Files:**
- Create: `apps/agent/app/graph/nodes/__init__.py`

- [ ] **Step 1: __init__.py 생성 (빈 파일)**

```python
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/graph/nodes/__init__.py
git commit -m "chore: graph/nodes 패키지 초기화"
```

---

### Task 7: Agent A — Discovery 서브그래프

**Files:**
- Create: `apps/agent/app/graph/nodes/discovery.py`

- [ ] **Step 1: discovery.py 구현**

```python
"""Agent A — Discovery subgraph with tool_call loop."""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts


def build_discovery_subgraph(tools: list) -> StateGraph:
    """Build Agent A subgraph that discovers biomarker candidates."""
    llm = get_llm()
    prompts = get_prompts()
    system_prompt = prompts["discovery_agent"]["system"]

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    tools_by_name = {t.name: t for t in tools}

    async def llm_call(state: AgentState):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"질환/주제: {state['query']}\n"
                "차등발현유전자(DEG)를 조회하여 바이오마커 후보 목록을 도출하세요."
            )),
        ]
        messages.extend(state.get("messages", []))
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(state: AgentState):
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = await tool.ainvoke(tool_call["args"])
            from langchain_core.messages import ToolMessage
            results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        return {"messages": results}

    def should_continue(state: AgentState) -> Literal["tool_node", "extract"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return "extract"

    async def extract_candidates(state: AgentState):
        """Extract candidate gene list from the final LLM response."""
        last_message = state["messages"][-1]
        # LLM's final text response contains the candidate list
        return {"candidates": _parse_gene_list(last_message.content)}

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("extract", extract_candidates)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "extract": "extract",
    })
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("extract", END)

    return graph.compile()


def _parse_gene_list(text: str) -> list[str]:
    """Parse gene symbols from LLM text output."""
    import re
    # Match common gene symbol patterns (uppercase letters + digits)
    genes = re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', text)
    # Filter common English words that look like gene symbols
    stopwords = {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "DEG", "TF", "MCP"}
    return list(dict.fromkeys(g for g in genes if g not in stopwords))
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/graph/nodes/discovery.py
git commit -m "feat: Agent A Discovery 서브그래프 구현"
```

---

### Task 8: Agent B — Network 서브그래프

**Files:**
- Create: `apps/agent/app/graph/nodes/network.py`

- [ ] **Step 1: network.py 구현**

```python
"""Agent B — Network subgraph with tool_call loop."""

import json
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts


def build_network_subgraph(tools: list) -> StateGraph:
    """Build Agent B subgraph that analyzes gene regulatory networks."""
    llm = get_llm()
    prompts = get_prompts()
    system_prompt = prompts["network_agent"]["system"]

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    tools_by_name = {t.name: t for t in tools}

    async def llm_call(state: AgentState):
        candidates = state.get("candidates", [])
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"후보 유전자 목록: {', '.join(candidates)}\n"
                "이 유전자들의 레귤론과 TF-Target 네트워크를 조회하세요."
            )),
        ]
        messages.extend(state.get("messages", []))
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(state: AgentState):
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = await tool.ainvoke(tool_call["args"])
            from langchain_core.messages import ToolMessage
            results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        return {"messages": results}

    def should_continue(state: AgentState) -> Literal["tool_node", "extract"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return "extract"

    async def extract_network(state: AgentState):
        """Extract network data from the final LLM response."""
        last_message = state["messages"][-1]
        return {"network_data": {"raw_response": last_message.content}}

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("extract", extract_network)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "extract": "extract",
    })
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("extract", END)

    return graph.compile()
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/graph/nodes/network.py
git commit -m "feat: Agent B Network 서브그래프 구현"
```

---

### Task 9: Agent C — Reasoning 서브그래프 (mock)

**Files:**
- Create: `apps/agent/app/graph/nodes/reasoning.py`

- [ ] **Step 1: reasoning.py 구현 (도구 없이 LLM만 호출)**

```python
"""Agent C — Reasoning subgraph (mock, no tools). RAG will be added later."""

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts


def build_reasoning_subgraph() -> StateGraph:
    """Build Agent C subgraph that evaluates candidates (mock, no RAG)."""
    llm = get_llm()
    prompts = get_prompts()
    system_prompt = prompts["reasoning_agent"]["system"]

    async def llm_call(state: AgentState):
        candidates = state.get("candidates", [])
        network_data = state.get("network_data", {})
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"질환/주제: {state['query']}\n"
                f"후보 유전자: {', '.join(candidates)}\n"
                f"네트워크 데이터: {network_data}\n\n"
                "위 정보를 바탕으로 후보군의 생물학적 타당성을 분석하고 "
                "우선순위를 매겨주세요."
            )),
        ]
        response = await llm.ainvoke(messages)
        return {
            "messages": [response],
            "reasoning": response.content,
        }

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)

    graph.add_edge(START, "llm_call")
    graph.add_edge("llm_call", END)

    return graph.compile()
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/graph/nodes/reasoning.py
git commit -m "feat: Agent C Reasoning 서브그래프 구현 (mock)"
```

---

### Task 10: Agent D — Validation 서브그래프

**Files:**
- Create: `apps/agent/app/graph/nodes/validation.py`

- [ ] **Step 1: validation.py 구현**

```python
"""Agent D — Validation subgraph with tool_call loop."""

import json
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.utils.config import get_llm, get_prompts


def build_validation_subgraph(tools: list) -> StateGraph:
    """Build Agent D subgraph that runs in-silico validation."""
    llm = get_llm()
    prompts = get_prompts()
    system_prompt = prompts["validation_agent"]["system"]

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    tools_by_name = {t.name: t for t in tools}

    async def llm_call(state: AgentState):
        candidates = state.get("candidates", [])
        reasoning = state.get("reasoning", "")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"후보 유전자: {', '.join(candidates)}\n"
                f"분석 결과: {reasoning}\n\n"
                "위 후보들에 대해 시뮬레이션을 실행하고 "
                "최종 마스터 레귤레이터를 확정하세요."
            )),
        ]
        messages.extend(state.get("messages", []))
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(state: AgentState):
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = await tool.ainvoke(tool_call["args"])
            from langchain_core.messages import ToolMessage
            results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        return {"messages": results}

    def should_continue(state: AgentState) -> Literal["tool_node", "extract"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return "extract"

    async def extract_validation(state: AgentState):
        """Extract validation results from the final LLM response."""
        last_message = state["messages"][-1]
        return {"validation_results": [{"raw_response": last_message.content}]}

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_node("extract", extract_validation)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "extract": "extract",
    })
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("extract", END)

    return graph.compile()
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/graph/nodes/validation.py
git commit -m "feat: Agent D Validation 서브그래프 구현"
```

---

## Chunk 4: 메인 그래프 + 엔트리포인트

### Task 11: 메인 그래프 (builder.py)

**Files:**
- Modify: `apps/agent/app/graph/builder.py`

- [ ] **Step 1: builder.py 구현 — 4개 서브그래프를 순차 연결**

```python
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
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/graph/builder.py
git commit -m "feat: 메인 그래프 빌더 — 4개 서브그래프 순차 연결"
```

---

### Task 12: 엔트리포인트 (main.py)

**Files:**
- Modify: `apps/agent/app/main.py`

- [ ] **Step 1: main.py 구현 — 사용자 입력 → 그래프 실행 → 결과 출력**

```python
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
```

- [ ] **Step 2: 커밋**

```bash
git add apps/agent/app/main.py
git commit -m "feat: main.py 엔트리포인트 — CLI 입력 및 파이프라인 실행"
```

---

### Task 13: 통합 실행 테스트

- [ ] **Step 1: .env 파일에 API 키 확인**

`.env`에 다음이 설정되어 있는지 확인:
```
ANTHROPIC_API_KEY=<your-key>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-key>
LANGCHAIN_PROJECT=ai-agent-biomarker
```

- [ ] **Step 2: 파이프라인 실행**

Run: `cd apps/agent && python -m app.main`
Input: `폐암`
Expected: 4개 에이전트가 순차 실행되고 각 단계 결과가 출력됨

- [ ] **Step 3: LangSmith 대시보드에서 트레이스 확인**

브라우저에서 LangSmith 대시보드 → ai-agent-biomarker 프로젝트 → 트레이스가 기록되었는지 확인

- [ ] **Step 4: 결과 확인 후 최종 커밋**

모든 것이 정상 동작하면 남은 변경사항 커밋.
