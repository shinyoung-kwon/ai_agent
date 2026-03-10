# AI Agent for Biomarker Discovery

## Overview

This project implements a multi-agent AI system designed to identify potential biomarkers using transcriptomic data and external biological databases.

The system integrates large language models, retrieval pipelines, and specialized tools through MCP servers to support automated evidence-based biomarker discovery.

Core technologies include:

- LangGraph for multi-agent orchestration
- MCP servers for tool execution
- LLM-based reasoning
- RAG-based literature retrieval
- Modular AI pipeline design

---

## Architecture

The system follows a modular multi-agent architecture.

AI agents collaborate through a 4-step automated loop to rapidly identify high-precision targets:

```
Transcriptomic Data
      ↓
Step 1. Discovery Agent — 관심 유전자셋(GoI) 규명
      ↓
Step 2. Network Agent — 레귤론(Regulon) 분석 및 TF-Target 네트워크 구축
      ↓
Step 3. Reasoning Agent — Agentic RAG 기반 후보군 필터링 및 타당성 검증
      ↓
Step 4. Validation Agent — 인실리코(In-silico) 시뮬레이션 및 마스터 레귤레이터 확정
      ↓
Final Biomarker Report
```

Each agent interacts with external tools through MCP servers.

---

## Agent Roles

### Agent A — Discovery Agent (Step 1)
Identifies genes of interest (GoI) from transcriptomic data.
Analyzes gene expression profiles to select initial biomarker candidates.

### Agent B — Network Agent (Step 2)
Performs regulon analysis and constructs TF-Target networks.
Maps regulatory relationships between transcription factors and target genes.

### Agent C — Reasoning Agent (Step 3)
Filters and validates biomarker candidates using Agentic RAG.
Retrieves scientific literature and evaluates biological plausibility.

### Agent D — Validation Agent (Step 4)
Runs in-silico simulation to validate candidates.
Determines final master regulators and produces the biomarker report.

---

## Project Structure

```
ai_agent_pj
│
├── README.md          # project documentation
├── .gitignore         # files and folders ignored by git
├── .env               # environment variables (not committed)
├── pyproject.toml     # project metadata and dependencies
│
├── apps
│   └── agent
│       └── app
│           ├── main.py          # agent execution entrypoint
│           │
│           ├── graph            # LangGraph workflow
│           │   ├── builder.py   # agent workflow builder
│           │   └── state.py     # shared agent state
│           │
│           ├── tools
│           │   ├── registry.py  # MCP tool registry
│           │   └── profiles     # agent-specific tool access
│           │
│           ├── retrieval        # RAG / vector search
│           ├── services         # biomarker discovery logic
│           └── utils            # config, logging, helpers
│
├── configs              # configuration files
│   ├── dev.yaml         # runtime configuration
│   └── prompts.yaml     # prompt templates
│
├── scripts              # helper scripts
│   ├── bootstrap.sh     # environment setup
│   ├── build_index.sh   # vector index creation
│   └── run_local.sh     # run agent locally
│
├── data
│   ├── raw              # raw datasets
│   ├── processed        # processed datasets
│   ├── indices          # vector database indexes
│   └── cache            # temporary cache files
│
└── mcp-servers              # MCP tool servers (test mock)
    ├── test_discovery_server.py  # Agent A — gene expression mock
    ├── test_network_server.py    # Agent B — TF-Target network mock
    └── test_validation_server.py # Agent D — in-silico simulation mock
```
