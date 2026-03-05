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

AI agents collaborate to perform the biomarker discovery workflow:

Transcriptomic Data
      ↓
Candidate Biomarker Generation Agent
      ↓
Evidence Retrieval Agent
      ↓
Evidence Evaluation Agent
      ↓
Biomarker Ranking Agent
      ↓
Final Report Generation

Each agent interacts with external tools through MCP servers.

---

## Agent Roles

### Candidate Generation Agent
Identifies potential biomarker candidates from transcriptomic data.

### Evidence Retrieval Agent
Retrieves supporting evidence from scientific literature and biological databases.

### Evidence Evaluation Agent
Analyzes the reliability and relevance of retrieved evidence.

### Biomarker Ranking Agent
Ranks biomarker candidates based on evidence strength.

### Report Generation Agent
Produces structured explanations and final biomarker reports.

---

## Project Structure

```
ai_agent_pj
│
├── README.md          # project documentation
├── .gitignore         # files and folders ignored by git
├── .env               # environment variables (not committed)
├── requirements.txt   # python dependencies
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
└── mcp-servers          # MCP tool servers
```


  # MCP tool servers for external tools

configs/
  dev.yaml
  prompts.yaml

scripts/
  bootstrap.sh
  build_index.sh
  run_local.sh

data/
  raw/
  processed/
  indices/
