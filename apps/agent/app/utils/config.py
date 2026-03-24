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
    from langchain.chat_models import init_chat_model

    config = get_config()
    llm_config = config["llm"]

    return init_chat_model(
        llm_config["model"],
        temperature=llm_config.get("temperature", 0),
    )


def setup_langsmith():
    """Configure LangSmith tracing from dev.yaml and .env."""
    config = get_config()
    ls_config = config.get("langsmith", {})
    project_name = ls_config.get("project", "default")

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project_name)

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("[LangSmith] WARNING: LANGSMITH_API_KEY not set. Tracing disabled.")
        os.environ["LANGSMITH_TRACING"] = "false"
    else:
        print(f"[LangSmith] Tracing enabled — project: {project_name}")


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
