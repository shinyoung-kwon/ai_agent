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
