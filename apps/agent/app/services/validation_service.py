"""Validation service — business logic for parsing validation results."""

import re


def parse_validation_results(text: str) -> list[dict]:
    """Parse confirmed biomarker genes from validation agent LLM response."""
    genes = re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', text)
    stopwords = {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "DEG", "TF", "MCP", "RAG", "GOI"}
    confirmed = list(dict.fromkeys(g for g in genes if g not in stopwords))
    return [{"confirmed_biomarkers": confirmed, "raw_response": text}]
