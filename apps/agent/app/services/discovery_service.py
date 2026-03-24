"""Discovery service — business logic for parsing biomarker candidates."""

import re


def parse_gene_list(text: str) -> list[str]:
    """Parse gene symbols from LLM text output."""
    genes = re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', text)
    stopwords = {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "DEG", "TF", "MCP", "RAG", "GOI"}
    return list(dict.fromkeys(g for g in genes if g not in stopwords))
