"""Network service — business logic for parsing gene network data."""

import re


def parse_network_data(text: str) -> dict:
    """Parse gene names and relationships from network agent LLM response."""
    genes = re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', text)
    stopwords = {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "DEG", "TF", "MCP", "RAG", "GOI"}
    unique_genes = list(dict.fromkeys(g for g in genes if g not in stopwords))
    return {
        "genes": unique_genes,
        "raw_response": text,
    }
