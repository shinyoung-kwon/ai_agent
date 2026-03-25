"""Summary service — generate concise stage summaries for inter-agent context."""

_MAX_LENGTH = 10000


def _truncate(text: str, max_length: int = _MAX_LENGTH) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def summarize_discovery(content: str, candidates: list[str]) -> str:
    """Summarize discovery stage output."""
    genes = ", ".join(candidates) if candidates else "없음"
    return f"후보 유전자: {genes}\n근거: {_truncate(content)}"


def summarize_network(content: str, network_data: dict) -> str:
    """Summarize network stage output."""
    genes = network_data.get("genes", [])
    gene_str = ", ".join(genes) if genes else "없음"
    return f"네트워크 유전자: {gene_str}\n분석: {_truncate(content)}"


def summarize_reasoning(content: str) -> str:
    """Summarize reasoning stage output."""
    return f"타당성 분석: {_truncate(content)}"
