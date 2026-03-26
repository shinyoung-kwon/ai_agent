"""Pydantic schemas for structured LLM output."""

from pydantic import BaseModel, Field


class DiscoveryOutput(BaseModel):
    """Discovery agent structured output."""
    candidates: list[str] = Field(
        description="Candidate biomarker gene symbols (e.g. EGFR, TP53)"
    )
    rationale: str = Field(
        description="Brief explanation for candidate selection"
    )
    interpretation: str = Field(
        description="Bioinformatics interpretation: DEG pattern summary, "
        "expression significance, and biological context of selected candidates"
    )


class NetworkOutput(BaseModel):
    """Network agent structured output."""
    genes: list[str] = Field(
        description="Gene symbols found in the regulatory network"
    )
    key_findings: str = Field(
        description="Key network analysis findings"
    )
    interpretation: str = Field(
        description="Bioinformatics interpretation: regulatory network topology, "
        "hub TF significance, and condition-specific regulatory implications"
    )


class CandidateRanking(BaseModel):
    """Single candidate ranking entry."""
    gene: str = Field(description="Gene symbol")
    rank: int = Field(description="Priority rank (1 = highest)")
    rationale: str = Field(
        description="Brief justification for this ranking"
    )


class ReasoningOutput(BaseModel):
    """Reasoning agent structured output."""
    rankings: list[CandidateRanking] = Field(
        description="Ranked candidates with rationale"
    )
    recommendation: str = Field(
        description="Final recommendation summary"
    )
    interpretation: str = Field(
        description="Bioinformatics interpretation: biological plausibility assessment, "
        "pathway convergence analysis, and clinical relevance of top candidates"
    )


class ValidationOutput(BaseModel):
    """Validation agent structured output."""
    confirmed_biomarkers: list[str] = Field(
        description="Confirmed biomarker gene symbols"
    )
    summary: str = Field(
        description="Validation result summary"
    )
    interpretation: str = Field(
        description="Bioinformatics interpretation: simulation confidence assessment, "
        "master regulator significance, and recommendations for experimental validation"
    )
