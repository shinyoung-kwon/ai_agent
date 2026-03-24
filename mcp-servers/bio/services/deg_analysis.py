"""DEG analysis service — PyDESeq2 based differential expression analysis.

Replace _build_sample_data() with real count matrix loading for production.
"""

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def _build_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sample count matrix and metadata for testing.

    Returns:
        (counts, metadata) — counts: genes x samples, metadata: sample conditions
    """
    np.random.seed(42)
    genes = [
        "BRCA1", "TP53", "EGFR", "MYC", "KRAS",
        "PTEN", "RB1", "AKT1", "VEGFA", "CDH1",
    ]
    n_control = 5
    n_treatment = 5
    samples = (
        [f"ctrl_{i}" for i in range(n_control)]
        + [f"treat_{i}" for i in range(n_treatment)]
    )

    # Simulate count data
    base_counts = np.random.poisson(lam=500, size=(len(samples), len(genes)))
    # Elevate expression for first 5 genes in treatment group
    base_counts[n_control:, :5] += np.random.poisson(
        lam=300, size=(n_treatment, 5),
    )

    counts = pd.DataFrame(base_counts, index=samples, columns=genes)
    metadata = pd.DataFrame(
        {"condition": ["control"] * n_control + ["treatment"] * n_treatment},
        index=samples,
    )
    return counts, metadata


def _run_deseq2(
    counts: pd.DataFrame, metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Run PyDESeq2 pipeline and return results DataFrame."""
    dds = DeseqDataSet(counts=counts, metadata=metadata, design="~condition")
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=("condition", "treatment", "control"))
    stat_res.summary()
    return stat_res.results_df


def run_deg_analysis(pvalue_cutoff: float = 0.05) -> list[dict]:
    """Run DEG analysis and return genes below the p-value cutoff.

    Args:
        pvalue_cutoff: adjusted p-value threshold for filtering.

    Returns:
        List of dicts with gene, log2FoldChange, pvalue, padj.
    """
    counts, metadata = _build_sample_data()
    results_df = _run_deseq2(counts, metadata)

    sig = results_df[results_df["padj"] < pvalue_cutoff].copy()
    sig = sig.sort_values("padj")

    degs = []
    for gene, row in sig.iterrows():
        degs.append({
            "gene": gene,
            "log2FoldChange": round(float(row["log2FoldChange"]), 4),
            "pvalue": float(row["pvalue"]),
            "padj": float(row["padj"]),
        })
    return degs


def get_gene_expression(gene: str) -> dict | None:
    """Look up expression stats for a specific gene.

    Args:
        gene: gene symbol (case-insensitive).

    Returns:
        Dict with expression stats, or None if not found.
    """
    counts, metadata = _build_sample_data()
    results_df = _run_deseq2(counts, metadata)

    gene_upper = gene.upper()
    if gene_upper not in results_df.index:
        return None

    row = results_df.loc[gene_upper]
    return {
        "gene": gene_upper,
        "log2FoldChange": round(float(row["log2FoldChange"]), 4),
        "pvalue": float(row["pvalue"]),
        "padj": float(row["padj"]),
        "baseMean": round(float(row["baseMean"]), 2),
    }
