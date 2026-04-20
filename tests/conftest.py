"""Shared pytest fixtures.

Real TCGA-COAD isn't redistributable and the Xena tsv is ~50 MB — tests can't
depend on it. Instead we synthesize a Xena-shaped fixture that covers most of
the graph's observable symbols, enough to exercise every data-path corner
(load → align → label → split) at < 100 ms per test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bovin_demo.data.labels import NEGATIVE_GENES, POSITIVE_GENES


# --- hard-coded symbol sets, kept in sync with bovin_pathway_v0.json --------
# We list the non-aggregate observable symbols explicitly here rather than
# re-parsing the graph in the fixture so a wrong edit to the JSON shows up
# as a test failure (instead of silently feeding fake "hits").
OBSERVABLE_SINGLE_SYMBOLS: tuple[str, ...] = (
    "ST6GAL1", "MGAT5", "CMAS", "MAVS", "EIF2AK2", "EIF2S1", "ATF4", "DDIT3",
    "EIF2AK3", "ATF6", "XBP1", "CYCS", "CASP9", "CASP3", "BCL2", "BAX",
    "TFAM", "DNM1L", "CALR", "HMGB1", "ANXA1", "HSPA1A", "HSP90AA1", "LDHA",
    "LDHB", "SLC16A1", "SLC16A3", "HIF1A", "VEGFA", "NFE2L2", "LRP1", "TLR4",
    "P2RX7", "AGER", "NLRP3", "CASP1", "IL1B", "CD47", "SIRPA", "CD80",
    "CD83", "CD86", "ITGAX", "HLA-A", "HLA-DRA", "CCR7", "IL12B", "CD8A",
    "CD3E", "CD4", "IFNG", "GZMB", "PRF1", "IL17A", "TNF", "IL6", "PDCD1",
    "CD274", "CTLA4", "CD24", "CD44", "SELL", "TCF7", "FOXP3", "ADGRE1",
    "MRC1", "NOS2", "ARG1",
)
# Aggregate components — include one from each so the aggregate resolves.
AGGREGATE_COMPONENTS: tuple[str, ...] = ("IFNA", "TRG")
# A handful of "padding" genes that aren't in the pathway, to mimic a real
# RNA-seq matrix where pathway genes are <1% of total.
PADDING_GENES: tuple[str, ...] = (
    "GAPDH", "ACTB", "TUBB", "RPL13", "UBC", "B2M", "HPRT1", "PPIA",
    "RPS18", "GUSB", "YWHAZ", "SDHA",
)


@pytest.fixture(scope="session")
def graph_json() -> dict:
    """Load the packaged graph JSON once per test session."""
    path = Path(__file__).resolve().parents[1] / "bovin_demo" / "graph" / "bovin_pathway_v0.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_expr(
    n_samples: int,
    gene_symbols: list[str],
    seed: int,
) -> pd.DataFrame:
    """Return samples × genes log2(RSEM+1)-like matrix, values ~ N(5, 2)."""
    rng = np.random.default_rng(seed)
    sample_ids = [f"TCGA-TEST-{i:04d}" for i in range(n_samples)]
    data = rng.normal(loc=5.0, scale=2.0, size=(n_samples, len(gene_symbols)))
    data = np.clip(data, 0, None)  # log2(RSEM+1) ≥ 0
    return pd.DataFrame(data, index=pd.Index(sample_ids, name="sample"), columns=gene_symbols)


@pytest.fixture(scope="session")
def synthetic_coad_frame() -> pd.DataFrame:
    """samples × genes frame with ~92% coverage of observable pathway symbols.

    We drop a small number of pathway symbols on purpose so the hit-rate test
    ``≥ 0.70`` actually measures something (a matrix that includes *every*
    observable gene would tautologically pass).
    """
    # Drop 6 observable symbols to simulate real-world gaps (92% coverage).
    missing = {"EIF2AK3", "ATF6", "SLC16A3", "AGER", "CD86", "SELL"}
    pathway_syms = [s for s in OBSERVABLE_SINGLE_SYMBOLS if s not in missing]
    # Guarantee the 6 ICD label genes are present even if one was in `missing`.
    label_genes = [g for g in POSITIVE_GENES + NEGATIVE_GENES if g not in pathway_syms]
    gene_symbols = list(dict.fromkeys(
        pathway_syms + list(AGGREGATE_COMPONENTS) + label_genes + list(PADDING_GENES)
    ))
    return _synthetic_expr(n_samples=80, gene_symbols=gene_symbols, seed=42)


@pytest.fixture
def xena_like_raw_dir(tmp_path: Path, synthetic_coad_frame: pd.DataFrame) -> Path:
    """Materialize the synthetic matrix as Xena-shaped tsv files under tmp_path.

    Xena's ``HiSeqV2`` is gene × sample; we write it that way so
    ``load_coad`` exercises the same transpose path as on real data.
    """
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    expr_gs = synthetic_coad_frame.T  # genes × samples
    expr_gs.index.name = "sample"  # Xena quirk — the index header is "sample"
    expr_gs.to_csv(raw / "HiSeqV2", sep="\t")

    clinical = pd.DataFrame(
        {
            "sample_type": ["Primary Tumor"] * len(synthetic_coad_frame),
            "gender": ["MALE", "FEMALE"] * (len(synthetic_coad_frame) // 2),
            "age_at_initial_pathologic_diagnosis": np.arange(50, 50 + len(synthetic_coad_frame)),
        },
        index=synthetic_coad_frame.index,
    )
    clinical.index.name = "sampleID"
    clinical.to_csv(raw / "COAD_clinicalMatrix", sep="\t")

    survival = pd.DataFrame(
        {
            "sample": synthetic_coad_frame.index,
            "OS": np.random.default_rng(7).integers(0, 2, size=len(synthetic_coad_frame)),
            "OS.time": np.random.default_rng(8).integers(30, 3000, size=len(synthetic_coad_frame)),
        }
    )
    survival.to_csv(raw / "COAD_survival.txt", sep="\t", index=False)

    return raw
