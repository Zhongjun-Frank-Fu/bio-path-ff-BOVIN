"""M2 · T2.1–T2.2 — load TCGA-COAD from UCSC Xena static tsv.

Xena file format (reference)
----------------------------
``HiSeqV2`` (gene × sample matrix):

    sample   TCGA-AA-...  TCGA-AA-...
    A1BG     4.8621       5.1202
    A1CF     2.3817       1.9902
    ...

``COAD_clinicalMatrix`` (sample × clinical fields): wide tsv, first column is
the TCGA sample barcode.  ``COAD_survival.txt`` is a tighter subset with
``OS`` / ``OS.time`` / ``PFI`` / ``PFI.time``.

``load_coad`` parses all three, transposes the expression matrix to
**samples × genes**, and joins the clinical / survival fields on sample id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPR_FILENAMES = ("HiSeqV2", "HiSeqV2.gz")
# Accept any *_clinicalMatrix / *_survival.txt so the same loader works on
# COAD, LUAD, or any future sampleMap cohort without a code change.
CLINICAL_FILENAMES = ("COAD_clinicalMatrix", "LUAD_clinicalMatrix")
SURVIVAL_FILENAMES = ("COAD_survival.txt", "LUAD_survival.txt")
CLINICAL_GLOB = "*_clinicalMatrix"
SURVIVAL_GLOB = "*_survival.txt"


@dataclass(frozen=True)
class CoadBundle:
    """Container returned by :func:`load_coad`.

    Attributes
    ----------
    expr : pd.DataFrame
        samples × genes, log2(RSEM+1) values. Index name = "sample".
    clinical : pd.DataFrame
        samples × clinical fields. Index aligned with ``expr`` on the
        intersection of barcodes.
    survival : pd.DataFrame | None
        samples × {OS, OS.time, PFI, PFI.time}. ``None`` if the optional
        survival file is not present.
    """

    expr: pd.DataFrame
    clinical: pd.DataFrame
    survival: pd.DataFrame | None

    @property
    def n_samples(self) -> int:
        return len(self.expr)

    @property
    def n_genes(self) -> int:
        return self.expr.shape[1]


def _resolve(raw_dir: Path, candidates: tuple[str, ...], glob: str | None = None) -> Path:
    for name in candidates:
        p = raw_dir / name
        if p.exists():
            return p
    # Fall back to glob so cohort-specific names (LUAD_clinicalMatrix, etc.)
    # don't require patching the explicit candidate list.
    if glob is not None:
        matches = sorted(raw_dir.glob(glob))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"none of {candidates} (or glob {glob!r}) found under {raw_dir}. "
        f"Run `make data-coad` / `make data-luad`."
    )


def _read_expr(path: Path) -> pd.DataFrame:
    """Read a Xena HiSeqV2 tsv (gzipped or plain) → samples × genes frame.

    Xena's HiSeqV2 puts genes in rows and samples in columns; we transpose
    so downstream code can treat rows as observations.
    """
    df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    df.index.name = "gene"
    df = df.T  # samples × genes
    df.index.name = "sample"
    # Drop any duplicate gene columns — HiSeqV2 sometimes has alias rows.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    # Enforce numeric; any stray non-numeric cell becomes NaN rather than object.
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _read_clinical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    df.index.name = "sample"
    return df


def _read_survival(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # The Xena survival file has "sample" column; make it the index.
    if "sample" in df.columns:
        df = df.set_index("sample")
    elif "_PATIENT" in df.columns:
        df = df.set_index("_PATIENT")
    df.index.name = "sample"
    return df


def load_coad(raw_dir: str | Path) -> CoadBundle:
    """Load TCGA-COAD RNA-seq + clinical + survival.

    Parameters
    ----------
    raw_dir : path-like
        Directory containing the files written by
        ``tools/download_tcga_coad.sh`` (defaults to ``data/raw``).

    Returns
    -------
    CoadBundle
        with ``expr`` (samples × genes), ``clinical``, and optional ``survival``.

    Raises
    ------
    FileNotFoundError
        If neither ``HiSeqV2`` nor ``HiSeqV2.gz`` is present.
    """
    raw = Path(raw_dir)
    expr_path = _resolve(raw, EXPR_FILENAMES)
    clin_path = _resolve(raw, CLINICAL_FILENAMES, glob=CLINICAL_GLOB)

    expr = _read_expr(expr_path)
    clinical = _read_clinical(clin_path)

    survival: pd.DataFrame | None = None
    try:
        surv_path = _resolve(raw, SURVIVAL_FILENAMES, glob=SURVIVAL_GLOB)
        survival = _read_survival(surv_path)
    except FileNotFoundError:
        survival = None

    # Align on intersection of sample barcodes — Xena clinical covers a few
    # samples that RNA-seq doesn't, and vice versa.
    common = expr.index.intersection(clinical.index)
    expr = expr.loc[common].copy()
    clinical = clinical.loc[common].copy()
    if survival is not None:
        survival = survival.loc[survival.index.intersection(common)].copy()

    return CoadBundle(expr=expr, clinical=clinical, survival=survival)
