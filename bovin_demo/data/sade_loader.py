"""A2-M5 · T5.1 — Sade-Feldman GSE120575 pseudobulk loader (external validation).

Sade-Feldman 2018 is a **scRNA-seq** cohort (16,291 CD45+-sorted cells × 48
patients) — we can't feed it to the bulk-trained pool model directly.
:func:`load_sade_feldman` aggregates cells to patient-level pseudobulk
(mean TPM across all cells per patient, per plan §2.1 spec), then selects
BOVIN observable symbols using the same alias table as the Tier A loaders.

Role in v2
----------
**External validation only** — the pool doesn't include Sade, and the
pre-registered caveat in the plan (§2.1, CD45+ enrichment removes tumor
cells → CRT/HMGB1 signal suppressed) makes a low AUC on this cohort an
*expected* outcome, not a failure of the pool model.

Returns an :class:`ICIBundle` so callers can use the same evaluation
plumbing as for pool members.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from bovin_demo.data.ici_loader import (
    ICIBundle,
    _DEFAULT_RECIST_TO_BINARY,
    _finalize_clinical,
    _select_bovin_rows,
    load_gene_aliases,
)


_DEFAULT_RAW_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw_ici" / "sade_feldman_gse120575"
)


def _parse_cell_metadata(path: Path) -> pd.DataFrame:
    """Parse the messy GEO metadata template → (cell_id → patient/timepoint/response).

    GEO's upload template begins with ~15 lines of freeform description, then
    a header row of column names (``title``, ``characteristics: patinet ID...``,
    ``characteristics: response``, ``characteristics: therapy``), then one row
    per cell. The column layout isn't stable in width (early rows use fewer
    tabs), so we find the header by looking for the literal "Sample name"
    token in column 0.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        rows = [L.rstrip("\n").split("\t") for L in f]

    hdr_idx = None
    for i, r in enumerate(rows):
        if r and r[0].strip() == "Sample name":
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError(f"could not find 'Sample name' header row in {path}")

    header = [c.strip() for c in rows[hdr_idx]]
    data_rows = [r for r in rows[hdr_idx + 1:] if r and r[0].strip().startswith("Sample")]

    # Build a dataframe padded / truncated to header width.
    w = len(header)
    norm = [(r + [""] * w)[:w] for r in data_rows]
    df = pd.DataFrame(norm, columns=header)

    # Canonical column names (spellings in GEO vary — lowercase + trim fuzzy matches).
    def _find(substring: str, fallback: str | None = None) -> str:
        for c in df.columns:
            if substring in c.lower():
                return c
        if fallback is not None:
            return fallback
        raise KeyError(f"no column contains {substring!r}; have {list(df.columns)}")

    col_title = _find("title")                   # cell barcode like "A10_P3_M11"
    col_patient = _find("patient", _find("patinet"))   # GEO typo'd "patinet" in 2018
    col_response = _find("response")
    col_therapy = _find("therapy")

    out = pd.DataFrame({
        "cell_id":      df[col_title].astype(str),
        "patient_ts":   df[col_patient].astype(str),      # e.g., "Pre_P1" / "Post_P3"
        "response_raw": df[col_response].astype(str).str.strip(),
        "therapy":      df[col_therapy].astype(str).str.strip(),
    })
    # Split Pre_P1 → (timepoint, patient)
    ts_pat = out["patient_ts"].str.extract(r"^(Pre|Post)_?(P\d+)$", expand=True)
    out["timepoint"] = ts_pat[0].str.lower().fillna("unknown")
    out["patient_id"] = ts_pat[1].fillna(out["patient_ts"])
    return out


def _read_scrna_matrix_filtered(
    path: Path,
    keep_symbols: set[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Streaming reader — keep only rows matching ``keep_symbols``.

    The full matrix is ~16k cells × ~25k genes in TPM (≈ 3 GB as float64).
    Loading it to pandas blows the Docker memory cap. Since we only need
    the 72 BOVIN observable symbols downstream, this helper parses
    line-by-line and appends only matching rows to a dict → DataFrame.
    Memory peak stays under ~100 MB.

    Returns ``(expr_genes_by_cells, patient_ts_by_cell)`` with ``expr``
    indexed by gene symbol (only BOVIN-relevant rows kept, duplicates
    collapsed by first occurrence).
    """
    rows: dict[str, np.ndarray] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\n").split("\t")
        tag_line    = f.readline().rstrip("\n").split("\t")

        cells = header_line[1:]
        tags  = tag_line[1:]
        if len(tags) != len(cells):
            raise ValueError(
                f"sade matrix: cell count {len(cells)} / tag count {len(tags)} mismatch"
            )

        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            sym = line[:tab].strip()
            if sym not in keep_symbols or sym in rows:
                continue
            # parse the rest as float32; TPM values fit trivially in 32-bit
            values = np.fromstring(line[tab + 1:], sep="\t", dtype=np.float32)
            if len(values) != len(cells):
                continue   # malformed row — skip
            rows[sym] = values

    if not rows:
        raise RuntimeError("no BOVIN symbols found in sade scRNA matrix")

    df = pd.DataFrame(rows).T
    df.columns = cells
    df.index.name = "hgnc_symbol"
    patient_ts = pd.Series(tags, index=cells, name="patient_ts")
    return df, patient_ts


def load_sade_feldman(
    raw_dir: str | Path = _DEFAULT_RAW_DIR,
    aliases_csv: str | Path | None = None,
    *,
    timepoint: str = "pre",                       # aggregate over Pre cells only
    min_cells_per_patient: int = 50,              # drop patients with fewer cells
) -> ICIBundle:
    """Load Sade-Feldman as a patient-level pseudobulk :class:`ICIBundle`.

    Parameters
    ----------
    raw_dir : path-like
        Directory containing the two GSE120575 files. Defaults to
        ``data/raw_ici/sade_feldman_gse120575``.
    aliases_csv : path-like, optional
        BOVIN alias table. Defaults to
        ``bovin_demo/data/static/bovin_gene_aliases.csv``.
    timepoint : {"pre", "post", "both"}
        Which cells to include in pseudobulk aggregation. Default ``"pre"``
        for parity with the ``filter_timepoint="pre"`` pool setting.
    min_cells_per_patient : int
        Drop patients whose retained-cell count falls below this floor
        (plan §2.1 note — low-cell patients have inflated pseudobulk variance).
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else _DEFAULT_RAW_DIR
    aliases = load_gene_aliases(aliases_csv or
        Path(__file__).resolve().parents[0] / "static" / "bovin_gene_aliases.csv")

    meta = _parse_cell_metadata(raw_dir / "GSE120575_patient_ID_single_cells.txt.gz")
    keep_symbols = set(aliases["hgnc_symbol"].dropna().astype(str))
    expr_g_by_c, _ = _read_scrna_matrix_filtered(
        raw_dir / "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz",
        keep_symbols=keep_symbols,
    )

    # Align metadata to expression columns.
    meta = meta.set_index("cell_id")
    common_cells = expr_g_by_c.columns.intersection(meta.index)
    expr_g_by_c = expr_g_by_c[common_cells]
    meta = meta.loc[common_cells]

    # Filter timepoint.
    if timepoint in {"pre", "post"}:
        keep = meta["timepoint"] == timepoint
        expr_g_by_c = expr_g_by_c.loc[:, keep]
        meta = meta.loc[keep]

    # Pseudobulk: mean TPM per patient across all retained cells.
    # Transpose to (cells × genes) so groupby-by-patient works on rows.
    expr_c_by_g = expr_g_by_c.T
    patient_series = meta["patient_id"]
    cells_per_patient = patient_series.value_counts()
    keep_patients = cells_per_patient[cells_per_patient >= min_cells_per_patient].index
    mask = patient_series.isin(keep_patients)
    expr_c_by_g = expr_c_by_g.loc[mask]
    meta = meta.loc[mask]

    pseudobulk = expr_c_by_g.groupby(meta["patient_id"]).mean()       # patients × genes
    # Match the pool's log-scale convention — source was TPM (non-log);
    # pool loaders use their native scale (cBioPortal/iAtlas = log-scale
    # already, TPM = linear). Apply log2(TPM+1) here so the downstream
    # per-cohort-z-score is comparable across cohorts.
    pseudobulk = np.log2(pseudobulk + 1.0)

    # Convert to genes × patients, then select BOVIN observable symbols.
    hgnc_expr, hit_rate = _select_bovin_rows(
        pseudobulk.T, aliases, source="hgnc",
    )
    expr = hgnc_expr.T                                                 # patients × 72

    # Clinical: one row per kept patient, response + therapy from metadata.
    response_map = meta.groupby("patient_id")["response_raw"].first()
    therapy_map  = meta.groupby("patient_id")["therapy"].first()

    clin = pd.DataFrame({
        "sample_id":    expr.index,
        "patient_id":   expr.index,
        "timepoint":    timepoint if timepoint != "both" else "mixed",
        "response_raw": [response_map.get(p, "unknown") for p in expr.index],
        "treatment":    [therapy_map.get(p, "unknown") for p in expr.index],
        "disease":      "melanoma",
    }).set_index("sample_id")
    # Map Sade's response strings through the canonical dict + a couple of
    # cohort-specific aliases. Default dict has "responder"/"non-responder";
    # Sade writes "Responder"/"Non-responder" (title-case) — cover both.
    title_case = {"Responder": 1.0, "Non-responder": 0.0, "NonResponder": 0.0}
    combined = {**_DEFAULT_RECIST_TO_BINARY, **title_case}
    clin["ici_response"] = clin["response_raw"].map(combined)

    clin = _finalize_clinical(clin, "sade_feldman_gse120575")
    return ICIBundle(
        cohort_id="sade_feldman_gse120575",
        expr=expr.loc[clin.index],
        clinical=clin,
        gene_id_source="hgnc",
        hit_rate=hit_rate,
    )
