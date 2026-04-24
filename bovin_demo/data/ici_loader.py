"""A2-M2 · T2.1–T2.3 — load Tier A ICI cohorts and pool them.

Mirrors the role of :mod:`bovin_demo.data.tcga_loader` but for the 6-cohort
Tier A ICI pool (AIM2-TRAINING-PLAN.md §2.1). The engine side of BOVIN stays
unchanged; this file is the only new data-path piece Aim 2 adds on the input
axis (plan §1 · "Reuse engine, swap label").

Key design choices
------------------
* **Reverse-lookup gene ID normalization.** Each cohort uses a native gene ID
  convention (HGNC symbol for Hugo/Gide/Hammerman/Cloughesy, NCBI Entrez for
  Riaz, versioned Ensembl for Seo). Instead of mass-converting each cohort's
  20k+ rows, we walk in from the BOVIN side: the 72 BOVIN observable symbols
  have their Entrez + Ensembl aliases pre-computed in
  ``bovin_demo/data/static/bovin_gene_aliases.csv``; the loader simply
  selects matching rows from each cohort's native-ID matrix.
* **Bundle, don't merge, at the per-cohort step.** ``ICIBundle`` carries both
  ``expr`` (samples × HGNC-aligned BOVIN symbols) and ``clinical``
  (aligned on sample_id). Cross-cohort pooling happens later in
  :func:`load_ici_pool`, which applies per-cohort z-score then concatenates
  on the gene intersection.
* **One label convention (for now): RECIST binary.** Each adapter writes a
  unified ``clinical`` frame with ``response_raw`` (source column as-is) and
  ``ici_response`` ∈ {0, 1, NaN}. The actual mapping function lives in
  :mod:`bovin_demo.data.labels` so configs can swap mapping rules.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ICIBundle:
    """Per-cohort ICI data aligned to the BOVIN 72-symbol observable set.

    Attributes
    ----------
    cohort_id : str
        Stable manifest ID (e.g., ``"riaz_gse91061"``).
    expr : pd.DataFrame
        Samples × HGNC-symbol subset. Only BOVIN observable symbols are kept;
        missing ones become NaN columns so downstream alignment is uniform
        across cohorts.
    clinical : pd.DataFrame
        Samples × {patient_id, response_raw, ici_response, timepoint,
        treatment, disease, cohort_id}. Index aligned with ``expr``.
    gene_id_source : str
        ``"hgnc"`` | ``"entrez"`` | ``"ensembl"`` — records which alias column
        drove the lookup (useful for audit + report generation).
    hit_rate : float
        Fraction of BOVIN observable symbols that were found in this cohort's
        raw expression matrix.
    """

    cohort_id: str
    expr: pd.DataFrame
    clinical: pd.DataFrame
    gene_id_source: str
    hit_rate: float

    @property
    def n_samples(self) -> int:
        return len(self.expr)

    @property
    def n_patients(self) -> int:
        return self.clinical["patient_id"].nunique()


@dataclass(frozen=True)
class ICIPoolBundle:
    """Pooled ICI data with per-cohort z-score applied.

    Attributes
    ----------
    expr : pd.DataFrame
        (samples × genes) — gene intersection across pooled cohorts,
        per-cohort z-score applied, index = unique ``sample_id``.
    clinical : pd.DataFrame
        Same index as ``expr``, with ``cohort_id`` column to support LOCO splits.
    cohorts : list[str]
        Cohort IDs actually pooled (drops any cohort that returned zero samples).
    genes : list[str]
        HGNC symbols in ``expr.columns``.
    per_cohort_hit_rates : dict[str, float]
        Diagnostic: per-cohort hit rate against BOVIN observable set.
    """

    expr: pd.DataFrame
    clinical: pd.DataFrame
    cohorts: list[str]
    genes: list[str]
    per_cohort_hit_rates: dict[str, float] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.expr)

    @property
    def n_patients(self) -> int:
        return self.clinical["patient_id"].nunique()


# ---------------------------------------------------------------------------
# Helpers: paths + alias lookup
# ---------------------------------------------------------------------------

_DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_ici"
_DEFAULT_ALIASES = Path(__file__).resolve().parents[0] / "static" / "bovin_gene_aliases.csv"

# RECIST mapping reused by every adapter. Override via labels.recist_binary_label
# if a cohort needs a non-default mapping (e.g., Hammerman CR/NR encoding).
_DEFAULT_RECIST_TO_BINARY: dict[str, float] = {
    "CR": 1.0, "PR": 1.0,
    "SD": 0.0, "PD": 0.0,
    "Complete Response": 1.0, "Partial Response": 1.0,
    "Stable Disease":    0.0, "Progressive Disease": 0.0,
    "responder":         1.0, "non-responder":       0.0,
    "NE": np.nan, "NA": np.nan, "Not Evaluable": np.nan,
}


def load_gene_aliases(path: str | Path = _DEFAULT_ALIASES) -> pd.DataFrame:
    """Return the 72-row BOVIN-symbol ↔ Entrez/Ensembl alias table."""
    df = pd.read_csv(path, dtype={"entrez_id": "Int64"})
    required = {"node_id", "node_symbol", "hgnc_symbol", "entrez_id", "ensembl_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def _select_bovin_rows(
    native_expr: pd.DataFrame,
    aliases: pd.DataFrame,
    source: str,
) -> tuple[pd.DataFrame, float]:
    """Pull out only the rows matching BOVIN observable symbols.

    Parameters
    ----------
    native_expr : pd.DataFrame
        Raw cohort matrix with gene IDs on the index (or in a column — caller
        is responsible for setting the right index). Columns are samples.
    aliases : pd.DataFrame
        Output of :func:`load_gene_aliases`.
    source : {"hgnc", "entrez", "ensembl"}
        Which alias column to join on.

    Returns
    -------
    tuple[pd.DataFrame, float]
        * ``hgnc_expr`` — index is **HGNC symbol**, columns unchanged
          (samples); only BOVIN observable symbols kept. Missing symbols are
          added as all-NaN rows so the output shape is always (72, n_samples).
        * ``hit_rate`` — fraction of BOVIN symbols actually found in the raw
          matrix (before the NaN fill-in).
    """
    if source == "hgnc":
        keys = aliases["hgnc_symbol"].tolist()
        # identity map for HGNC — each symbol maps to itself.
        lookup = {s: s for s in keys}
    elif source == "entrez":
        keys = aliases["entrez_id"].dropna().astype(int).tolist()
        lookup = aliases.dropna(subset=["entrez_id"]).assign(
            entrez_id=lambda d: d["entrez_id"].astype(int)
        ).set_index("entrez_id")["hgnc_symbol"].to_dict()
    elif source == "ensembl":
        # Strip version suffix on both sides of the join so "ENSG00...15" matches "ENSG00...".
        keys = aliases["ensembl_id"].dropna().tolist()
        lookup = aliases.dropna(subset=["ensembl_id"]).set_index("ensembl_id")["hgnc_symbol"].to_dict()
        native_expr = native_expr.copy()
        native_expr.index = native_expr.index.astype(str).str.split(".", n=1).str[0]
    else:
        raise ValueError(f"unknown gene ID source: {source!r}")

    hit_keys = [k for k in keys if k in native_expr.index]
    hit_rate = len(hit_keys) / len(set(aliases["hgnc_symbol"])) if len(aliases) else 0.0

    # Build a (72 × n_samples) frame — missing rows filled with NaN for uniform
    # shape across cohorts so the downstream pool concat works without reindex.
    all_hgnc = aliases["hgnc_symbol"].drop_duplicates().tolist()
    hit_frame = native_expr.loc[hit_keys].copy()
    hit_frame.index = [lookup[k] for k in hit_keys]
    # If the same HGNC symbol got hit by multiple native IDs (rare), average them.
    hit_frame = hit_frame.groupby(level=0).mean()
    hgnc_expr = hit_frame.reindex(all_hgnc)
    hgnc_expr.index.name = "hgnc_symbol"
    return hgnc_expr, hit_rate


def _finalize_clinical(
    clin: pd.DataFrame,
    cohort_id: str,
) -> pd.DataFrame:
    """Ensure clinical frame has the canonical column set before return."""
    required = {"patient_id", "response_raw", "ici_response",
                "timepoint", "treatment", "disease"}
    missing = required - set(clin.columns)
    if missing:
        raise ValueError(
            f"cohort {cohort_id!r}: clinical frame missing {missing}"
        )
    clin = clin.copy()
    clin["cohort_id"] = cohort_id
    # Coerce ici_response to nullable float.
    clin["ici_response"] = clin["ici_response"].astype("Float64")
    return clin


# ---------------------------------------------------------------------------
# Per-cohort adapters
# ---------------------------------------------------------------------------

def _load_riaz(raw_dir: Path, aliases: pd.DataFrame) -> ICIBundle:
    """Riaz 2017 · GSE91061 · nivolumab melanoma · 65 patients (RNA-seq)."""
    cdir = raw_dir / "riaz_gse91061"
    raw = pd.read_csv(cdir / "GSE91061_BMS038109Sample.hg19KnownGene.raw.csv.gz")
    raw = raw.rename(columns={"Unnamed: 0": "entrez_id"}).set_index("entrez_id")
    hgnc_expr, hit_rate = _select_bovin_rows(raw, aliases, source="entrez")
    expr = hgnc_expr.T                       # samples × genes

    # sample ID → patient + timepoint:  "Pt1_Pre_AD101148-6" → (Pt1, pre)
    pattern = re.compile(r"^(Pt\d+)_(Pre|On)_")
    meta = pd.DataFrame({
        "sample_id": expr.index,
        "patient_id": [m.group(1) if (m := pattern.match(s)) else None
                       for s in expr.index],
        "timepoint":  [m.group(2).lower() if (m := pattern.match(s)) else "unknown"
                       for s in expr.index],
    })

    clin_raw = pd.read_csv(cdir / "bms038_clinical_data.csv")
    # Clinical table has Sample column in lowercase ("Pt1_pre"); join on patient+timepoint instead.
    clin_raw["timepoint"] = clin_raw["SampleType"].str.lower()
    clin_raw = clin_raw.rename(columns={"PatientID": "patient_id", "BOR": "response_raw"})
    merged = meta.merge(
        clin_raw[["patient_id", "timepoint", "response_raw"]],
        on=["patient_id", "timepoint"], how="left",
    ).set_index("sample_id")
    merged["ici_response"] = merged["response_raw"].map(_DEFAULT_RECIST_TO_BINARY)
    merged["treatment"] = "anti_pd1"
    merged["disease"]   = "melanoma"

    clin = _finalize_clinical(merged, "riaz_gse91061")
    # Align expr index with clin.
    common = expr.index.intersection(clin.index)
    return ICIBundle(
        cohort_id="riaz_gse91061",
        expr=expr.loc[common],
        clinical=clin.loc[common],
        gene_id_source="entrez",
        hit_rate=hit_rate,
    )


def _parse_geo_characteristics(
    series_matrix_path: Path,
    field_prefix: str,
) -> tuple[list[str], list[str]]:
    """Extract (sample_titles, field_values) from a GEO series_matrix.

    Scans ``!Sample_characteristics_ch1`` rows for the first one whose value
    starts with ``field_prefix:`` (e.g., ``"anti-pd-1 response"``), returns
    the values (prefix stripped) aligned with the ``!Sample_title`` row.
    Returns ([], []) if no matching field is found.
    """
    titles: list[str] = []
    values: list[str] = []
    with gzip.open(series_matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("!"):
                continue
            tag, _, rest = line.partition("\t")
            cells = [t.strip().strip('"') for t in rest.strip().split("\t")]
            if tag == "!Sample_title":
                titles = cells
            elif tag == "!Sample_characteristics_ch1" and cells:
                first = cells[0].lower()
                if first.startswith(field_prefix.lower() + ":"):
                    values = [c.split(":", 1)[1].strip() if ":" in c else c for c in cells]
                    break
    return titles, values


def _load_hugo(raw_dir: Path, aliases: pd.DataFrame) -> ICIBundle:
    """Hugo 2016 · GSE78220 · pembro/nivo melanoma · 28 samples (pre only).

    Response labels live in ``series_matrix`` under the
    ``anti-pd-1 response:`` characteristics field — full RECIST strings for
    all 28 samples, so no paper-Table-S1 dependency.
    """
    cdir = raw_dir / "hugo_gse78220"
    raw = pd.read_excel(cdir / "GSE78220_PatientFPKM.xlsx", engine="openpyxl")
    raw = raw.rename(columns={"Gene": "hgnc_symbol"}).set_index("hgnc_symbol")
    raw = raw[~raw.index.duplicated(keep="first")]
    hgnc_expr, hit_rate = _select_bovin_rows(raw, aliases, source="hgnc")
    expr = hgnc_expr.T

    # Pull RECIST response from series_matrix. Titles align by position with
    # the expression matrix columns via the shared patient-id convention
    # ("Pt1" in both files, modulo trailing ".baseline" suffix on expr side).
    sm_titles, sm_responses = _parse_geo_characteristics(
        cdir / "GSE78220_series_matrix.txt.gz",
        field_prefix="anti-pd-1 response",
    )
    response_by_patient = dict(zip(sm_titles, sm_responses))  # "Pt1" → "Progressive Disease"

    patterns = [re.compile(r"^(Pt\d+[A-Za-z]?)\.baseline$"),
                re.compile(r"^(Pt\d+[A-Za-z]?)$")]
    def _patient(s: str) -> str:
        for p in patterns:
            if (m := p.match(s)):
                return m.group(1)
        return s

    pt_ids = [_patient(s) for s in expr.index]
    response_raw = [response_by_patient.get(p, None) for p in pt_ids]

    meta = pd.DataFrame({
        "sample_id":    expr.index,
        "patient_id":   pt_ids,
        "timepoint":    "pre",
        "response_raw": response_raw,
        "treatment":    "anti_pd1",
        "disease":      "melanoma",
    }).set_index("sample_id")
    meta["ici_response"] = meta["response_raw"].map(_DEFAULT_RECIST_TO_BINARY)

    clin = _finalize_clinical(meta, "hugo_gse78220")
    return ICIBundle(
        cohort_id="hugo_gse78220",
        expr=expr.loc[clin.index],
        clinical=clin,
        gene_id_source="hgnc",
        hit_rate=hit_rate,
    )


def _load_gide(raw_dir: Path, aliases: pd.DataFrame) -> ICIBundle:
    """Gide 2019 · PRJEB23709 → cBioPortal mel_iatlas_gide_2019 · 74 pts."""
    cdir = raw_dir / "gide_prjeb23709"
    raw = pd.read_csv(cdir / "data_mrna_seq_expression.txt", sep="\t")
    raw = raw.rename(columns={"Hugo_Symbol": "hgnc_symbol"}).set_index("hgnc_symbol")
    raw = raw[~raw.index.duplicated(keep="first")]
    hgnc_expr, hit_rate = _select_bovin_rows(raw, aliases, source="hgnc")
    expr = hgnc_expr.T

    # cBioPortal clinical_sample: SAMPLE_ID + RESPONSE (full RECIST string) + SAMPLE_TREATMENT.
    sample_clin = pd.read_csv(cdir / "data_clinical_sample.txt", sep="\t", comment="#")
    sample_clin = sample_clin.rename(columns={
        "SAMPLE_ID":         "sample_id",
        "PATIENT_ID":        "patient_id",
        "RESPONSE":          "response_raw",
        "SAMPLE_TREATMENT":  "timepoint_raw",
    }).set_index("sample_id")
    sample_clin["timepoint"] = sample_clin["timepoint_raw"].str.lower().fillna("unknown")
    sample_clin["ici_response"] = sample_clin["response_raw"].map(_DEFAULT_RECIST_TO_BINARY)
    sample_clin["treatment"] = "anti_pd1"  # approximation — cohort mixes mono + combo
    sample_clin["disease"]   = "melanoma"

    common = expr.index.intersection(sample_clin.index)
    clin = _finalize_clinical(
        sample_clin.loc[common, ["patient_id", "response_raw", "ici_response",
                                  "timepoint", "treatment", "disease"]],
        "gide_prjeb23709",
    )
    return ICIBundle(
        cohort_id="gide_prjeb23709",
        expr=expr.loc[common],
        clinical=clin,
        gene_id_source="hgnc",
        hit_rate=hit_rate,
    )


def _load_hammerman(raw_dir: Path, aliases: pd.DataFrame) -> ICIBundle:
    """Hammerman 2020 · GSE165278 · ipilimumab melanoma · 21 pts.

    Response is encoded in sample ID prefix: ``CR####`` vs ``NR####``.
    """
    cdir = raw_dir / "hammerman_gse165278"
    raw = pd.read_excel(cdir / "GSE165278_TPM_original_23k_genes.xlsx", engine="openpyxl")
    raw = raw.rename(columns={"Gene": "hgnc_symbol"}).set_index("hgnc_symbol")
    raw = raw[~raw.index.duplicated(keep="first")]
    hgnc_expr, hit_rate = _select_bovin_rows(raw, aliases, source="hgnc")
    expr = hgnc_expr.T

    def _decode(sid: str) -> tuple[str, str, float]:
        """Return (patient_id, response_raw, ici_response).

        CR = complete responder (response).
        NR = non-responder (no response).
        SD = stable disease — per RECIST binary convention (plan §2.2) maps to 0.
        Anything else → unknown (NaN label).
        """
        if sid.startswith("CR"):
            return sid, "CR", 1.0
        if sid.startswith("NR"):
            return sid, "NR", 0.0
        if sid.startswith("SD"):
            return sid, "SD", 0.0
        return sid, "unknown", float("nan")

    decoded = [_decode(s) for s in expr.index]
    meta = pd.DataFrame({
        "sample_id":    expr.index,
        "patient_id":   [d[0] for d in decoded],
        "response_raw": [d[1] for d in decoded],
        "ici_response": [d[2] for d in decoded],
        "timepoint":    "pre",        # GSE165278 doesn't declare pre/on; treat as pre
        "treatment":    "anti_ctla4",
        "disease":      "melanoma",
    }).set_index("sample_id")

    clin = _finalize_clinical(meta, "hammerman_gse165278")
    return ICIBundle(
        cohort_id="hammerman_gse165278",
        expr=expr.loc[clin.index],
        clinical=clin,
        gene_id_source="hgnc",
        hit_rate=hit_rate,
    )


_CLOUGHESY_MANUAL_LABELS = (
    Path(__file__).resolve().parents[0] / "static" / "cloughesy_manual_labels.csv"
)


def _load_cloughesy(raw_dir: Path, aliases: pd.DataFrame) -> ICIBundle:
    """Cloughesy 2019 · GSE121810 · neoadjuvant pembro GBM · 29 pts.

    Sample suffix ``_A`` = adjuvant arm (post-surgery pembro only).
    Sample suffix ``_B`` = neoadjuvant arm (pre-surgery pembro).
    No patient has both — each suffix is a distinct patient.

    Per-patient response is **not** in GEO (series_matrix carries only
    ``therapy: neoadjuvant pembrolizumab``) and not in PMC Table S1; the
    swimmer plot in Fig 1b has no readable values. Per-patient OS ships with
    ``static/cloughesy_manual_labels.csv`` — a stub CSV whose
    ``ici_response`` column starts empty. Fill it from an author email
    (see the adjacent ``.TODO.md``); this adapter picks up any non-null
    labels automatically on the next load.
    """
    cdir = raw_dir / "cloughesy_gse121810"
    raw = pd.read_excel(cdir / "GSE121810_Prins.PD1NeoAdjv.Jul2018.HUGO.PtID.xlsx",
                        engine="openpyxl")
    raw = raw.rename(columns={"Genes": "hgnc_symbol"}).set_index("hgnc_symbol")
    raw = raw[~raw.index.duplicated(keep="first")]
    hgnc_expr, hit_rate = _select_bovin_rows(raw, aliases, source="hgnc")
    expr = hgnc_expr.T

    pattern = re.compile(r"^(Pt\d+)_(A|B)$")
    meta = pd.DataFrame({
        "sample_id":    expr.index,
        "patient_id":   [m.group(1) if (m := pattern.match(s)) else s
                         for s in expr.index],
        "arm":          [m.group(2) if (m := pattern.match(s)) else "?"
                         for s in expr.index],
        "timepoint":    "baseline",   # _A + _B both pre-pembro; _B has same-day biopsy at surgery
        "treatment":    "anti_pd1",
        "disease":      "gbm",
    })

    # Pull in manual labels if the static CSV has any non-null values.
    # The file ships with all-NaN ``ici_response`` (M2.1 fill-in pending),
    # so behaviour is identical to "no labels" until someone edits it.
    if _CLOUGHESY_MANUAL_LABELS.exists():
        manual = pd.read_csv(_CLOUGHESY_MANUAL_LABELS)
        manual = manual[["patient_id", "response_raw", "ici_response"]]
        meta = meta.merge(manual, on="patient_id", how="left")
    else:
        meta["response_raw"] = pd.NA
        meta["ici_response"] = pd.NA

    meta = meta.set_index("sample_id")
    clin = _finalize_clinical(meta, "cloughesy_gse121810")
    return ICIBundle(
        cohort_id="cloughesy_gse121810",
        expr=expr.loc[clin.index],
        clinical=clin,
        gene_id_source="hgnc",
        hit_rate=hit_rate,
    )


def _load_seo(raw_dir: Path, aliases: pd.DataFrame) -> ICIBundle:
    """Seo 2020 · GSE165252 · atezolizumab EAC · 40 pts.

    Gene IDs are Ensembl with version suffix; :func:`_select_bovin_rows`
    handles the version strip. Response is in ``series_matrix`` under
    ``characteristics_ch1`` as ``response: responder`` / ``non-responder``.
    """
    cdir = raw_dir / "seo_gse165252"
    raw = pd.read_csv(cdir / "GSE165252_norm.cnt_PERFECT.txt.gz", sep="\t")
    raw = raw.rename(columns={"Unnamed: 0": "ensembl_id"}).set_index("ensembl_id")
    hgnc_expr, hit_rate = _select_bovin_rows(raw, aliases, source="ensembl")
    expr = hgnc_expr.T

    # Parse series_matrix — it's a minischematic file where each "!Sample_XXX" row
    # gives one field across all samples. We pull title + characteristics_ch1 rows.
    sm_path = cdir / "GSE165252_series_matrix.txt.gz"
    titles: list[str] = []
    geo_ids: list[str] = []
    chars: list[list[str]] = []
    with gzip.open(sm_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("!"):
                continue
            tag, _, rest = line.partition("\t")
            if tag == "!Sample_title":
                titles = [t.strip('"') for t in rest.strip().split("\t")]
            elif tag == "!Sample_geo_accession":
                geo_ids = [t.strip('"') for t in rest.strip().split("\t")]
            elif tag == "!Sample_characteristics_ch1":
                chars.append([t.strip('"') for t in rest.strip().split("\t")])

    # Build sample-level metadata from titles: "Patient tumor sample_1_on_treatment" etc.
    def _parse_title(title: str) -> tuple[str, str]:
        m = re.match(r"Patient tumor sample_(\d+)_([a-zA-Z_]+)", title)
        if not m:
            return ("unknown", "unknown")
        return (f"Pt{m.group(1)}", m.group(2).lower())

    # find the characteristics row that contains "response:"
    response_row: list[str] = []
    for row in chars:
        if row and any("response:" in x.lower() for x in row):
            response_row = [x.split(":", 1)[1].strip().lower() if ":" in x else ""
                            for x in row]
            break

    # Map the series_matrix order (GSM IDs) to the expression matrix column order.
    # Expression matrix uses trial sample codes (e.g., "AZ1911-049TE"); series_matrix
    # uses GSM accessions in the same column order as titles. Without a lookup table,
    # we join by positional index assuming the two files share sample order.
    # This is a documented limitation — M2.1 should cross-check via
    # !Sample_description or external mapping.
    parsed_titles = [_parse_title(t) for t in titles]
    if len(parsed_titles) != expr.shape[0]:
        # If lengths differ we fall back to NaN metadata but keep expr.
        meta_df = pd.DataFrame({
            "sample_id":    expr.index,
            "patient_id":   expr.index,
            "timepoint":    "unknown",
            "response_raw": pd.NA,
            "ici_response": pd.NA,
            "treatment":    "anti_pdl1",
            "disease":      "esophageal_adenocarcinoma",
        }).set_index("sample_id")
    else:
        # Normalize timepoint: Seo encodes pre-treatment biopsy as "baseline";
        # downstream filter_timepoint="pre" expects that literal, so map here.
        _tp_map = {"baseline": "pre", "on_treatment": "on", "resection": "post"}
        meta_df = pd.DataFrame({
            "sample_id":    expr.index,
            "patient_id":   [p for p, _ in parsed_titles],
            "timepoint":    [_tp_map.get(t, "unknown") for _, t in parsed_titles],
            "response_raw": response_row if response_row else [pd.NA] * len(parsed_titles),
            "treatment":    "anti_pdl1",
            "disease":      "esophageal_adenocarcinoma",
        }).set_index("sample_id")
        meta_df["ici_response"] = (
            meta_df["response_raw"].map(_DEFAULT_RECIST_TO_BINARY)
            if response_row else pd.Series([pd.NA] * len(meta_df), index=meta_df.index)
        )

    clin = _finalize_clinical(meta_df, "seo_gse165252")
    return ICIBundle(
        cohort_id="seo_gse165252",
        expr=expr.loc[clin.index],
        clinical=clin,
        gene_id_source="ensembl",
        hit_rate=hit_rate,
    )


_ADAPTERS = {
    "riaz_gse91061":        _load_riaz,
    "hugo_gse78220":        _load_hugo,
    "gide_prjeb23709":      _load_gide,
    "hammerman_gse165278":  _load_hammerman,
    "cloughesy_gse121810":  _load_cloughesy,
    "seo_gse165252":        _load_seo,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TIER_A_COHORTS: tuple[str, ...] = tuple(_ADAPTERS.keys())


def load_ici_cohort(
    cohort_id: str,
    raw_dir: str | Path = _DEFAULT_RAW_DIR,
    aliases_csv: str | Path = _DEFAULT_ALIASES,
) -> ICIBundle:
    """Load one ICI cohort.

    Parameters
    ----------
    cohort_id : str
        One of :data:`TIER_A_COHORTS`.
    raw_dir : path-like
        Root of ``data/raw_ici/`` as written by ``tools/download_ici_pool.sh``.
    aliases_csv : path-like
        BOVIN 72-symbol alias table (default in ``bovin_demo/data/static/``).
    """
    if cohort_id not in _ADAPTERS:
        raise KeyError(f"unknown cohort {cohort_id!r}; known: {sorted(_ADAPTERS)}")
    raw_dir = Path(raw_dir)
    aliases = load_gene_aliases(aliases_csv)
    return _ADAPTERS[cohort_id](raw_dir, aliases)


def _per_cohort_zscore(expr: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene column within the cohort; zero-variance → 0 (not NaN)."""
    mu = expr.mean(axis=0)
    sd = expr.std(axis=0, ddof=0).replace(0, np.nan)
    return ((expr - mu) / sd).fillna(0.0)


def load_ici_pool(
    cohorts: tuple[str, ...] | list[str] = TIER_A_COHORTS,
    raw_dir: str | Path = _DEFAULT_RAW_DIR,
    aliases_csv: str | Path = _DEFAULT_ALIASES,
    *,
    filter_timepoint: str | None = None,
    require_label: bool = False,
) -> ICIPoolBundle:
    """Load several cohorts and concat into a pooled matrix with per-cohort z-score.

    Parameters
    ----------
    cohorts : sequence of str
        Which cohort IDs to pool (default: all Tier A).
    raw_dir, aliases_csv : path-like
        Same semantics as :func:`load_ici_cohort`.
    filter_timepoint : str | None
        If given (e.g., ``"pre"``), drop samples whose ``timepoint`` ≠ this value.
        Recommended for training to avoid on-treatment information leakage.
    require_label : bool
        If ``True``, drop samples with missing ``ici_response``. Useful for the
        pooled training split; leave ``False`` for inspection/debug views.

    Returns
    -------
    ICIPoolBundle
    """
    bundles = [load_ici_cohort(c, raw_dir, aliases_csv) for c in cohorts]

    # Pre-filter timepoint / label per cohort so each cohort's z-score is computed
    # on the *trainable* sample subset (not on on-treatment outliers).
    if filter_timepoint is not None:
        bundles = [
            ICIBundle(
                cohort_id=b.cohort_id,
                expr=b.expr.loc[b.clinical["timepoint"] == filter_timepoint],
                clinical=b.clinical.loc[b.clinical["timepoint"] == filter_timepoint],
                gene_id_source=b.gene_id_source,
                hit_rate=b.hit_rate,
            )
            for b in bundles
        ]
    if require_label:
        bundles = [
            ICIBundle(
                cohort_id=b.cohort_id,
                expr=b.expr.loc[b.clinical["ici_response"].notna()],
                clinical=b.clinical.loc[b.clinical["ici_response"].notna()],
                gene_id_source=b.gene_id_source,
                hit_rate=b.hit_rate,
            )
            for b in bundles
        ]
    # Drop empty cohorts (e.g., Hugo post-require_label if clinical TODO not yet filled).
    bundles = [b for b in bundles if len(b.expr)]

    if not bundles:
        raise RuntimeError("no cohorts survived filtering — loosen filter_timepoint / require_label")

    # Per-cohort z-score on the full 72-symbol frame (NaN columns stay NaN).
    zscored = [
        ICIBundle(b.cohort_id, _per_cohort_zscore(b.expr), b.clinical,
                  b.gene_id_source, b.hit_rate)
        for b in bundles
    ]

    # Gene intersection: keep a column only if **every** cohort has ≥ 1 non-NaN sample there.
    gene_masks = [b.expr.notna().any(axis=0) for b in zscored]
    genes = sorted(set.intersection(*[set(m.index[m]) for m in gene_masks]))

    pooled_expr = pd.concat([b.expr[genes] for b in zscored], axis=0)
    pooled_expr.index.name = "sample_id"

    pooled_clin = pd.concat([b.clinical for b in zscored], axis=0)
    pooled_clin = pooled_clin.loc[pooled_expr.index]  # align row order

    return ICIPoolBundle(
        expr=pooled_expr,
        clinical=pooled_clin,
        cohorts=[b.cohort_id for b in zscored],
        genes=genes,
        per_cohort_hit_rates={b.cohort_id: b.hit_rate for b in bundles},
    )
