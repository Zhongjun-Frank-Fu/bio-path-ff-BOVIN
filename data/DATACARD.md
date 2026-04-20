# Data Card · BOVIN-Pathway Demo

> **Status (M2):** download + loader + alignment + label pipeline wired.
> Real TCGA-COAD is fetched on demand via `make data-coad`; tests run on a
> synthetic Xena-shaped fixture (see `tests/conftest.py`). Once the real
> TSV is downloaded the rows below marked `_populated by M2 · T2.1_` auto-fill
> from `data/checksums.txt`.

## Primary Dataset · TCGA-COAD

| Field | Value |
|---|---|
| Cohort | TCGA-COAD (Colon Adenocarcinoma) |
| Modality | bulk RNA-seq, log2(RSEM+1) |
| Sample size (target) | ~460 primary tumors |
| Source | UCSC Xena (static TSV, not TCGAbiolinks API) |
| Expression URL | https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/HiSeqV2.gz |
| Clinical URL | https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap/COAD_clinicalMatrix |
| Survival URL | https://tcga.xenahubs.net/download/survival/COAD_survival.txt |
| License | NIH / TCGA Data Use Policy — https://www.cancer.gov/tcga/using-data |
| Date downloaded | _populated by M2 · T2.1 (see data/checksums.txt header)_ |
| MD5 (expression) | _populated by M2 · T2.1_ |
| MD5 (clinical) | _populated by M2 · T2.1_ |
| MD5 (survival) | _populated by M2 · T2.1_ |

## Download workflow

```bash
make data-coad        # fetch HiSeqV2.gz + clinical + survival, write checksums
make data-coad-force  # force re-download (overwrites data/raw/*)
```

Raw files land in `data/raw/`, which is gitignored. Only `data/checksums.txt`
and this card are committed.

## Alignment to BOVIN-Pathway

| Field | Value |
|---|---|
| Observable nodes (target surface) | 70 / 82 |
| Aggregate symbols handled | `IFNA/IFNB` → mean of present components; `TRG/TRD` → same |
| Hit-rate DoD | ≥ 0.70 (PLAN §5) |
| Miss policy | NaN column kept in output; downstream model masks via `observable` flag |
| Implementation | `bovin_demo.data.gene_mapping.map_to_pathway_nodes` |

## Label

| Field | Value |
|---|---|
| Kind | **Surrogate** (demo only — not an ICI response label) |
| Formula | `z(CALR) + z(HMGB1) + z(HSPA1A) + z(HSP90AA1) − z(CD47) − z(CD24)` |
| Binarization | Median split → 0/1 |
| Rationale | Peng et al. Fig.1E-F reported CRT/HSP-survival association in TCGA-COAD |
| Missing-gene policy | Silent drop from signature; `LabelReport.genes_missing` records the loss |
| Implementation | `bovin_demo.data.labels.icd_readiness_label` |
| Known limitation | Not a true ICI-response label. **Aim 2 switches to IMvigor210.** |

## Split

| Field | Value |
|---|---|
| Strategy | Stratified 60 / 20 / 20 on the binary label |
| Seed | 42 (override via `BOVIN_SEED` env or `--seed`) |
| Determinism | Identical indices across runs at fixed seed (test guards this) |
| Implementation | `bovin_demo.data.split.stratified_split` |

## Secondary (zero-shot) · TCGA-LUAD

Used only for M6 · T6.4 zero-shot transfer check. Same processing path as COAD.
Download helper will be added alongside M6 (not part of M2 DoD).

## What we don't use (and why)

| Not used | Why |
|---|---|
| IMvigor210 at M0–M6 | Access pending; Aim 2 fold-in per PLAN §9. |
| H&E / WSI | Aim 1.2; multimodal hooks are stubbed in `bovin_demo.data`. |
| Longitudinal on-treatment | Aim 1 baseline-only per PLAN §0. |
| TCGAbiolinks API | Flaky; we pin Xena static TSV for reproducibility. |

## Reproducibility

- Raw data not committed (see `.gitignore`).
- `data/checksums.txt` is committed and **rewritten** by every `make data-coad` run.
- Processed tensors (`data/processed/*.npz`) are regenerable from
  `python -m bovin_demo.cli train` (lands in M4).
- CI tests do not download real data — they use the synthetic fixture in
  `tests/conftest.py` (80 samples × 76 genes, ~92% pathway coverage).
