#!/usr/bin/env bash
# A2-M1 · T1.1 — download 6-cohort ICI pool for Aim 2 real-RECIST training.
#
# Cohorts (Tier A per AIM2-TRAINING-PLAN.md §2.1):
#   1. Riaz 2017        GSE91061       melanoma,  nivo,         N=65
#   2. Hugo 2016        GSE78220       melanoma,  pembro/nivo,  N=27
#   3. Gide 2019        PRJEB23709     melanoma,  combo,        N=74   (via cBioPortal mel_iatlas_gide_2019, iAtlas-harmonized plain TSV — R1 mitigation)
#   4. Hammerman 2020   GSE165278      melanoma,  ipi,          N=21
#   5. Cloughesy 2019   GSE121810      GBM,       neoadj pembro, N=29
#   6. Seo 2020         GSE165252      EAC,       atezolizumab, N=40
#
# Each cohort lands under data/raw_ici/<id>/ with expression + (when GEO) series_matrix.
# SHA256s collected at data/raw_ici/checksums.txt. Idempotent — re-running skips existing files.
#
# Usage:
#   bash tools/download_ici_pool.sh              # fetch + sha256 (idempotent)
#   bash tools/download_ici_pool.sh --force      # redownload even if present
#   bash tools/download_ici_pool.sh --only seo   # fetch one cohort only

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${HERE}/data/raw_ici"
CHK="${RAW}/checksums.txt"
FORCE=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --only)  ONLY="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RAW"

# ---------------------------------------------------------------------------
# file registry — one line per download target.
# format: cohort_id|subpath|url
# ---------------------------------------------------------------------------
FILES=(
  # ── Riaz GSE91061 ──
  "riaz_gse91061|GSE91061_BMS038109Sample.hg19KnownGene.raw.csv.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE91nnn/GSE91061/suppl/GSE91061_BMS038109Sample.hg19KnownGene.raw.csv.gz"
  "riaz_gse91061|GSE91061_series_matrix.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE91nnn/GSE91061/matrix/GSE91061_series_matrix.txt.gz"
  # Riaz clinical + sample tables — not on GEO, served from the authors' own GitHub repo.
  "riaz_gse91061|bms038_clinical_data.csv|https://raw.githubusercontent.com/riazn/bms038_analysis/master/data/bms038_clinical_data.csv"
  "riaz_gse91061|SampleTableCorrected.9.19.16.csv|https://raw.githubusercontent.com/riazn/bms038_analysis/master/data/SampleTableCorrected.9.19.16.csv"

  # ── Hugo GSE78220 ──
  "hugo_gse78220|GSE78220_PatientFPKM.xlsx|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78220/suppl/GSE78220_PatientFPKM.xlsx"
  "hugo_gse78220|GSE78220_series_matrix.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78220/matrix/GSE78220_series_matrix.txt.gz"

  # ── Gide PRJEB23709 (via cBioPortal iAtlas-harmonized TSV — R1 mitigation).
  #    ORCESTRA .rds initially considered but rejected — it's a MultiAssayExperiment
  #    S4 object that pyreadr can't parse (needs R / rpy2). cBioPortal ships plain TSV.
  "gide_prjeb23709|data_mrna_seq_expression.txt|https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/mel_iatlas_gide_2019/data_mrna_seq_expression.txt"
  "gide_prjeb23709|data_clinical_patient.txt|https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/mel_iatlas_gide_2019/data_clinical_patient.txt"
  "gide_prjeb23709|data_clinical_sample.txt|https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/mel_iatlas_gide_2019/data_clinical_sample.txt"

  # ── Hammerman GSE165278 ──
  "hammerman_gse165278|GSE165278_TPM_original_23k_genes.xlsx|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165278/suppl/GSE165278_TPM_original_23k_genes.xlsx"
  "hammerman_gse165278|GSE165278_series_matrix.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165278/matrix/GSE165278_series_matrix.txt.gz"

  # ── Cloughesy GSE121810 ──
  "cloughesy_gse121810|GSE121810_Prins.PD1NeoAdjv.Jul2018.HUGO.PtID.xlsx|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE121nnn/GSE121810/suppl/GSE121810_Prins.PD1NeoAdjv.Jul2018.HUGO.PtID.xlsx"
  "cloughesy_gse121810|GSE121810_series_matrix.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE121nnn/GSE121810/matrix/GSE121810_series_matrix.txt.gz"

  # ── Seo GSE165252 ──
  "seo_gse165252|GSE165252_norm.cnt_PERFECT.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165252/suppl/GSE165252_norm.cnt_PERFECT.txt.gz"
  "seo_gse165252|GSE165252_vst_PERFECT.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165252/suppl/GSE165252_vst_PERFECT.txt.gz"
  "seo_gse165252|GSE165252_series_matrix.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165252/matrix/GSE165252_series_matrix.txt.gz"

  # ── Sade-Feldman GSE120575 (scRNA — external validation only, not in pool) ──
  #    121 MB expression + 81 KB cell metadata. Aggregation to patient pseudobulk
  #    happens in bovin_demo/data/sade_loader.py, not here.
  "sade_feldman_gse120575|GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz"
  "sade_feldman_gse120575|GSE120575_patient_ID_single_cells.txt.gz|https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/GSE120575_patient_ID_single_cells.txt.gz"
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

fetch_one() {
  local cohort="$1" name="$2" url="$3"
  local dir="${RAW}/${cohort}"
  local dest="${dir}/${name}"
  mkdir -p "$dir"
  if [[ -f "$dest" && $FORCE -eq 0 ]]; then
    echo "[skip]   ${cohort}/${name} — already present"
    return
  fi
  echo "[fetch]  ${cohort}/${name}"
  echo "         ← ${url}"
  # -L follow redirects (Zenodo uses 301), -f fail on HTTP error, retry for flaky ENA
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$dest.partial" "$url"
  mv "$dest.partial" "$dest"
  local sz
  sz=$(ls -l "$dest" | awk '{print $5}')
  echo "         → ${sz} bytes"
}

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
for line in "${FILES[@]}"; do
  IFS='|' read -r cohort name url <<< "$line"
  if [[ -n "$ONLY" && "$cohort" != *"$ONLY"* ]]; then
    continue
  fi
  fetch_one "$cohort" "$name" "$url"
done

# ---------------------------------------------------------------------------
# rewrite checksums.txt — single source of truth for raw ICI pool
# ---------------------------------------------------------------------------
{
  echo "# SHA256 checksums for BOVIN-Bench Tier A ICI pool (A2-M1 · T1.1)"
  echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# hash <two-spaces> cohort/filename"
  for line in "${FILES[@]}"; do
    IFS='|' read -r cohort name _ <<< "$line"
    local_path="${RAW}/${cohort}/${name}"
    if [[ -f "$local_path" ]]; then
      hash=$(sha256_of "$local_path")
      printf "%s  %s/%s\n" "$hash" "$cohort" "$name"
    fi
  done
} > "$CHK"

echo
echo "[done] raw ICI pool tree:"
( cd "$RAW" && find . -type f -name '*.gz' -o -name '*.xlsx' -o -name '*.rds' -o -name '*.txt' 2>/dev/null | sort | xargs -I{} sh -c 'ls -lh "{}" | awk "{print \$5, \$9}"' )
echo
echo "[done] checksums @ ${CHK}"
wc -l "$CHK"
