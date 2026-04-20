#!/usr/bin/env bash
# M2 · T2.1 — download TCGA-COAD RNA-seq + clinical + survival from UCSC Xena.
#
# We deliberately use Xena static files, not TCGAbiolinks API (PLAN §3.1):
# reproducibility trumps convenience. All three files land under data/raw/,
# MD5s are appended to data/checksums.txt, and DATACARD.md references both.
#
# Usage:
#   bash tools/download_tcga_coad.sh            # fetch + checksum (idempotent)
#   bash tools/download_tcga_coad.sh --force    # redownload even if present

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${HERE}/data/raw"
CHK="${HERE}/data/checksums.txt"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

mkdir -p "$RAW"

# Xena static endpoints for TCGA-COAD (sampleMap cohort).
BASE="https://tcga.xenahubs.net/download/TCGA.COAD.sampleMap"
SURV="https://tcga.xenahubs.net/download/survival"

declare -A FILES=(
  ["HiSeqV2.gz"]="${BASE}/HiSeqV2.gz"
  ["COAD_clinicalMatrix"]="${BASE}/COAD_clinicalMatrix"
  ["COAD_survival.txt"]="${SURV}/COAD_survival.txt"
)

fetch() {
  local name="$1" url="$2" dest="${RAW}/$1"
  if [[ -f "$dest" && $FORCE -eq 0 ]]; then
    echo "[skip]    $name — already present (use --force to redownload)"
    return
  fi
  echo "[fetch]   $name ← $url"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
}

for name in "${!FILES[@]}"; do
  fetch "$name" "${FILES[$name]}"
done

# Decompress expression matrix (Xena gz is plain gzip, one tsv inside)
if [[ -f "${RAW}/HiSeqV2.gz" && ! -f "${RAW}/HiSeqV2" ]]; then
  echo "[gunzip]  HiSeqV2.gz → HiSeqV2"
  gunzip -k "${RAW}/HiSeqV2.gz"
fi

# Rewrite checksums.txt — we own this file entirely after download.
{
  echo "# MD5 checksums for raw TCGA tsv (M2 · T2.1)"
  echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for f in HiSeqV2.gz HiSeqV2 COAD_clinicalMatrix COAD_survival.txt; do
    if [[ -f "${RAW}/${f}" ]]; then
      md5=$(md5 -q "${RAW}/${f}" 2>/dev/null || md5sum "${RAW}/${f}" | awk '{print $1}')
      printf "%s  %s\n" "$md5" "$f"
    fi
  done
} > "$CHK"

echo
echo "[done] raw files in ${RAW}:"
ls -lh "${RAW}"
echo
echo "[done] checksums:"
cat "$CHK"
