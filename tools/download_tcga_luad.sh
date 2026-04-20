#!/usr/bin/env bash
# M6 · T6.4 — TCGA-LUAD RNA-seq for the zero-shot transfer check.
# Same Xena sampleMap cohort as COAD, same format → the COAD loader reads
# either with no code change (tolerates ``LUAD_clinicalMatrix``).
#
# Usage:
#   bash tools/download_tcga_luad.sh            # fetch if absent
#   bash tools/download_tcga_luad.sh --force    # redownload

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${HERE}/data/raw_luad"
CHK="${HERE}/data/checksums_luad.txt"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

mkdir -p "$RAW"

BASE="https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap"
SURV="https://tcga.xenahubs.net/download/survival"

declare -A FILES=(
  ["HiSeqV2.gz"]="${BASE}/HiSeqV2.gz"
  ["LUAD_clinicalMatrix"]="${BASE}/LUAD_clinicalMatrix"
  ["LUAD_survival.txt"]="${SURV}/LUAD_survival.txt"
)

fetch() {
  local name="$1" url="$2" dest="${RAW}/$1"
  if [[ -f "$dest" && $FORCE -eq 0 ]]; then
    echo "[skip]    $name — already present"
    return
  fi
  echo "[fetch]   $name ← $url"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
}

for name in "${!FILES[@]}"; do
  fetch "$name" "${FILES[$name]}"
done

if [[ -f "${RAW}/HiSeqV2.gz" && ! -f "${RAW}/HiSeqV2" ]]; then
  echo "[gunzip]  HiSeqV2.gz → HiSeqV2"
  gunzip -k "${RAW}/HiSeqV2.gz"
fi

{
  echo "# MD5 checksums for raw TCGA-LUAD tsv (M6 · T6.4)"
  echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for f in HiSeqV2.gz HiSeqV2 LUAD_clinicalMatrix LUAD_survival.txt; do
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
