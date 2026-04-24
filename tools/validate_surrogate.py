"""Empirical sanity checks for the surrogate ICD-readiness label.

Runs 4 independent probes that should pass if the signature captures
real biology (not just noise). Each check prints pass/fail verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bovin_demo.data import icd_readiness_label, load_coad
from bovin_demo.data.labels import (
    NEGATIVE_GENES,
    POSITIVE_GENES,
    icd_readiness_signature,
)


def banner(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def probe_1_damp_coregulation(bundle):
    banner("PROBE 1 · Do the 4 DAMP genes positively co-vary?")
    print("Biology: DAMPs are stress-response genes → should rise together when ICD fires.")
    df = bundle.expr[list(POSITIVE_GENES)]
    corr = df.corr()
    print(f"\nPearson correlation matrix (4 DAMPs):")
    print(corr.round(3).to_string())
    # Average off-diagonal
    off = (corr.sum().sum() - 4) / (4 * 3)
    print(f"\nMean off-diagonal r = {off:.3f}")
    verdict = "✅ PASS (r > 0.2)" if off > 0.2 else "⚠ WEAK"
    print(f"Verdict: {verdict}")


def probe_2_evasion_coregulation(bundle):
    banner("PROBE 2 · Do CD47 and CD24 positively co-vary?")
    print("Biology: both are 'don't-eat-me' signals; tumors often co-express.")
    r = bundle.expr[list(NEGATIVE_GENES)].corr().iloc[0, 1]
    print(f"\nPearson r (CD47, CD24) = {r:.3f}")
    verdict = "✅ PASS" if r > 0.0 else "⚠ UNEXPECTED (they should co-vary)"
    print(f"Verdict: {verdict}")


def probe_3_signature_vs_immune_markers(bundle):
    banner("PROBE 3 · ICD-ready signature vs independent immune-activity markers")
    print("Claim: if signature captures real ICD, it should correlate with")
    print("       downstream immune activity (CD8A, GZMB, IFNG).")
    score, _ = icd_readiness_signature(bundle.expr)
    immune = ["CD8A", "GZMB", "IFNG", "PRF1"]
    print(f"\nCorrelation of signature score vs independent immune markers:")
    for g in immune:
        if g not in bundle.expr.columns:
            print(f"  {g:6s} — not in matrix, skip")
            continue
        r = np.corrcoef(score, bundle.expr[g])[0, 1]
        verdict = "✅" if r > 0.1 else ("~" if r > 0 else "❌")
        print(f"  {g:6s}  r = {r:+.3f}   {verdict}")


def probe_4_survival_association(bundle):
    banner("PROBE 4 · Signature high vs low → survival difference?")
    if bundle.survival is None:
        print("No survival file in this bundle. Skip.")
        return
    surv = bundle.survival
    # Find OS columns (names vary across Xena releases)
    os_col = next((c for c in surv.columns if c.upper() in ("OS", "VITAL_STATUS")), None)
    time_col = next((c for c in surv.columns if "OS.TIME" in c.upper() or c == "OS.time"), None)
    if os_col is None or time_col is None:
        print(f"Survival columns not found. Available: {list(surv.columns)[:8]}")
        return

    # Join signature score with survival
    score, _ = icd_readiness_signature(bundle.expr)
    df = pd.DataFrame({"score": score}).join(surv[[os_col, time_col]])
    df = df.dropna()
    df["high"] = df["score"] > df["score"].median()
    # Simple comparison: median OS.time of events in high vs low
    events = df[df[os_col] == 1]
    print(f"\nCohort: {len(df)} with paired signature + OS")
    print(f"Events (deaths): {len(events)}")
    if len(events) < 20:
        print("Too few events, skip.")
        return
    hi_t = events.loc[events["high"], time_col].median()
    lo_t = events.loc[~events["high"], time_col].median()
    print(f"Median OS time (deaths) — high-signature: {hi_t:.0f} days")
    print(f"Median OS time (deaths) — low-signature:  {lo_t:.0f} days")
    # Not a proper log-rank, but directional check
    if hi_t > lo_t:
        print("✅ Higher signature → longer survival (directionally consistent with Peng Fig.1E-F)")
    else:
        print("❌ Higher signature → shorter survival (inconsistent)")


def probe_5_label_pan_cancer_stability(bundle):
    banner("PROBE 5 · Pan-cancer stability · signature on LUAD")
    from pathlib import Path
    luad_dir = Path("data/raw_luad")
    if not luad_dir.exists():
        print(f"LUAD raw dir missing ({luad_dir}); skip.")
        return

    luad = load_coad(luad_dir)
    luad_score, lrep = icd_readiness_signature(luad.expr)
    print(f"\nCOAD signature distribution:  mean={0:.2f}  std={1.0:.2f} (by construction of z-score)")
    print(f"LUAD signature distribution:  mean={luad_score.mean():+.2f}  std={luad_score.std():.2f}")
    print(f"LUAD signature range: [{luad_score.min():+.2f}, {luad_score.max():+.2f}]")
    print("If the signature collapses on LUAD (e.g. std < 0.5), it's COAD-specific.")
    verdict = "✅ PASS" if luad_score.std() > 1.0 else "⚠ COLLAPSED on LUAD"
    print(f"Verdict: {verdict}")


def main() -> None:
    bundle = load_coad("data/raw")
    print(f"TCGA-COAD loaded: {bundle.n_samples} patients × {bundle.n_genes} genes")

    probe_1_damp_coregulation(bundle)
    probe_2_evasion_coregulation(bundle)
    probe_3_signature_vs_immune_markers(bundle)
    probe_4_survival_association(bundle)
    probe_5_label_pan_cancer_stability(bundle)


if __name__ == "__main__":
    main()
