"""Try to replicate Peng 2025 Fig.1 claims on the same TCGA-COAD data.

Peng 2025 Nat Commun 17:301 reports (via GEPIA2):
  Fig.1C  TCGA-COAD · high CRT → longer PFI (log-rank p=0.00033)
  Fig.1D  TCGA-LUAD · high CRT → longer PFI (log-rank p=0.0043)
  Fig.1E  TCGA-COAD · CRT vs CD3E correlation, p=0.003024
  Fig.1F  TCGA-COAD · CRT vs CD8A correlation, p=0.007492

We test the 3 claims (C, E, F) with our Xena-derived data. Our earlier
Probe 4 used OS, not PFI, and used the 6-gene composite signature rather
than CRT alone — both are misalignments with Peng's actual claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from bovin_demo.data import load_coad


def banner(title: str) -> None:
    print(f"\n{'='*75}\n{title}\n{'='*75}")


def logrank_2group(time_a, event_a, time_b, event_b) -> dict:
    """Simple log-rank test (no lifelines dep). Returns chi2, p."""
    from scipy.stats import chi2

    # Combined sorted event times from both groups
    ta = np.asarray(time_a, dtype=float)
    ea = np.asarray(event_a, dtype=int)
    tb = np.asarray(time_b, dtype=float)
    eb = np.asarray(event_b, dtype=int)
    all_events = np.sort(np.unique(np.concatenate([ta[ea == 1], tb[eb == 1]])))

    obs_a = exp_a = var_a = 0.0
    for t in all_events:
        at_risk_a = (ta >= t).sum()
        at_risk_b = (tb >= t).sum()
        at_risk = at_risk_a + at_risk_b
        if at_risk == 0:
            continue
        d_a = ((ta == t) & (ea == 1)).sum()
        d_b = ((tb == t) & (eb == 1)).sum()
        d = d_a + d_b
        if d == 0:
            continue
        e_a = at_risk_a * d / at_risk
        obs_a += d_a
        exp_a += e_a
        var_a += (at_risk_a * at_risk_b * d * (at_risk - d)) / (at_risk ** 2 * (at_risk - 1)) if at_risk > 1 else 0.0

    if var_a == 0:
        return {"chi2": 0.0, "p": 1.0, "obs_a": obs_a, "exp_a": exp_a}
    z2 = (obs_a - exp_a) ** 2 / var_a
    return {"chi2": z2, "p": float(chi2.sf(z2, 1)), "obs_a": obs_a, "exp_a": exp_a}


def peng_fig1c(bundle) -> None:
    banner("PENG Fig.1C replication · COAD CRT-high vs CRT-low · PFI log-rank")
    print("Peng reports: high-CRT colon cancer → longer PFI, p = 0.00033")

    if bundle.survival is None:
        print("no survival file; skip")
        return

    surv = bundle.survival
    # Xena survival columns: OS, OS.time, DSS, DSS.time, DFI, DFI.time, PFI, PFI.time
    if "PFI" not in surv.columns or "PFI.time" not in surv.columns:
        print(f"no PFI columns. available: {list(surv.columns)[:10]}")
        return

    crt = bundle.expr["CALR"]  # Peng uses CRT = CALR mRNA
    df = pd.DataFrame({"crt": crt}).join(surv[["PFI", "PFI.time"]]).dropna()
    median_crt = df["crt"].median()
    hi = df[df["crt"] > median_crt]
    lo = df[df["crt"] <= median_crt]
    print(f"\nCohort: {len(df)} with matched CRT + PFI")
    print(f"Median CRT = {median_crt:.3f}  (log2(RSEM+1))")
    print(f"  HIGH group: n={len(hi):<4d}  PFI events = {int(hi['PFI'].sum())}")
    print(f"  LOW  group: n={len(lo):<4d}  PFI events = {int(lo['PFI'].sum())}")

    # Median PFI time among events (direction check)
    hi_t = hi.loc[hi["PFI"] == 1, "PFI.time"].median()
    lo_t = lo.loc[lo["PFI"] == 1, "PFI.time"].median()
    print(f"\nMedian PFI time (among events):")
    print(f"  HIGH-CRT: {hi_t:.0f} days")
    print(f"  LOW-CRT:  {lo_t:.0f} days")

    result = logrank_2group(
        hi["PFI.time"].values, hi["PFI"].values,
        lo["PFI.time"].values, lo["PFI"].values,
    )
    direction = "HIGH longer ✅" if hi_t > lo_t else "HIGH shorter ❌"
    print(f"\nLog-rank χ² = {result['chi2']:.3f}   p = {result['p']:.4f}   [{direction}]")
    print(f"Peng's  χ² = (n/a)             p = 0.00033")


def peng_fig1e_f(bundle) -> None:
    banner("PENG Fig.1E/F replication · COAD CRT vs CD3E / CD8A correlation")
    print("Peng reports (via GEPIA2):")
    print("  Fig.1E  CRT vs CD3E  p = 0.003024")
    print("  Fig.1F  CRT vs CD8A  p = 0.007492")

    crt = bundle.expr["CALR"]
    for gene in ["CD3E", "CD8A"]:
        target = bundle.expr[gene]
        # Use log values directly (same as Xena's log2(RSEM+1)); Peng used TPM
        r, p = pearsonr(crt, target)
        direction = "positive ✅" if r > 0 else "negative ❌"
        print(f"\n  CRT vs {gene}:  r = {r:+.3f}   p = {p:.4g}   [{direction}]")


def composite_vs_crt_alone(bundle) -> None:
    """Show the 6-gene composite signature's direction differs from CRT alone."""
    banner("WHY our Probe 4 failed · composite signature vs CRT alone")

    if bundle.survival is None or "PFI" not in bundle.survival.columns:
        print("no PFI; skip")
        return

    from bovin_demo.data.labels import icd_readiness_signature

    sig, _ = icd_readiness_signature(bundle.expr)
    df = pd.DataFrame({"sig": sig, "crt": bundle.expr["CALR"]}).join(
        bundle.survival[["PFI", "PFI.time"]]
    ).dropna()

    for col, name in [("crt", "CRT alone (Peng's thing)"),
                      ("sig", "composite z-signature (our label)")]:
        med = df[col].median()
        hi = df[df[col] > med]
        lo = df[df[col] <= med]
        hi_t = hi.loc[hi["PFI"] == 1, "PFI.time"].median() if (hi["PFI"] == 1).any() else float("nan")
        lo_t = lo.loc[lo["PFI"] == 1, "PFI.time"].median() if (lo["PFI"] == 1).any() else float("nan")
        r = logrank_2group(hi["PFI.time"].values, hi["PFI"].values,
                           lo["PFI.time"].values, lo["PFI"].values)
        direction = "HIGH longer ✅" if hi_t > lo_t else "HIGH shorter ❌"
        print(f"\n{name}:")
        print(f"  median PFI time (events) · HIGH={hi_t:.0f}d  LOW={lo_t:.0f}d  [{direction}]")
        print(f"  log-rank χ²={r['chi2']:.3f}  p={r['p']:.4f}")


def main() -> None:
    bundle = load_coad("data/raw")
    print(f"TCGA-COAD via UCSC Xena: {bundle.n_samples} patients × {bundle.n_genes} genes")
    if bundle.survival is not None:
        print(f"Survival columns: {list(bundle.survival.columns)}")

    peng_fig1c(bundle)
    peng_fig1e_f(bundle)
    composite_vs_crt_alone(bundle)


if __name__ == "__main__":
    main()
