"""Try different CRT-high/low cutoffs to replicate Peng's Fig.1C direction.

GEPIA2 default uses quartile split (top 25% vs bottom 25%), not median.
Also tries: top 30%/30%, top 40%/40%, log-transformed, removal of recurrent tumors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bovin_demo.data import load_coad
from bovin_demo.tools_inline_logrank import logrank_2group  # noqa


bundle = load_coad("data/raw")
print(f"Cohort: {bundle.n_samples} patients · clinical columns = {bundle.clinical.columns.tolist()[:10]}...")

# Filter to primary tumor only (01) — GEPIA2 excludes metastatic (06, 07)
if "sample_type" in bundle.clinical.columns:
    primary_mask = bundle.clinical["sample_type"].str.contains("Primary Tumor", na=False)
    print(f"Primary-tumor-only samples: {primary_mask.sum()} / {len(primary_mask)}")
else:
    primary_mask = pd.Series(True, index=bundle.expr.index)


from bovin_demo.tools_inline_logrank import logrank_2group


crt = bundle.expr["CALR"]
df = pd.DataFrame({"crt": crt}).join(bundle.survival[["PFI", "PFI.time"]]).dropna()
df = df[df.index.isin(bundle.clinical.index[primary_mask])] if primary_mask.any() else df
print(f"After filtering: {len(df)} patients")

for pct in [0.50, 0.25, 0.30, 0.40]:
    n = len(df)
    hi_thr = df["crt"].quantile(1 - pct)
    lo_thr = df["crt"].quantile(pct)
    hi = df[df["crt"] >= hi_thr]
    lo = df[df["crt"] <= lo_thr]
    hi_t = hi.loc[hi["PFI"] == 1, "PFI.time"].median()
    lo_t = lo.loc[lo["PFI"] == 1, "PFI.time"].median()
    result = logrank_2group(hi["PFI.time"].values, hi["PFI"].values,
                            lo["PFI.time"].values, lo["PFI"].values)
    label = f"top-{int(pct*100)}% vs bot-{int(pct*100)}%"
    direction = "HIGH longer ✅" if hi_t > lo_t else "HIGH shorter ❌"
    print(f"  {label:20s}  n_hi={len(hi):3d}  n_lo={len(lo):3d}  "
          f"median PFI hi={hi_t:4.0f}d  lo={lo_t:4.0f}d  "
          f"χ²={result['chi2']:6.2f}  p={result['p']:.4f}  [{direction}]")
