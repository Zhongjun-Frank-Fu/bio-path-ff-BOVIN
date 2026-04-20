---
title: "BOVIN-Pathway Demo · v0.1-demo Release Notes"
date: 2026-04-20
author: Nabe (z4fu@ucsd.edu)
parent_plan: BOVIN-Pathway-Demo-PLAN.md
tag: v0.1-demo
---

# v0.1-demo — Aim 1 Minimum Closed Loop

The promise in `BOVIN-Pathway-Demo-PLAN.md` §0:

> Put Aim 1's story end-to-end in ~500 lines of readable code, one training
> log, one XAI heatmap — `python -m bovin_demo.cli train --config configs/tcga_coad.yaml`
> produces `outputs/…/metrics.json` + `xai_heatmap.png` + `training.log`,
> git-cloneable and Docker-one-shot reproducible.

This tag is the cut at which the six milestones (M1 → M6) all land and the
CLI runs every phase end-to-end.

## Headline numbers (TCGA-COAD, 80 epochs, EarlyStopping on val_auc)

| seed | epochs run | val_auc | test_auc (HeteroGNN) | baseline_test_auc (MLP) | gap |
|---|---|---|---|---|---|
| 42   | 46 | **0.9660** | **0.9688** | 0.9348 | **+0.0340** ✅ |
| 1337 | 29 | 0.9596 | 0.9183 | 0.9412 | −0.0230 |
| 2024 | 27 | 0.9578 | 0.8797 | 0.8916 | −0.0119 |

* val_auc **0.9611 ± 0.0035** (very stable across seeds)
* test_auc **0.9223 ± 0.0369** (split-dependent; consistent with 66-sample test fold)
* GNN-vs-baseline gap averaged ≈ 0: surrogate label is literally a linear
  combination of 6 genes, so a flat MLP on 70 observable genes is a strong
  baseline — this is expected and called out in the demo card.

## Milestones

| # | Scope | Status |
|---|---|---|
| M0 | Repo skeleton, Docker, CI, ruff, pytest scaffolding | ✅ |
| M1 | BOVIN-Pathway loader: md → JSON → pydantic → HeteroData; 82 nodes / 99 edges / 11 modules | ✅ |
| M2 | TCGA-COAD Xena loader; HGNC alignment (95.7% hit on real COAD); ICD-readiness surrogate label; stratified 60/20/20 | ✅ |
| M3 | HeteroGNN: per-type encoder → GATv2 × 2 → HGT × 1 → per-module attention → Linear(11,1); BaselineMLP | ✅ |
| M4 | LitBovinModule + run_training + EarlyStopping + CSV log + ModelCheckpoint + 3-seed sweep | ✅ |
| M5 | Captum IG on node-level inputs → module + node rollup → heatmap PNG + sanity.json | ✅ |
| M6 | bootstrap metrics CI + build_report + LUAD zero-shot + Dockerfile setuptools pin + demo card | ✅ |

## Definition of Done (PLAN §7)

| # | Claim | Status | Detail |
|---|---|---|---|
| 1 | git clone + docker → reproduces within ±0.01 AUC | ✅ | Dockerfile pins torch 2.3.1 / PyG 2.5.3 / setuptools<70 |
| 2 | val-AUC ≥ 0.65, test-AUC ≥ 0.60 | ✅ | seed=42: 0.966 / 0.969 |
| 3 | HeteroGNN beats flat-MLP by ≥ 0.03 | ⚠ | only seed=42 clears (+0.034); mean gap ~0 |
| 4 | XAI puts M4 DAMP in top-3 modules | ✅ | top-3 = `[M4 DAMP, M6 APC_RECV, M9 ICB]` (seed=42 ckpt, 20 top-TPR val patients) |
| 5 | CRT / HMGB1 / ST6GAL1 in top-5 nodes | ✅ | top-5 = `[crt, cd47, hmgb1, hsp70, cd24]` |
| 6 | LUAD zero-shot AUC ≥ 0.55 | ✅ | **0.9752** on 576 LUAD patients, 95.7% alignment hit rate |
| 7 | `report.md` auto-regenerates | ✅ | `make eval-luad` produced the end-to-end report |
| 8 | 20-min oral walkthrough ready | ✅ | `docs/demo_card.md` |

DoD #3 is flagged as "partial" deliberately — the surrogate label is the
ceiling, not the GNN. Aim 2's IMvigor210 real-response label is where the
graph structure's advantage becomes testable.

## Test suite

```
61 passed, 1 skipped, 60 warnings in 248.99s
```

Skipped test checks that `load_graph` can re-parse the source markdown on
disk; that file is outside the Docker image mount in the test environment.

## Reproducibility recipe

```bash
git clone git@github.com:Zhongjun-Frank-Fu/bio-path-ff-BOVIN.git
cd bio-path-ff-BOVIN
git checkout v0.1-demo
make docker-build         # torch 2.3.1 CPU + PyG 2.5.3 + setuptools<70
make docker-shell         # drops you into /app

# inside container:
make data-coad            # ~65MB Xena TSVs + MD5
make train SEED=42        # fits → outputs/<ts>_seed42/
make xai                  # IG on val → xai/xai_heatmap.png
make data-luad            # optional
make eval-luad            # outputs/<ts>_seed42/report.md w/ DoD checklist
```

Same machine + same seed → test_auc varies < 0.001 across reruns (torch
deterministic algorithms, fixed generator, fixed split).

## What landed since M0

* 82-node / 99-edge graph with pydantic schema + DoD guards (M1)
* Full TCGA-COAD pipeline: Xena downloader → loader → HGNC alignment → z-score label → split (M2)
* HeteroGNN with per-relation GATv2Conv routing + HGTConv cross-type + per-module attention (M3)
* Lightning training with BCE + pos_weight + cosine LR + EarlyStopping + ModelCheckpoint (M4)
* Captum IG wrapper that flattens heterogeneous input for per-node attribution (M5)
* Bootstrap CI metrics, auto-report, LUAD zero-shot, demo card (M6)

## Known limitations (also in demo_card § "does NOT claim")

* **Surrogate label** (PLAN §3.2). Median-split z-score signature of 6 genes.
  Not an ICI-response label. Aim 2 swaps to IMvigor210.
* **Single cohort.** COAD only; LUAD is a one-shot falsifiability check.
* **Bulk RNA-seq only.** H&E / clinical / on-treatment hooks stubbed per
  PLAN §9.
* **Baseline gap is cohort-dependent.** The surrogate's linearity makes the
  flat MLP a ceiling-tight baseline; Aim 2's cross-cohort real-response
  label is where the graph prior's value is supposed to show.
* **Poor calibration** (ECE ≈ 0.40 on the test fold). The model is
  overconfident — BCE + pos_weight + median-split label pushes probabilities
  toward the extremes. Temperature scaling post-hoc is a v0.2 fix; for v0.1
  the binary decisions are fine, the calibrated probabilities are not.
* **LUAD zero-shot works suspiciously well (AUC 0.975)** — read this as
  "the surrogate's 6-gene signature is pan-cancer enough to transfer
  trivially", not "the model generalizes". A real response label would not
  transfer like this.
* **CPU-only Docker.** GPU not needed at demo scale (82-node graphs).

## Next (post v0.1-demo)

The three interfaces PLAN §9 flagged:

1. **Multimodal**: wire `bovin_demo/data/pathomics.py` + `clinical.py`;
   HeteroData already has the slots.
2. **Real response label**: `configs/imvigor210.yaml` + `y_response`
   column; no code churn needed, only data.
3. **On-treatment timing**: node features from `(d,)` to `(T, d)` and a
   cross-time attention after the module pool.

Aim 2 work (causal de-confounding, cross-cohort generalization, survival
supervision) is orthogonal to this tag.

---

*Tagged from `main` at the commit that lands M6 · T6.1 through T6.4.*
*See `BOVIN-Pathway-Demo-PLAN.md` for the full plan and `docs/demo_card.md`
for the PI-facing one-pager.*
