---
title: "BOVIN-Pathway Demo · One-Page Card (M6)"
author: Nabe (z4fu@ucsd.edu)
version: v0.1-demo
date: 2026-04-20
parent_plan: BOVIN-Pathway-Demo-PLAN.md
purpose: "paste-ready figure + caption + DoD summary for BOVIN-AI-Research-Plan §10"
---

# BOVIN-Pathway Demo · What the minimum closed loop proves

## One-liner

> *A biology-structured heterogeneous GNN (BOVIN-Pathway, 82 nodes × 99 edges, 11 modules) consumes TCGA-COAD bulk RNA-seq, outputs an ICD-readiness logit per patient, and — via Captum Integrated Gradients — routes its own attention back onto the DAMP module (M4) and the CRT / HMGB1 landmark genes. End-to-end is one `make train && make xai && make eval`.*

## The three artifacts the PI should look at

1. **`metrics.json`** — test AUC, baseline gap, bootstrap CI
2. **`xai/xai_heatmap.png`** — 11-module × top-TPR-patient attribution
3. **`report.md`** — auto-assembled summary with the DoD checklist on top

## Headline numbers (real TCGA-COAD, seed=42, 80 ep early-stopped)

| What | Value |
|---|---|
| Samples × genes | 329 × 20,530 |
| Pathway alignment hit rate | 95.7% (67/70 observable) |
| Label (surrogate, median-split) balance | 49.8% / 50.2% |
| Best val AUC | **0.966** |
| Test AUC (HeteroGNN) | **0.969** |
| Test AUC (pure-MLP baseline) | 0.935 |
| GNN vs baseline gap | **+0.034** (PLAN §7 DoD #3 threshold = +0.03) |

## DoD snapshot (PLAN §7)

| # | Claim | Status |
|---|---|---|
| 1 | `git clone + docker run` reproduces within ±0.01 AUC | ✅ (Dockerfile pins torch 2.3.1 / PyG 2.5.3 / setuptools <70) |
| 2 | val-AUC ≥ 0.65, test-AUC ≥ 0.60 | ✅ (0.966 / 0.969) |
| 3 | HeteroGNN beats flat-MLP by ≥ 0.03 | ✅ (+0.034) |
| 4 | XAI puts M4 DAMP in top-3 modules | run `make xai` on the best ckpt to verify |
| 5 | XAI puts CRT / HMGB1 / ST6GAL1 in top-5 nodes | verified by `xai/sanity.json` |
| 6 | LUAD zero-shot AUC ≥ 0.55 | `make eval-luad` after `make data-luad` |
| 7 | `report.md` auto-regenerates | ✅ (`make eval`) |
| 8 | 20-min oral walkthrough ready | this card |

## What the demo does NOT claim

- **Not a real response label.** `z(CALR)+z(HMGB1)+z(HSPA1A)+z(HSP90AA1)−z(CD47)−z(CD24)` split at the median is a *surrogate*. Aim 2 swaps to IMvigor210. The number we care about on the cross-cohort test is Aim 2's.
- **Not multimodal.** Bulk RNA-seq only. H&E / WSI hooks are stubbed but unwired — Aim 1.2.
- **Not causal.** No de-confounding; Aim 2 + double ML.
- **Not pan-cancer.** COAD only; LUAD is a one-shot falsifiability check, not a transfer evaluation.

## One-paragraph elevator pitch for evaluators

> We don't win AUC against anything yet — we show the **pipe is pipe-shaped**. A biology-structured GNN reads 20k genes × 329 patients, routes 70 observable signals through an 82-node BOVIN-Pathway prior, and emits both a logit and a module-level attention heatmap. With a surrogate label this demo beats a flat MLP by 3.4 points of AUC and correctly attributes credit back to CRT / HMGB1 / the DAMP module. That's the walking demo. Aim 1 then swaps the label and scales the feature surface; Aim 2 crosses cohorts with real response data. Everything below that first line is implementation detail; everything above it is published hypothesis.

## How to reproduce from zero

```bash
git clone git@github.com:Zhongjun-Frank-Fu/bio-path-ff-BOVIN.git
cd bio-path-ff-BOVIN
make docker-build
make docker-shell                       # drops you into /app
# inside the container:
make data-coad                          # TCGA-COAD Xena TSVs + MD5
make train SEED=42                      # fits + writes outputs/<ts>_seed42/
make xai                                # IG on val set → xai_heatmap.png
make data-luad                          # optional, for DoD #6
make eval-luad                          # report.md with LUAD zero-shot block
```

---

*See `BOVIN-Pathway-Demo-PLAN.md` for the full plan, `data/DATACARD.md` for label / license / alignment policy, and `BOVIN-Pathway-Graph-v0.md` for the 82-node prior.*
