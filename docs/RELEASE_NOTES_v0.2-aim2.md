# v0.2-aim2 · Release Notes

**Date**: 2026-04-23
**Parent plan**: [`AIM2-TRAINING-PLAN.md`](../AIM2-TRAINING-PLAN.md) (v2 of Aim 2 plan — signed off 2026-04-22)
**Predecessor**: v0.1-demo (Aim 1 · TCGA-COAD surrogate closed loop)

## What v0.2-aim2 adds

| Component | Status |
|---|---|
| `data/raw_ici/` · 6-cohort Tier A ICI pool · 173 MB raw + SHA256 | ✅ |
| `tools/download_ici_pool.sh` (idempotent, 6 cohorts via GEO + cBioPortal) | ✅ |
| `tools/build_bovin_gene_aliases.py` → `bovin_demo/data/static/bovin_gene_aliases.csv` (72 symbols · 100% Entrez · 97.2% Ensembl) | ✅ |
| `bovin_demo/data/ici_loader.py` · `ICIBundle` + `ICIPoolBundle` + 6 per-cohort adapters + `load_ici_pool` | ✅ |
| `bovin_demo/data/labels.py::recist_binary_label()` · CR/PR → 1, SD/PD → 0 | ✅ |
| `bovin_demo/data/split.py::leave_one_cohort_out()` · plan §2.4 primary eval | ✅ |
| `bovin_demo/data/build.py::build_data_and_split()` · shared dispatch helper | ✅ |
| `configs/ici_pool.yaml` · inherits `default.yaml` + overrides data/label/split/eval | ✅ |
| `bovin_demo/train/loop.py` · ICI pool + LOCO branches wired through `run_training` | ✅ |
| `tools/run_ici_loco.py` · 3-seed × 5-fold driver | ✅ |
| `tools/merge_loco_summaries.py` · 3-seed aggregation + DoD flags | ✅ |
| `bovin_demo/xai/runner.py` · ICI pool support via shared build helper | ✅ |
| `bovin_demo/eval/hypothesis_tests.py` · H1–H4 pass/fail evaluator | ✅ |
| `bovin_demo/eval/aim2_report.py` · `outputs/AIM2_REPORT.md` generator | ✅ |
| `bovin_demo/data/sade_loader.py` · Sade-Feldman scRNA → patient pseudobulk (streaming, BOVIN-filtered) | ✅ |
| `bovin_demo/eval/external_transfer.py` · zero-shot forward on Sade pseudobulk | ✅ |
| `bovin_demo/train/loop.py` · `_train_baseline_rf` + `TrainingResult.rf_test_auc` (in-loop, ~20 lines) | ✅ |
| `tools/run_rf_loco.py` · standalone RF LOCO sweep (skips GNN retraining) | ✅ |
| `tools/combine_gnn_rf.py` · merges 2 summaries into `loco_3seed_with_rf.json` | ✅ |
| `tests/test_ici_loader.py` (15 tests) + LOCO tests in `tests/test_data.py` (+6) | ✅ · 82+ passed, 0 failed |
| `notebooks/03_ici_pool_inspector.ipynb` · per-cohort + pooled data walkthrough | ✅ · code verified; outputs render in JupyterLab |

## Headline findings — honest negative result

**0 of 4 pre-registered hypotheses pass.**

| Hypothesis | Result |
|---|---|
| H1 · ICD axis IG + | ❌ 2/4 DAMP nodes match expected sign |
| H2 · "don't-eat-me" IG − | ❌ 1/3 nodes match |
| H3 · GNN − MLP ≥ +0.03 | ❌ LOCO gap = **−0.030** (95% CI excludes 0, negative direction) |
| H4 · LOCO ≥ 0.60 & worst ≥ 0.55 | ❌ mean **0.568**, worst (Hugo) **0.407** |

**External validation (A2-M5)**: Sade-Feldman scRNA pseudobulk (19 patients, pre-tx). Pool-trained model scores **AUC 0.311** [0.086, 0.590] — below the 0.5 chance line, consistent with the plan's pre-registered CD45+-enrichment caveat. Signal is inverted, not absent (`1 − p̂` AUC ≈ 0.689). See `outputs/20260423_042824_seed42/external_sade_feldman.json`.

**Post-hoc RF baseline (A2-M4.1)**: 3-seed × 5-fold RandomForest on identical splits gives **LOCO mean AUC 0.652 ± 0.136** — **+0.055 over MLP, +0.084 over GNN**. Per-cohort numbers: Gide 0.776 · Seo 0.758 · Hammerman 0.713 · Riaz 0.594 · Hugo 0.420. At N=256 the implicit regularization of a 200-tree ensemble beats both deep models. The BOVIN graph structure confers no current advantage — v2.1 scale-up (Tier B, +IMvigor210 → N≈550) is the needed test. See `outputs/loco_3seed_with_rf.json` and AIM2_REPORT.md §4.6.

**DoD scoreboard** (plan §3.1): **2/8 applicable DoDs pass** (DoD #7 variance + DoD #8 reproducibility). DoDs #1–#6 fail, all numbers captured in `outputs/AIM2_REPORT.md`.

**What the data actually shows**:
- LOCO mean AUC **0.568 ± 0.110** — on par with TIDE / IMPRES on similar cohorts. The BOVIN gene panel has modest response signal.
- GNN does **not** beat a flat 70-gene MLP (gap −0.030). The graph topology adds no statistically significant predictive power on this 256-patient pool.
- Captum IG puts **HLA-A, CD8A, ARG1, LDHB, SLC16A3** in top-5 — an antigen-presentation + T-cell + myeloid-suppression axis, not the pre-registered CRT/HMGB1 ICD axis. Immunologically reasonable but inconsistent with BOVIN's core prior.

## Plan §7 postmortem (pre-committed response to H3 failure)

Per plan §7 the cost of this negative result was known upfront: **no architecture or hyperparameter tuning after seeing the test folds; no cohort subsetting to chase H3.** Both constraints honored.

**Next-step commitments** (plan §8):
- **Aim 2.1 / Tier B** · add IMvigor210 (+348 patients → N ≈ 550) — the smallest-effort scale-up that should restore statistical power. No model change.
- **Aim 2.2 / Tier C** · dbGaP DUCs (Liu, Braun, Miao) — pushes N to ≥ 1,000, enabling deep-model comparisons.
- **Cloughesy labels** · 29 patients pending author contact; stub at `bovin_demo/data/static/cloughesy_manual_labels.csv`.

## Breaking changes

None. `configs/tcga_coad.yaml` + the v0.1-demo pipeline continue to produce the same 0.966 val-AUC on COAD. All v0.2 additions are pure extensions:
- `configs/ici_pool.yaml` is a new config (v0.1 users unaffected).
- `run_training` dispatches on `cfg.data.source` with TCGA as default.
- Test suite grew from 61 → 82+ (all passing).

## Files added / modified

```
AIM2-TRAINING-PLAN.md                                          (new · v2 plan)
bovin-bench/manifest.yaml                                      (+4 Tier A cohort entries)
bovin-bench/cohorts/{gide,hammerman,cloughesy,seo}/DATACARD.md (new)
bovin_demo/data/__init__.py                                    (ici exports)
bovin_demo/data/build.py                                       (new · shared dispatch)
bovin_demo/data/ici_loader.py                                  (new · 660 lines)
bovin_demo/data/sade_loader.py                                 (new · Sade scRNA pseudobulk)
bovin_demo/data/labels.py                                      (+recist_binary_label)
bovin_demo/data/split.py                                       (+leave_one_cohort_out)
bovin_demo/data/static/bovin_gene_aliases.csv                  (new · 72 rows)
bovin_demo/data/static/cloughesy_manual_labels.{csv,TODO.md}   (new · 29-row stub)
bovin_demo/eval/aim2_report.py                                 (new · reads external_sade_feldman.json)
bovin_demo/eval/external_transfer.py                           (new · Sade zero-shot forward)
bovin_demo/eval/hypothesis_tests.py                            (new)
bovin_demo/train/loop.py                                       (ici_pool + LOCO dispatch)
bovin_demo/xai/runner.py                                       (ici_pool support)
configs/ici_pool.yaml                                          (new)
docs/demo_card.md                                              (Aim 2 section prepended)
docs/RELEASE_NOTES_v0.2-aim2.md                                (this file)
notebooks/03_ici_pool_inspector.ipynb                          (new · 48 cells)
notebooks/_build_03_inspector.py                               (new · notebook generator)
tests/test_data.py                                             (+6 LOCO tests)
tests/test_ici_loader.py                                       (new · 15 tests)
tools/build_bovin_gene_aliases.py                              (new)
tools/download_ici_pool.sh                                     (new)
tools/merge_loco_summaries.py                                  (new)
tools/run_ici_loco.py                                          (new)
```

## How to cite this release in a paper draft

> We implemented the BOVIN-Pathway HeteroGNN (82 nodes / 99 edges, v0.1-demo) and evaluated it on a 6-cohort real-ICI pool (Riaz + Hugo + Gide + Hammerman + Cloughesy + Seo; N ≈ 256; 203 after `pre-treatment + RECIST-labeled` filter across 5 cohorts). Under 5-fold leave-one-cohort-out across 3 seeds, the model achieves test AUC **0.568 ± 0.110** versus a flat 70-gene MLP at **0.597 ± 0.117** (mean gap **−0.030**, bootstrap 95% CI excluding 0 in the negative direction). All four pre-registered hypotheses (ICD axis directionality, anti-phagocytic axis directionality, GNN advantage, cross-cohort generalization) fail at the predeclared thresholds. We therefore defer the "BOVIN graph structure adds predictive power" claim to Aim 2.1 (pool scale-up via IMvigor210 and dbGaP controlled-access cohorts) and report the current result as negative evidence.

---

_Release curated by `bovin_demo.eval.aim2_report` + manual post-mortem. All code and numbers are reproducible from `git clone` + `make docker-build` + the Quickstart block in `README.md`._
