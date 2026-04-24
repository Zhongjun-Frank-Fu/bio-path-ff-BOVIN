# BOVIN-Pathway Demo · Aim 1 Minimum Closed Loop + Aim 2 Real-ICI Pool

> Biology-structured HeteroGNN that consumes either TCGA-COAD bulk RNA-seq
> or a 6-cohort real-ICI pool aligned to the **BOVIN-Pathway graph (82 nodes
> / 99 edges, 11 modules)** and produces an ICD-readiness (Aim 1) or RECIST
> response (Aim 2) logit + module-level **Integrated-Gradients** attributions.
>
> Status:
> - **v0.1-demo** · M0 → M6 on TCGA-COAD surrogate (val_auc ≈ 0.97, gap vs MLP +0.034). Plan: [BOVIN-Pathway-Demo-PLAN.md](../BOVIN-Pathway-Demo-PLAN.md).
> - **v0.2-aim2** · A2-M1 → A2-M7 on 6-cohort real RECIST pool (203 patients, LOCO mean AUC **0.568 ± 0.110**, GNN-MLP gap **−0.030**, **0/4 pre-registered hypotheses pass**). Negative result reported honestly per plan §7. Full evidence: [`outputs/AIM2_REPORT.md`](outputs/AIM2_REPORT.md). Plan: [AIM2-TRAINING-PLAN.md](AIM2-TRAINING-PLAN.md).

---

## Why this repo exists

This is not an attempt to beat SOTA. It is the **minimum closed loop** that proves:

1. A 82-node biology-structured GNN can be assembled and trained on TCGA-COAD;
2. It beats a flat-MLP baseline on a surrogate ICD-readiness label;
3. XAI automatically recovers CRT / HMGB1 / DAMP as dominant contributors.

That's the "can we walk?" demo. Aim 2 swaps the surrogate label for IMvigor210; Aim 1.2 adds WSI.

---

## Quickstart

### 1. Local (fastest, CPU)

```bash
git clone git@github.com:Zhongjun-Frank-Fu/bio-path-ff-BOVIN.git
cd bio-path-ff-BOVIN
make install           # pip install -e ".[dev]" + pre-commit
make smoke             # python -c "import bovin_demo; print(bovin_demo.__version__)"
make test              # pytest -v
```

### 2. Docker (reproducible)

```bash
make docker-build      # builds bovin-pathway-demo:0.1.0 (linux/amd64)
make docker-run        # smoke import inside container
# later, when M4 lands:
make train             # python -m bovin_demo.cli train --config configs/tcga_coad.yaml
```

Apple Silicon users: the Makefile defaults to `--platform=linux/amd64`.

### 3. Aim 2 — real-ICI pool quickstart (v0.2-aim2)

```bash
# inside the Docker container:
bash tools/download_ici_pool.sh                          # 6 cohorts (~173 MB) + SHA256
python tools/build_bovin_gene_aliases.py                 # one-off: 72 BOVIN symbols → Entrez/Ensembl
python -m bovin_demo.cli train --config configs/ici_pool.yaml   # stratified seed 42 (~5 min)
python tools/run_ici_loco.py --seeds 42 1337 2024        # 3-seed × 5-fold LOCO (~75 min)
python tools/merge_loco_summaries.py outputs/<date>_loco_summary.json ...  # aggregate to 3-seed merged
python -m bovin_demo.eval.hypothesis_tests \
    --run-dir outputs/<stratified_dir> \
    --loco-merged outputs/loco_3seed_merged.json          # evaluate H1-H4
python -m bovin_demo.eval.aim2_report \
    --stratified-run outputs/<stratified_dir> \
    --loco-merged    outputs/loco_3seed_merged.json       # write outputs/AIM2_REPORT.md
```

Expected result (per plan §3.1 DoD): all 4 pre-registered hypotheses fail on the current 5-cohort labeled pool — the pipeline runs cleanly, the numbers are stable across seeds, but the graph structure + 256-patient pool is insufficient to show the claimed GNN-vs-MLP advantage. See `outputs/AIM2_REPORT.md`.

Cloughesy labels (29 patients) are pending author contact — fill `bovin_demo/data/static/cloughesy_manual_labels.csv` and everything re-runs unchanged.

---

## Repo layout

```
bio-path-ff-BOVIN/
├── bovin_demo/              主代码包 (Python 包名保留 bovin_demo)
│   ├── graph/               Pathway graph loader → HeteroData (M1)
│   ├── data/                TCGA loader + HGNC mapping + label + split + dataset (M2)
│   ├── model/               HeteroGNN + module attention + baseline MLP (M3)
│   ├── train/               LitBovinModule + run_training + 3-seed sweep (M4)
│   ├── xai/                 Captum IG + aggregate + heatmap viz (M5)
│   ├── eval/                bootstrap metrics + report.md + LUAD zero-shot (M6)
│   └── cli.py               bovin-demo 入口 (sanity/train/xai/eval)
├── configs/                 OmegaConf YAML (default + tcga_coad)
├── tests/                   pytest · 61 passed, 1 skipped
├── data/                    DATACARD + checksums.txt 跟踪;  raw/ 不跟踪
├── docs/                    demo_card.md + RELEASE_NOTES_v0.1-demo.md
├── tools/                   download_tcga_coad/luad.sh + parse_graph_v0 + walkthrough
├── outputs/                 每次 run 一个时间戳目录 (gitignored)
├── Dockerfile               torch 2.3.1 CPU + PyG 2.5.3 + setuptools<70
├── Makefile                 make train / xai / eval / eval-luad
├── pyproject.toml
└── .github/workflows/ci.yml
```

Module dependency is strictly one-way:
`graph → heterodata → model → train → {xai, eval}`, and `cli` assembles them.

---

## Current CLI surface (M0)

```
$ bovin-demo --help
usage: bovin-demo [-h] [--version] [--seed SEED] command ...

positional arguments:
  command
    sanity              Import + graph loader smoke check (M1)
    train               Train HeteroGNN (M4, not yet implemented)
    xai                 Integrated-Gradients readout (M5, NYI)
    eval                Evaluate + produce report.md (M6, NYI)
```

At M0, only `sanity` returns 0. The other three subcommands exit with code 2 and
a clear "not yet implemented at this milestone" message — deliberate, so no one
mistakes a placeholder for a broken run.

---

## Reproducibility contract

- Fixed seed (default 42, override with `BOVIN_SEED` env var or `--seed`).
- Every run gets its own `outputs/YYYYMMDD_HHMMSS/` directory — no overwriting.
- `data/checksums.txt` pins raw TCGA files; processed tensors are regenerable.
- Docker image pins torch 2.3.1 CPU + PyG 2.5.3.

Target: same machine, same seed → AUC variation `< 0.01`.

---

## Definition of Done (M0)

```
[x] `make install` succeeds on a clean env
[x] `python -c "import bovin_demo"` prints the version
[x] `pytest -v` is green (smoke + M1 placeholders)
[x] `ruff check bovin_demo tests` passes
[x] `.github/workflows/ci.yml` has lint + test + docker-build jobs
[x] Pre-commit installs and runs clean
```

## Definition of Done (M1)

```
[x] tools/parse_graph_v0.py regenerates bovin_demo/graph/bovin_pathway_v0.json from the source markdown
[x] load_graph() returns 82 nodes / 99 edges, all 11 modules populated
[x] pydantic schema rejects dangling edges and duplicate ids
[x] to_heterodata() produces ≥5 node types and ≥5 relation types
[x] GATv2Conv forward pass runs on the largest intra-type edge slice
[x] tools/render_graph_overview.py emits a module-colored sanity PNG
```

## Definition of Done (M2)

```
[x] make data-coad fetches Xena HiSeqV2 + clinical + survival and writes data/checksums.txt
[x] load_coad() returns a CoadBundle(expr, clinical, survival) aligned on sample barcode
[x] map_to_pathway_nodes() hit-rate ≥ 0.70 on the synthetic Xena fixture (and, empirically, on real COAD)
[x] ICD-readiness label formula (z-score signature + median split) produces ~50/50 classes
[x] stratified_split(seed=42) is deterministic and label-balanced across 60/20/20 folds
[x] data/DATACARD.md documents source, license, formula, and "this is a surrogate" warning
```

## Definition of Done (M3)

```
[x] HeteroGNN: per-node-type Linear encoder → HeteroConv(GATv2Conv) × 2 → HGTConv × 1
[x] ModuleAttentionPool: per-module soft-attention → 11-dim patient embedding
[x] Readiness head: Linear(len(module_ids), 1) produces a single logit
[x] BaselineMLP: flat 70-gene vector → 2-layer MLP (intentionally graph-blind)
[x] build_classifier(data) factory yields a wired, grad-flowing HeteroGNNClassifier
[x] `bovin-demo sanity` runs graph load → HeteroData → both models forward → prints logits
```

## Definition of Done (M4)

```
[x] PatientGraphDataset: one HeteroData per patient with [z_expr, observed_flag] features
[x] LitBovinModule (pl.LightningModule): BCE + pos_weight + BinaryAUROC + cosine LR
[x] run_training(config, seed): seeds → fits → early-stops on val_auc → tests on hold-out
[x] Per-run artifacts: outputs/YYYYMMDD_HHMMSS_seed{N}/{ckpt,logs,training.log,metrics.json}
[x] BaselineMLP trained on the same split — metrics.json records gap for DoD #3
[x] `bovin-demo train --seeds 42,1337,2024` runs a stability sweep and prints mean ± std
[x] val-AUC ≥ 0.65 on real TCGA-COAD (empirical: 0.966 at seed=42, 80-epoch early-stopped)
```

## Definition of Done (M5)

```
[x] FlatInputWrapper + Captum IntegratedGradients attribute logit to per-node features
[x] Module rollup (|attr| sum per module) + node ranking (mean |attr|) utilities
[x] Heatmap: top-TPR patients × 11 modules, Dossier palette strip on top
[x] Sanity JSON records DoD #4 (M4 in top-3 modules) and #5 (CRT/HMGB1 in top-5 nodes)
[x] `bovin-demo xai --run-dir <path>` one-liner drives the full pipeline
```

## Definition of Done (M6)

```
[x] bootstrap metrics (AUC/ACC/F1/Brier/ECE with 500× CI)
[x] build_report produces run_dir/report.md (DoD checklist at the top + heatmap embed)
[x] LUAD zero-shot transfer: run_luad_zero_shot scores any ckpt on data/raw_luad
[x] Dockerfile pins setuptools<70 (fixes Lightning 2.2 pkg_resources crash)
[x] docs/demo_card.md one-page paste-in for Research Plan §10
[x] `bovin-demo eval --run-dir <path> [--luad-raw-dir data/raw_luad]` — one command for report.md
```

v0.1-demo is the tag to cut once DoD #4 + #5 are verified on real TCGA-COAD and DoD #6 passes on real LUAD.

---

## Caveats (stated loudly on purpose)

- **Surrogate label.** The `CRT+HMGB1+HSP-signature − CD47 − CD24` label is *not* an
  ICI-response label. It proves architecture, not efficacy. See DATACARD.md.
- **Single cohort.** COAD only at M6; LUAD is a one-off zero-shot inference check.
- **Bulk RNA-seq only.** No multimodal. Hooks for pathomics/clinical are stubbed
  but not wired.

---

## See also

- [BOVIN-Pathway-Demo-PLAN.md](../BOVIN-Pathway-Demo-PLAN.md) — full implementation plan
- [BOVIN-Pathway-Demo-Plan.html](../BOVIN-Pathway-Demo-Plan.html) — visual dashboard (Gantt + DAG + DoD)
- [BOVIN-AI-Dossier.html](../BOVIN-AI-Dossier.html) — research plan + pathway graph + roadmap
- [BOVIN-Pathway-Graph-v0.md](../BOVIN-Pathway-Graph-v0.md) — 82 nodes / 99 edges source of truth
