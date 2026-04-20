# BOVIN-Pathway Demo · Aim 1 Minimum Closed Loop

> Biology-structured HeteroGNN that consumes TCGA-COAD bulk RNA-seq
> aligned to the **BOVIN-Pathway graph (82 nodes / 99 edges, 11 modules)**
> and produces an **ICD-readiness** score + module-level **Integrated-Gradients** attributions.
>
> Status: **M6 · Report + Reproducibility complete (v0.1-demo).** All six milestones (graph · data · model · training · XAI · report) land; the full path is one `make data-coad && make train && make xai && make eval`.
> See [BOVIN-Pathway-Demo-PLAN.md](../BOVIN-Pathway-Demo-PLAN.md) for the full plan, DoD, and risk register.

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
git clone <this-repo>
cd bovin-pathway-demo
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

---

## Repo layout

```
bovin-pathway-demo/
├── bovin_demo/              主代码包
│   ├── graph/               Pathway graph loader → HeteroData (M1)
│   ├── data/                TCGA loader + HGNC mapping + split (M2)
│   ├── model/               HeteroGNN + module attention + baseline MLP (M3)
│   ├── train/               Lightning module + training loop (M4)
│   ├── xai/                 Captum IG + heatmap viz (M5)
│   ├── eval/                Metrics + report generation (M6)
│   └── cli.py               bovin-demo entry point
├── configs/                 Hydra-style YAML (default + tcga_coad)
├── tests/                   pytest (M0 smoke + M1 placeholders)
├── data/                    raw/processed gitignored; DATACARD + checksums tracked
├── notebooks/               01_demo_walkthrough (M6)
├── outputs/                 per-run outputs/YYYYMMDD_HHMMSS/ (gitignored)
├── Dockerfile
├── Makefile
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
