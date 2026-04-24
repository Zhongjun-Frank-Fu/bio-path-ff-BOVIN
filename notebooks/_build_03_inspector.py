"""Build notebooks/03_ici_pool_inspector.ipynb from a structured template.

This is a one-shot generator script — run once, produces the .ipynb, then
gets committed alongside it. Not expected to be re-run except when the
notebook structure needs major changes.

Conventions:
    - Every cohort follows the same (a)source / (b)shape / (c)columns-fold /
      (d)head / (e)distribution cell sequence, so the notebook is grep-friendly.
    - Column lists are rendered via <details><summary>…</summary>…</details>
      so they collapse by default.
    - No hard-coded patient counts — show what the raw file says.
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent / "03_ici_pool_inspector.ipynb"

cells: list[dict] = []

def md(src: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    })

def code(src: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    })

# ════════════════════════════════════════════════════════════════════════════
# §0 · Title + imports + path setup
# ════════════════════════════════════════════════════════════════════════════
md("""# 03 · BOVIN-Bench Tier A ICI Pool · Database Inspector

**Purpose** — Aim 2 第一轮数据处理：6 个公开 ICI cohort 下载后逐个体检。

**对每个 cohort 做 5 件事**：
- (a) **来源** — GEO / ENA / cBioPortal 以及文章引用
- (b) **Shape** — 表达矩阵和 clinical 的维度
- (c) **Columns** — 完整列清单（折叠显示）
- (d) **Head(10)** — 前 10 行的实际内容
- (e) **分布** — 表达值的整体分布 + 每样本汇总

**6 个 Tier A cohort**：

| § | Cohort | Accession | 瘤种 | 治疗 | 源格式 |
|---|---|---|---|---|---|
| 1 | Riaz 2017 | GSE91061 | melanoma | nivolumab | raw counts (hg19 known-gene, Entrez) |
| 2 | Hugo 2016 | GSE78220 | melanoma | pembro / nivo | FPKM (HGNC symbol) |
| 3 | Gide 2019 | PRJEB23709 → cBioPortal `mel_iatlas_gide_2019` | melanoma | pembro / nivo / ipi+nivo | iAtlas harmonized (HGNC symbol) |
| 4 | Hammerman 2020 | GSE165278 | melanoma | ipilimumab | TPM (HGNC symbol) |
| 5 | Cloughesy 2019 | GSE121810 | GBM | neoadjuvant pembro | raw counts (HGNC symbol) |
| 6 | Seo 2020 | GSE165252 | esophageal | atezolizumab | normalized counts + VST (Ensembl) |

§7 末尾做 **synthesis** — 跨 cohort 基因交集 + per-cohort z-score + pooled matrix.

> 此 notebook 需在 Docker 容器（`make docker-shell`）或装好 `pandas + pyreadr + openpyxl` 的环境里跑。
""")

code("""# --- 0.1 Imports ---
from __future__ import annotations
import gzip, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.family"] = "DejaVu Sans"
""")

code("""# --- 0.2 Path setup ---
def _find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "raw_ici").exists():
            return p
    return here

ROOT = _find_repo_root()
RAW  = ROOT / "data" / "raw_ici"
assert RAW.exists(), f"{RAW} not found — run `bash tools/download_ici_pool.sh` first"

print(f"ROOT = {ROOT}")
print(f"RAW  = {RAW}")
print()
print("Files on disk:")
for p in sorted(RAW.rglob("*")):
    if p.is_file() and not p.name.startswith("."):
        rel = p.relative_to(RAW)
        kb = p.stat().st_size / 1024
        print(f"  {str(rel):70s}  {kb:>10.1f} KB")
""")

code("""# --- 0.3 Helper: collapsible column list ---
def show_columns(df: pd.DataFrame, title: str, preview: int = 20) -> None:
    \"\"\"Render the full column list inside a <details> fold, with a short preview outside.\"\"\"
    n = len(df.columns)
    preview_items = ", ".join(f"<code>{c}</code>" for c in df.columns[:preview])
    full_items = "<br>".join(
        f"<code>{i:>4d}</code> &nbsp; <code>{c}</code> &nbsp; <i>{str(df[c].dtype)}</i>"
        for i, c in enumerate(df.columns)
    )
    more = f" &middot; (first {preview} shown)" if n > preview else ""
    html = (
        f"<div style='font-size:0.95em'><b>{title}</b> &mdash; {n} columns{more}<br>"
        f"{preview_items}"
        f"<details style='margin-top:0.6em'><summary><b>Expand full column list ({n})</b></summary>"
        f"<div style='padding:0.5em 0 0 1em; font-family:monospace; font-size:0.9em'>{full_items}</div>"
        f"</details></div>"
    )
    display(HTML(html))

def show_source(title: str, lines: list[str]) -> None:
    bullets = "".join(f"<li>{x}</li>" for x in lines)
    display(HTML(f"<div><b>{title}</b><ul style='margin-top:0.3em'>{bullets}</ul></div>"))

def distribution_panel(expr: pd.DataFrame, title: str,
                        sample_axis: str = "columns",
                        log_transform: bool = False) -> None:
    \"\"\"Plot overall value distribution + per-sample library size / boxplot.\"\"\"
    if sample_axis == "columns":
        arr = expr.select_dtypes(include=[np.number]).values
        per_sample_sum = expr.select_dtypes(include=[np.number]).sum(axis=0)
    else:
        arr = expr.select_dtypes(include=[np.number]).values
        per_sample_sum = expr.select_dtypes(include=[np.number]).sum(axis=1)
    flat = arr.ravel()
    flat = flat[~np.isnan(flat)]
    plot_vals = np.log2(flat + 1) if log_transform else flat

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].hist(plot_vals, bins=80, color="#5B8DEF", alpha=0.85)
    axes[0].set_title("value histogram" + (" · log2(x+1)" if log_transform else ""))
    axes[0].set_xlabel("expression")
    axes[0].set_ylabel("# gene×sample")
    axes[1].boxplot([expr[c].dropna().values for c in expr.select_dtypes(include=[np.number]).columns[:40]],
                    widths=0.6, showfliers=False)
    axes[1].set_title("per-sample distribution (first 40)")
    axes[1].set_xticklabels([])
    axes[1].set_ylabel("expression")
    axes[2].hist(per_sample_sum.values, bins=30, color="#F2A33A", alpha=0.85)
    axes[2].set_title("per-sample total (library-size-like)")
    axes[2].set_xlabel("sum across genes")
    axes[2].set_ylabel("# samples")
    plt.suptitle(title, fontsize=12, y=1.04)
    plt.tight_layout()
    plt.show()

    print(f"[stats] N values = {flat.size:,}")
    print(f"         min={flat.min():.2f}  max={flat.max():.2f}  mean={flat.mean():.2f}  median={np.median(flat):.2f}")
    print(f"         % exactly 0 = {(flat==0).mean()*100:.1f}%")
""")

# ════════════════════════════════════════════════════════════════════════════
# §1 Riaz
# ════════════════════════════════════════════════════════════════════════════
md("""## §1 · Riaz 2017 · GSE91061

Nivolumab in metastatic melanoma. The headline melanoma ICI cohort, but sample counts are mixed pre + on-treatment.""")

code("""# (a) Source
show_source("Riaz 2017 · GSE91061", [
    "Riaz N, Havel JJ, Makarov V, <i>et al.</i> <i>Cell</i> 2017;171(4):934-949.e16.",
    "GEO: <a href='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE91061'>GSE91061</a>",
    "Modality: bulk RNA-seq · hg19 known-gene raw counts",
    "Treatment: nivolumab (anti-PD-1); ~30% post-ipilimumab",
    "Label source: <code>BOR</code> column in <code>bms038_clinical_data.csv</code> (paper supp, not GEO — deferred to M2)",
])
""")

code("""# (b) Shape — expression matrix
riaz_expr_path = RAW / "riaz_gse91061" / "GSE91061_BMS038109Sample.hg19KnownGene.raw.csv.gz"
riaz_expr = pd.read_csv(riaz_expr_path)
riaz_expr = riaz_expr.rename(columns={"Unnamed: 0": "entrez_id"}).set_index("entrez_id")
print(f"Riaz expression shape: {riaz_expr.shape}  ({riaz_expr.shape[0]:,} genes × {riaz_expr.shape[1]} samples)")
print(f"Index type: {riaz_expr.index.dtype} (Entrez gene IDs — need entrez→HGNC mapping in M2 loader)")
print(f"Sample ID pattern: {list(riaz_expr.columns[:3])} …")

# parse sample IDs: PtX_{Pre|On}_{Tag}
pattern = re.compile(r"^Pt(\\d+)_(Pre|On)_")
sample_meta = pd.DataFrame({
    "sample_id": riaz_expr.columns,
    "patient":   [pattern.match(c).group(1) if pattern.match(c) else None for c in riaz_expr.columns],
    "timepoint": [pattern.match(c).group(2).lower() if pattern.match(c) else None for c in riaz_expr.columns],
})
print(f"\\nUnique patients: {sample_meta['patient'].nunique()}")
print(f"Timepoints:\\n{sample_meta['timepoint'].value_counts()}")
""")

code("""# (c) Columns — full list folded
show_columns(riaz_expr.iloc[:0], "Riaz expression — sample columns")
display(HTML(f"<i>Index: {riaz_expr.index.name} — {len(riaz_expr):,} gene rows · Entrez integer IDs</i>"))
""")

code("""# (d) head(10) — first 10 genes × first 5 samples
riaz_expr.iloc[:10, :5]
""")

code("""# (e) Distribution — raw counts are heavy-tailed; show on log2(x+1) axis
distribution_panel(riaz_expr, title="Riaz GSE91061 · raw counts (log2 for display)", log_transform=True)
""")

# ════════════════════════════════════════════════════════════════════════════
# §2 Hugo
# ════════════════════════════════════════════════════════════════════════════
md("""## §2 · Hugo 2016 · GSE78220

Pembrolizumab / nivolumab in metastatic melanoma. **Pre-treatment samples only** — the cleanest label-to-expression alignment of the 6 cohorts.""")

code("""show_source("Hugo 2016 · GSE78220", [
    "Hugo W, Zaretsky JM, Sun L, <i>et al.</i> <i>Cell</i> 2016;165(1):35-44.",
    "GEO: <a href='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78220'>GSE78220</a>",
    "Modality: bulk RNA-seq · FPKM (HGNC gene symbols)",
    "Treatment: pembrolizumab + nivolumab (混合 anti-PD-1)",
    "Label source: Hugo 2016 Table S1 (paper supp) — RECIST CR/PR/SD/PD 直接可用",
])
""")

code("""hugo_expr = pd.read_excel(RAW/"hugo_gse78220"/"GSE78220_PatientFPKM.xlsx", engine="openpyxl")
hugo_expr = hugo_expr.rename(columns={"Gene": "symbol"}).set_index("symbol")
print(f"Hugo expression shape: {hugo_expr.shape}  ({hugo_expr.shape[0]:,} genes × {hugo_expr.shape[1]} samples)")
print(f"Index: HGNC gene symbols")
print(f"Sample ID pattern: {list(hugo_expr.columns[:3])}  (all 'baseline' — pre-treatment)")
""")

code("""show_columns(hugo_expr.iloc[:0], "Hugo expression — sample columns")
display(HTML(f"<i>Index: symbol — {len(hugo_expr):,} gene rows</i>"))
""")

code("""hugo_expr.iloc[:10, :5]
""")

code("""distribution_panel(hugo_expr, title="Hugo GSE78220 · FPKM", log_transform=True)
""")

# ════════════════════════════════════════════════════════════════════════════
# §3 Gide
# ════════════════════════════════════════════════════════════════════════════
md("""## §3 · Gide 2019 · PRJEB23709 → cBioPortal `mel_iatlas_gide_2019`

Pembro / nivo / ipi+nivo in melanoma. **R1 risk mitigated** — ENA stores raw fastq (~500 GB for 74 patients), so we use the iAtlas-harmonized version on cBioPortal instead of running salmon ourselves.

Also the only cohort with **ready-to-use clinical metadata in tabular form** (`data_clinical_patient.txt` + `data_clinical_sample.txt`).""")

code("""show_source("Gide 2019 · PRJEB23709 (via cBioPortal iAtlas mirror)", [
    "Gide TN, Quek C, Menzies AM, <i>et al.</i> <i>Cancer Cell</i> 2019;35(2):238-255.e6.",
    "Primary: ENA <a href='https://www.ebi.ac.uk/ena/browser/view/PRJEB23709'>PRJEB23709</a> (fastq only — not used)",
    "Used mirror: <a href='https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019'>cBioPortal mel_iatlas_gide_2019</a>",
    "Modality: bulk RNA-seq · iAtlas-harmonized (HGNC symbols, log-scale)",
    "Treatment: anti-PD-1 (pembro/nivo) monotherapy or combo with anti-CTLA-4 (ipilimumab)",
    "Label source: inline in clinical files — <code>RESPONDER</code>, <code>CLINICAL_BENEFIT</code>, <code>OS_MONTHS</code>, <code>PFS_MONTHS</code>",
])
""")

code("""gide_expr_path = RAW / "gide_prjeb23709" / "data_mrna_seq_expression.txt"
gide_expr = pd.read_csv(gide_expr_path, sep="\\t")
gide_expr = gide_expr.rename(columns={"Hugo_Symbol": "symbol"}).set_index("symbol")
# deduplicate — cBioPortal files occasionally have duplicate symbol rows
gide_expr = gide_expr[~gide_expr.index.duplicated(keep="first")]

gide_patient = pd.read_csv(RAW/"gide_prjeb23709"/"data_clinical_patient.txt", sep="\\t", comment="#")
gide_sample  = pd.read_csv(RAW/"gide_prjeb23709"/"data_clinical_sample.txt",  sep="\\t", comment="#")

print(f"Gide expression:        {gide_expr.shape}   ({gide_expr.shape[0]:,} genes × {gide_expr.shape[1]} samples)")
print(f"Gide clinical_patient:  {gide_patient.shape}")
print(f"Gide clinical_sample:   {gide_sample.shape}")
""")

code("""show_columns(gide_expr.iloc[:0], "Gide expression — sample columns")
show_columns(gide_patient, "Gide clinical_patient — columns", preview=14)
show_columns(gide_sample,  "Gide clinical_sample — columns",  preview=19)
""")

code("""# head(10) for each of the 3 files
print("— expression (10 × 5) —")
display(gide_expr.iloc[:10, :5])
print("— clinical_patient (10 rows) —")
display(gide_patient.head(10))
print("— clinical_sample (10 rows) —")
display(gide_sample.head(10))
""")

code("""# distribution + response breakdown
distribution_panel(gide_expr, title="Gide iAtlas · harmonized log-scale expression", log_transform=False)

print("\\nRESPONDER breakdown (patient-level):")
print(gide_patient["RESPONDER"].value_counts(dropna=False))
print("\\nCLINICAL_BENEFIT breakdown:")
print(gide_patient["CLINICAL_BENEFIT"].value_counts(dropna=False))
""")

# ════════════════════════════════════════════════════════════════════════════
# §4 Hammerman
# ════════════════════════════════════════════════════════════════════════════
md("""## §4 · Hammerman 2020 · GSE165278

Ipilimumab in melanoma. Smallest cohort (N=21). **Sample IDs encode response** (`CR####` vs `NR####`) — no external label mapping needed, response is literally in the column name.""")

code("""show_source("Hammerman 2020 · GSE165278", [
    "Hammerman P et al. GSE165278 (2020); ipilimumab in cutaneous melanoma.",
    "GEO: <a href='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165278'>GSE165278</a>",
    "Modality: bulk RNA-seq · TPM (HGNC symbols)",
    "Treatment: ipilimumab (anti-CTLA-4)",
    "Label source: <b>encoded in sample ID prefix</b> — <code>CR*</code> = complete responder, <code>NR*</code> = non-responder",
])
""")

code("""hamm_expr = pd.read_excel(RAW/"hammerman_gse165278"/"GSE165278_TPM_original_23k_genes.xlsx", engine="openpyxl")
hamm_expr = hamm_expr.rename(columns={"Gene": "symbol"}).set_index("symbol")
# dedupe symbol (TPM file sometimes has repeats from Ensembl-derived conflicts)
hamm_expr = hamm_expr[~hamm_expr.index.duplicated(keep="first")]
print(f"Hammerman expression shape: {hamm_expr.shape}")

# decode response from sample IDs
resp = pd.Series(
    ["CR" if c.startswith("CR") else "NR" if c.startswith("NR") else "?"
     for c in hamm_expr.columns],
    index=hamm_expr.columns, name="response_raw"
)
print(f"\\nResponse prefix breakdown:\\n{resp.value_counts()}")
""")

code("""show_columns(hamm_expr.iloc[:0], "Hammerman expression — sample columns")
""")

code("""hamm_expr.iloc[:10, :6]
""")

code("""distribution_panel(hamm_expr, title="Hammerman GSE165278 · TPM", log_transform=True)
""")

# ════════════════════════════════════════════════════════════════════════════
# §5 Cloughesy
# ════════════════════════════════════════════════════════════════════════════
md("""## §5 · Cloughesy 2019 · GSE121810

**Neoadjuvant** pembrolizumab in recurrent glioblastoma (GBM). The only non-melanoma bulk-RNA cohort in Tier A. RECIST is tricky in GBM (MRI-based mRANO); expect lower response rate and use OS>12mo as aux label (§2.2 of plan).""")

code("""show_source("Cloughesy 2019 · GSE121810", [
    "Cloughesy TF, Mochizuki AY, Orpilla JR, <i>et al.</i> <i>Nat Med</i> 2019;25(3):477-486.",
    "GEO: <a href='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121810'>GSE121810</a>",
    "Modality: bulk RNA-seq · raw counts (HGNC symbols)",
    "Treatment: <b>neoadjuvant</b> pembrolizumab (anti-PD-1), then surgery",
    "Label source: series_matrix characteristics_ch1 + Cloughesy 2019 Table 1 (extracted in M2 loader)",
    "Sample ID pattern: <code>PtN_A</code> (adjuvant post-op) / <code>PtN_B</code> (baseline) — filter to baseline for v2 training",
])
""")

code("""clo_expr = pd.read_excel(RAW/"cloughesy_gse121810"/"GSE121810_Prins.PD1NeoAdjv.Jul2018.HUGO.PtID.xlsx", engine="openpyxl")
clo_expr = clo_expr.rename(columns={"Genes": "symbol"}).set_index("symbol")
clo_expr = clo_expr[~clo_expr.index.duplicated(keep="first")]
print(f"Cloughesy expression shape: {clo_expr.shape}")

tp = pd.Series(
    [c.split("_")[-1] if "_" in c else "?" for c in clo_expr.columns],
    index=clo_expr.columns, name="timepoint_tag"
)
print(f"\\nTimepoint tag breakdown (A/B suffix):\\n{tp.value_counts()}")
""")

code("""show_columns(clo_expr.iloc[:0], "Cloughesy expression — sample columns")
""")

code("""clo_expr.iloc[:10, :6]
""")

code("""distribution_panel(clo_expr, title="Cloughesy GSE121810 · raw counts (log2 for display)", log_transform=True)
""")

# ════════════════════════════════════════════════════════════════════════════
# §6 Seo
# ════════════════════════════════════════════════════════════════════════════
md("""## §6 · Seo 2020 · GSE165252

Atezolizumab in esophageal adenocarcinoma (EAC). Ships **two matrices**: `norm.cnt` (library-size-normalized) and `vst` (variance-stabilizing transformed). Gene IDs are **Ensembl with version suffix** — requires stripping and mapping to HGNC in M2 loader.""")

code("""show_source("Seo 2020 · GSE165252", [
    "Seo AN, Gerke TA, Wu C, <i>et al.</i> data in GEO GSE165252 (2020/2021).",
    "GEO: <a href='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165252'>GSE165252</a>",
    "Modality: bulk RNA-seq · <b>two provided matrices</b>: <code>norm.cnt</code> and <code>vst</code>",
    "Gene ID: <b>Ensembl w/ version</b> (e.g., <code>ENSG00000000003.15</code>) — strip version + map HGNC in M2",
    "Treatment: atezolizumab (anti-PD-L1)",
    "Label source: series_matrix characteristics_ch1 (response, PFI) — extracted in M2 loader",
])
""")

code("""seo_norm = pd.read_csv(RAW/"seo_gse165252"/"GSE165252_norm.cnt_PERFECT.txt.gz", sep="\\t")
seo_norm = seo_norm.rename(columns={"Unnamed: 0": "ensembl_id"}).set_index("ensembl_id")

seo_vst  = pd.read_csv(RAW/"seo_gse165252"/"GSE165252_vst_PERFECT.txt.gz", sep="\\t")
seo_vst  = seo_vst.rename(columns={"Unnamed: 0": "ensembl_id"}).set_index("ensembl_id")

print(f"Seo norm.cnt shape: {seo_norm.shape}")
print(f"Seo vst shape:      {seo_vst.shape}")
print(f"\\nSame sample columns? {list(seo_norm.columns) == list(seo_vst.columns)}")
print(f"\\nFirst 3 Ensembl IDs: {list(seo_norm.index[:3])}")
""")

code("""show_columns(seo_norm.iloc[:0], "Seo norm.cnt — sample columns")
display(HTML("<i>vst matrix has the same samples — see above.</i>"))
""")

code("""print("— Seo norm.cnt (first 10 genes × 5 samples) —")
display(seo_norm.iloc[:10, :5])
print("— Seo vst (first 10 genes × 5 samples) —")
display(seo_vst.iloc[:10, :5])
""")

code("""# Two panels side by side: norm.cnt (right-skewed) vs vst (near-normal after stabilization)
distribution_panel(seo_norm, title="Seo GSE165252 · norm.cnt (library-size-normalized)", log_transform=True)
distribution_panel(seo_vst,  title="Seo GSE165252 · VST (variance-stabilizing transform)", log_transform=False)
""")

# ════════════════════════════════════════════════════════════════════════════
# §7 Synthesis
# ════════════════════════════════════════════════════════════════════════════
md("""## §7 · Synthesis · Pooled Matrix Preview

> **Scope of this section.** This is the **exploratory** pooled matrix, not the
> production loader. Heavy gene-ID mapping (Entrez→HGNC for Riaz, Ensembl→HGNC for Seo)
> is deferred to **A2-M2** (`bovin_demo/data/ici_loader.py`).
> Here we intersect on the **4 HGNC-symbol cohorts** (Hugo / Gide / Hammerman / Cloughesy)
> to sanity-check cross-cohort batch behavior before writing the full loader.

### Synthesis procedure

```
for each of {Hugo, Gide, Hammerman, Cloughesy}:
    1. take expr matrix, index = HGNC symbol, columns = samples
    2. transpose → samples × genes
    3. per-cohort z-score across samples (zero mean, unit variance per gene)
    4. tag each row with cohort_id

concat 4 cohorts on the gene intersection → pooled matrix
```

**Why per-cohort z-score first** (plan §2.3): removes cohort-level mean/variance offsets
before the pooled z-score that `PatientGraphDataset` applies. Two-step normalization
matches the existing demo's convention in `bovin_demo/data/dataset.py:58`.

**Why only 4 cohorts here**: Riaz uses Entrez IDs (integers), Seo uses versioned Ensembl
IDs; M2 loader does the proper HGNC mapping. Showing the synthesis with 4 is enough
to see the shape and distribution of a pooled frame.
""")

code("""# §7.1 · build the 4 per-cohort frames (samples × genes · HGNC symbol)
def standardize(expr_genes_by_samples: pd.DataFrame, cohort: str) -> pd.DataFrame:
    df = expr_genes_by_samples.T.copy()              # samples × genes
    # per-cohort z-score across samples, per gene column
    df = (df - df.mean(axis=0)) / df.std(axis=0).replace(0, np.nan)
    df["cohort_id"] = cohort
    return df

hugo_z = standardize(hugo_expr, "hugo_gse78220")
gide_z = standardize(gide_expr, "gide_prjeb23709")
hamm_z = standardize(hamm_expr, "hammerman_gse165278")
clo_z  = standardize(clo_expr,  "cloughesy_gse121810")

print("per-cohort sample counts:")
for name, df in [("Hugo", hugo_z), ("Gide", gide_z), ("Hammerman", hamm_z), ("Cloughesy", clo_z)]:
    print(f"  {name:12s}  {df.shape[0]:>4d} samples × {df.shape[1]-1:>5d} genes")
""")

code("""# §7.2 · gene intersection across all 4 cohorts
genes_per = {
    "Hugo":       set(c for c in hugo_z.columns if c != "cohort_id"),
    "Gide":       set(c for c in gide_z.columns if c != "cohort_id"),
    "Hammerman":  set(c for c in hamm_z.columns if c != "cohort_id"),
    "Cloughesy":  set(c for c in clo_z.columns  if c != "cohort_id"),
}
all_inter = set.intersection(*genes_per.values())
print(f"gene intersection across 4 symbol-indexed cohorts: {len(all_inter):,} HGNC symbols")
print(f"\\nper-cohort gene count:")
for k, v in genes_per.items():
    print(f"  {k:12s}  total={len(v):>6,}  ∩ intersection / its own = {len(all_inter)/len(v)*100:>5.1f}%")
""")

code("""# §7.3 · concatenate into pooled matrix
cols = sorted(all_inter) + ["cohort_id"]
pooled = pd.concat([
    hugo_z[cols],
    gide_z[cols],
    hamm_z[cols],
    clo_z[cols],
], axis=0, ignore_index=False)
pooled.index.name = "sample_id"

print(f"Pooled shape: {pooled.shape}  ({pooled.shape[0]} samples × {pooled.shape[1]-1} genes + cohort_id)")
print(f"\\nSamples per cohort:\\n{pooled['cohort_id'].value_counts()}")
""")

code("""# §7.4 · pooled database description — shape / columns / head / distribution

# shape + first 10 rows
print(f"=== Pooled matrix · shape = {pooled.shape} ===")
display(pooled.iloc[:10, :8])  # first 10 samples × first 8 gene columns
""")

code("""# column fold (all genes + cohort_id column)
show_columns(pooled.iloc[:0], "Pooled matrix — columns (post per-cohort z-score)", preview=15)
""")

code("""# pooled distribution — separate panel per cohort to visualize residual batch
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# left: overall pooled histogram, colored by cohort
for cohort, sub in pooled.groupby("cohort_id"):
    vals = sub.drop(columns=["cohort_id"]).values.ravel()
    vals = vals[~np.isnan(vals)]
    axes[0].hist(vals, bins=80, alpha=0.45, label=cohort, density=True)
axes[0].set_title("pooled · per-gene z-score distribution (density, colored by cohort)")
axes[0].set_xlabel("z-score"); axes[0].set_ylabel("density")
axes[0].set_xlim(-5, 5)
axes[0].legend(fontsize=8)

# right: per-sample library-check via per-sample mean z-score (should hover around 0)
per_sample_mean = pooled.drop(columns=["cohort_id"]).mean(axis=1)
for cohort, sub in pooled.groupby("cohort_id"):
    axes[1].hist(per_sample_mean.loc[sub.index].values, bins=15, alpha=0.45, label=cohort)
axes[1].set_title("pooled · per-sample mean(z-score)  (≈0 = normalization clean)")
axes[1].set_xlabel("mean z-score across genes per sample"); axes[1].set_ylabel("# samples")
axes[1].axvline(0, color="k", ls="--", lw=0.8)
axes[1].legend(fontsize=8)

plt.tight_layout(); plt.show()

# summary stats
flat = pooled.drop(columns=["cohort_id"]).values.ravel()
flat = flat[~np.isnan(flat)]
print(f"[pooled stats] N = {flat.size:,}  |  mean = {flat.mean():+.3f}  std = {flat.std():.3f}  |  % |z|>3 = {(np.abs(flat)>3).mean()*100:.2f}%")
""")

md("""### 本节结论 — 可写进 DATACARD 或 report

| | 维度 | 数值（本 notebook 一次运行） |
|---|---|---|
| 合成范围 | cohort 数 | 4（Hugo / Gide / Hammerman / Cloughesy） |
| | 暂缓 | Riaz（Entrez） / Seo（Ensembl） → M2 loader |
| shape | samples × genes | 见 §7.3 输出 |
| 列 | 基因交集大小 | 见 §7.2 输出 |
| 分布 | 预期 | 每 cohort mean ≈ 0, std ≈ 1（z-score 本身保证） |
| 批次残留 | per-sample mean(z) | 集中在 0 附近 = normalization 干净 |

**下一步 (A2-M2)**：
1. 写 `bovin_demo/data/ici_loader.py` — 覆盖 6 cohort + Entrez/Ensembl→HGNC 映射
2. 对齐到 BOVIN 70 observable node 基因列表（命中率 DoD ≥ 70%）
3. 产出 `expr_pool.parquet` + `labels_pool.csv` + `cohort_ids.csv` 三元组
4. 测试 `tests/test_ici_loader.py` — pooled N ∈ [230, 260], label NaN rate < 20%

---

*Notebook 03 · ICI Pool Inspector · A2-M1 交付物 · 2026-04-22*
""")

# ════════════════════════════════════════════════════════════════════════════
# write out
# ════════════════════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ·  {len(cells)} cells")
