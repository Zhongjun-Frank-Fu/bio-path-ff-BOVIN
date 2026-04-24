---
title: "BOVIN-Bench Training Plan · v2 (Aim 2 · Real ICI Labels)"
author: Nabe (z4fu@ucsd.edu)
version: v2.0
date: 2026-04-22
supersedes: TRAINING-PLAN-v1.md (已废弃 · 见 §0)
parent_plan: BOVIN-Pathway-Demo-PLAN.md (Aim 1 demo)
graph_prior: BOVIN-Pathway-Graph-v0.md (82 nodes / 99 edges)
engine: bovin-pathway-demo/ (v0.1-demo · 已交付)
new_data: bovin-bench/ cohorts (Riaz / Hugo / Gide / Hammerman / Cloughesy / Seo)
target_duration: 2–3 周（单人 20h/w）
stack: **不变**（Python 3.11 + PyTorch 2.3 + torch-geometric 2.5 + Lightning 2.2）
status: ready-for-review（本文档要 Nabe 审阅后再动工）
---

# §0 · 为什么有 v2（v1 错在哪）

v1 错在**偷偷换模型 + 忽略既有代码库**。具体：

1. v1 提出 LASSO + GBM 作为主模型，**把 `bovin-pathway-demo/` 里已经跑通的 HeteroGNN 从 headline 降到"以后再说"**——这是对 BOVIN 核心主张（图结构有预测力）的架构级倒退。
2. v1 提了一个从零的建模 pipeline 工作量（2 周），**而既有 `bovin-pathway-demo/` 已经交付了 v0.1-demo**：HeteroGNN + Lightning + Captum IG + bootstrap metrics + LUAD 零样本，61 tests passed，Docker 可复现。v1 等于把 2 周已完成的工程**再做一遍更弱的版本**。
3. v1 的 "BaselineMLP ablation" 在 demo 里**已经有了**（`bovin_demo/model/baseline_mlp.py`），M4 DoD 已验证 GNN 比 MLP 强 +0.0340。LASSO 是 MLP 的更弱版本，不提供新信息。

**v1 作为计划的错误定性**：**Premature re-invention**——没查既有资产就提了替代方案。

**v2 的原则**：**Reuse the engine, swap the label.** Aim 2 就是 v0.1-demo 的 README 明写的"swaps the surrogate label for IMvigor210"——我们把它扩展成**6-cohort 真 ICI 标签 pool**。

---

# §1 · v2 原则：Reuse the engine, swap the label

**保持不动的（v0.1-demo 的全部）**：

| 层 | 组件 | 文件 |
|---|---|---|
| 图 | GraphDict loader / to_heterodata | `graph/loader.py`, `graph/heterodata.py` |
| 模型 | HeteroGNN (GATv2 × 2 + HGT × 1) | `model/hetero_gnn.py` |
| Readout | ModuleAttentionPool | `model/readout.py` |
| Baseline | BaselineMLP (graph-blind ablation) | `model/baseline_mlp.py` |
| Train | LitBovinModule + run_training + 3-seed sweep | `train/lit_module.py`, `train/loop.py` |
| XAI | Captum IG + module rollup + heatmap | `xai/*` |
| Eval | bootstrap metrics + report.md | `eval/metrics.py`, `eval/report.py` |
| CLI | `bovin-demo train / xai / eval` | `cli.py` |

**新增的（v2 只动这些）**：

| 层 | 新文件/修改 | 原因 |
|---|---|---|
| 数据 | `data/ici_loader.py` ← 对偶 `tcga_loader.py` | 读 6 个 ICI cohort 的 processed matrix |
| 数据 | `data/cohort_manifest.py`（小） | 把 `bovin-bench/manifest.yaml` 读成 dict，驱动 loader |
| 标签 | `data/labels.py` 新增 `recist_binary_label()` | Real RECIST CR/PR→1, SD/PD→0，替代 `icd_readiness_label` |
| Split | `data/split.py` 新增 `leave_one_cohort_out()` | 从 stratified 60/20/20 升级到 LOCO |
| Eval | `eval/loco_transfer.py` ← 对偶 `eval/luad_transfer.py` | LOCO 报告生成 + Sade-Feldman 外部验证 |
| Config | `configs/ici_pool.yaml` | label.kind=real, data.source=ici_pool |
| 测试 | `tests/test_ici_loader.py`, `tests/test_recist_label.py` | 守住新代码 |

**`bovin-bench/` 的新定位**：**数据与 manifest 层**，不含任何训练代码。
- `bovin-bench/manifest.yaml` — 每 cohort 的 accession、下载 URL、raw 文件名、字段映射、SHA256
- `bovin-bench/cohorts/<cohort>/` — 每 cohort 的 README + raw/ + processed/
- `bovin-bench/schema/{features,labels,splits}.json` — 契约

**工程量对比**：

| 事项 | v1 估算 | v2 实际 |
|---|---|---|
| 模型代码 | 从零写 LASSO + GBM pipeline | **零行新代码**（复用 HeteroGNN） |
| 训练循环 | 从零写 CV + bootstrap | **零行新代码**（复用 LitBovinModule） |
| 数据层 | 写 6 个 loader + pooled matrix | 写 1 个 `ici_loader.py`（~200 行）+ 6 个 `tools/download_<cohort>.sh` |
| 评估 | 从零写 LOCO 评估器 | 写 `leave_one_cohort_out()`（~50 行）+ `loco_transfer.py`（~150 行） |
| **总计** | **~2–3 周从零搭** | **~2 周，大部分在数据** |

---

# §2 · Aim 2 范围（数据 + 标签 + 划分）

## 2.1 数据 · 6-cohort bulk RNA-seq pool

**从 Step 0 核查（STEP0-FINDINGS.md）确定的 Tier A**：

| # | Cohort | Accession | 瘤种 | N 患者 | 治疗 | 格式 |
|---|---|---|---|---|---|---|
| 1 | Riaz 2017 | GSE91061 | 黑色素瘤 | **65** | nivolumab | raw counts (hg19) |
| 2 | Hugo 2016 | GSE78220 | 黑色素瘤 | **27** | pembro / nivo | FPKM |
| 3 | Gide 2019 | ENA PRJEB23709 | 黑色素瘤 | **74** | pembro / nivo / ipi+nivo | raw counts (salmon quant) |
| 4 | Hammerman 2020 | GSE165278 | 黑色素瘤 | **21** | ipilimumab | TPM / counts (查 supp) |
| 5 | Cloughesy 2019 | GSE121810 | GBM | **29** | neoadjuvant pembro | counts |
| 6 | Seo 2020 | GSE165252 | 食管腺癌 | **40** | atezolizumab | counts |
| **合计** | | | **~256 患者** | | |

**外部验证 cohort**（不入训练 pool）：
- Sade-Feldman 2018 · GSE120575 · **scRNA-seq pseudobulk** · 48 患者 · anti-PD-1/CTLA-4/combo
- **CRT/HSP 信号受 CD45⁺ 过滤压制**——demo plan §3.2 已经指出过这类 caveat，Sade 保留是为了跨模态压力测试，预期 AUC 明显下降

**明确不在 v2 范围**（Tier B/C）：
- IMvigor210 (348) · Liu (121) · Van Allen (42) · Braun (311) · Miao (35) · Cristescu (236)
- 上述 6 个 cohort 全部**推迟到 v2.1**（Tier B 接入 IMvigor210 R 包）与 **v2.2**（Tier C 走 dbGaP 申请）

## 2.2 标签 · RECIST 二值化 · 新 `recist_binary_label()`

替换 `labels.py` 的 `icd_readiness_label()` 作为主标签。接口保持：

```python
# bovin_demo/data/labels.py（新增）
def recist_binary_label(
    clinical: pd.DataFrame,
    response_col: str = "response_raw",
    mapping: dict[str, float] | None = None,
) -> tuple[pd.Series, LabelReport]:
    """
    Primary label for Aim 2. Maps RECIST BOR → {1, 0, NaN}.

    Default mapping:
        CR, PR, Complete Response, Partial Response  → 1.0
        SD, PD, Stable Disease, Progressive Disease  → 0.0
        NE, NA, Not Evaluable, None                  → NaN (caller filters)

    Returns (label, report) — 与 icd_readiness_label 同契约。
    """
```

**辅助标签**（记录不训练）：
- `label_dcb` — Durable Clinical Benefit（Hugo 原文使用的定义）
- `label_os12mo` — OS > 12 个月（Cloughesy 这种 neoadjuvant 可能没 RECIST，用 OS 代理）

**保留 `icd_readiness_label()` 作为 legacy**——COAD demo 的 surrogate 是整个工程的起点，不能删。通过 `configs/*.yaml::label.kind` 切换。

## 2.3 特征对齐 · 跨 cohort normalization

v2 在 `PatientGraphDataset` 之前加一层 **"per-cohort z-score"**：

```
每个 cohort 独立做 gene-level z-score
    ↓
合并成 pooled_expr (256 patients × ~70 BOVIN 节点基因)
    ↓
PatientGraphDataset 内部再做一次全局 z-score（跨 pooled）
    ↓
HeteroData
```

**两次 z-score 的意义**：
- 第一次（per cohort）去掉批次均值/方差偏差
- 第二次（pooled）让 GNN 看到的输入分布稳定，和 `bovin_demo/data/dataset.py` line 58 的既有逻辑保持一致

**不做**：ComBat / scVI / Harmony——demo plan §7 风险表就决定了"批次矫正留后续版本"。v2 继承这个决定。**如果** v2 跑下来 LOCO AUC 比 pooled AUC 差 > 0.15，再考虑加 ComBat。

## 2.4 划分 · Leave-One-Cohort-Out (LOCO)

替代 demo 的 `stratified_split(60/20/20)`（适合单 cohort）。

```python
# bovin_demo/data/split.py（新增）
def leave_one_cohort_out(
    cohort_ids: pd.Series,
    labels: pd.Series,
    *,
    holdout_cohort: str,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Split:
    """
    Train on all cohorts except holdout; test on holdout.
    Within train, carve out val_frac for early-stopping (stratified by label).
    """
```

**6 LOCO fold**：每个 fold 把 1 个 cohort 留做 test，其他 5 个做 train。主指标是 **6 个 per-fold AUC + 均值 ± SD**。

---

# §3 · 目标指标 + DoD（Aim 2 现实版）

## 3.1 为什么不能照搬 demo 的 val-AUC 0.966

Demo 的 0.966 是在**代理标签**（CRT/HMGB1/HSP signature 中位数切分）上跑出来的。**标签本身就是基因的函数**——GNN 实质上是在学一个已知的 z-score 公式。这不算造假（demo plan §3.2 已经明说），但**不能作为 Aim 2 的基线预期**。

真 RECIST 标签有：
- 临床评估的主观误差（同一片子不同 reviewer ± 10% 灵敏度差异）
- SD 边界 case（肿瘤缩小 29% 算 SD，缩小 30% 算 PR）
- 响应率随机性（40% 左右但 cohort 间差很大：Cloughesy GBM ~20%，Gide combo ~50%）

**Realistic AUC target（参考领域同类工作）**：

| 方法 | 在哪个 cohort | 报告 AUC |
|---|---|---|
| Auslander IMPRES 2018 | Hugo | 0.83 |
| Auslander IMPRES 2018 | Riaz | 0.72 |
| Cristescu T-cell-inflamed 2018 | 多 cohort pooled | 0.70 |
| Gide Δ-signature 2019 | Gide on-treatment | 0.85 |
| TIDE (Jiang 2018) | Hugo | 0.75 |
| TIDE | Riaz | 0.72 |

**Aim 2 合理 DoD**：

```
[ ] A2-DoD #1 · Pooled train+test 5-fold CV mean AUC ≥ 0.65
[ ] A2-DoD #2 · LOCO 6-fold mean AUC ≥ 0.60（容忍 cohort 异质性）
[ ] A2-DoD #3 · HeteroGNN 比 BaselineMLP pooled AUC 高 ≥ 0.03（与 demo DoD #3 对齐）
[ ] A2-DoD #4 · Sade-Feldman 外部 AUC ≥ 0.55（CD45 caveat）
[ ] A2-DoD #5 · IG top-5 节点中至少包含 CRT/HMGB1/CD47/CD24 中的 2 个
[ ] A2-DoD #6 · IG top-3 模块包含 M4 DAMP 或 M6（ICD effector）
[ ] A2-DoD #7 · 3-seed run（42/1337/2024）LOCO AUC SD < 0.08
[ ] A2-DoD #8 · report.md 自动生成 + demo_card 更新 + docker-build 仍绿
```

达到 ≥ 6 条可以做内部 demo；8 条全绿可以投 preprint 前的内审。

## 3.2 BOVIN 科学假设前置注册

四条假设在训练前固定，不允许训练后改：

| # | 假设 | Pass 条件 |
|---|---|---|
| **H1** | ICD 轴（CRT/HMGB1/HSP70/HSP90）节点的 IG attribution 方向一致为正（= responder 升高） | ≥ 3/4 节点的 mean(IG per patient) > 0，bootstrap p < 0.05 |
| **H2** | "Don't eat me" 轴（CD47/CD24/SIRPA）IG 方向为负 | ≥ 2/3 节点 mean IG < 0 |
| **H3** | BOVIN 图结构有增量价值（GNN > MLP） | pooled test AUC 差 ≥ 0.03，bootstrap p < 0.05 |
| **H4** | LOCO 跨 cohort 仍保留判别力（不是批次效应过拟合） | LOCO mean AUC ≥ 0.6，per-cohort worst ≥ 0.55 |

H3 不过 = "BOVIN 的图拓扑在真标签上没额外信号"。这个情况的应对在 §7。

---

# §4 · 里程碑 · Aim 2 M1–M7（2–3 周）

**命名约定**：Aim 1 demo 是 M0–M6（已交付 v0.1-demo）。Aim 2 是 **A2-M1 到 A2-M7**，接在后面。

## A2-M1 · Data Acquisition（2–3 天）

- **A2-T1.1**：写 `tools/download_ici_pool.sh`——6 个 cohort 的 wget/curl + SHA256 校验（3h）
- **A2-T1.2**：Gide ENA PRJEB23709 单独处理——可能需要 FTP 拉 fastq 或 use author-provided processed matrix（4h · 最大风险）
- **A2-T1.3**：每 cohort 写 `bovin-bench/cohorts/<cohort>/DATACARD.md`——URL、时间戳、MD5、license、字段映射（3h）
- **A2-T1.4**：`bovin-bench/manifest.yaml` 更新——6 个 cohort 的 accession + response_col + treatment_col 字段名（2h）

**DoD**：
- [ ] `data/raw_ici/<cohort>/` 下 6 个 cohort 原始文件齐备
- [ ] `bovin-bench/manifest.yaml` 驱动 loader 能定位到每一份文件
- [ ] SHA256 写入 `data/checksums.txt`

## A2-M2 · ICI Loader（3 天）

- **A2-T2.1**：`bovin_demo/data/ici_loader.py` 定义 `ICIBundle`（对偶 `CoadBundle`） + `load_ici_cohort(cohort_id)` (4h)
- **A2-T2.2**：Gene harmonization——读 6 个 cohort，HGNC 对齐到 BOVIN 70 observable node 基因列表（复用 `gene_mapping.py`）（3h）
- **A2-T2.3**：Per-cohort z-score + pooled concat → `load_ici_pool() -> ICIPoolBundle`（3h）
- **A2-T2.4**：`labels.py::recist_binary_label()`（2h）
- **A2-T2.5**：`tests/test_ici_loader.py`——命中率 ≥ 70%、每 cohort 响应率 reasonable、pooled N = 256 ± 5（3h）

**DoD**：
- [ ] `ici_pool.expr` shape = (256, ~70)，missing rate < 20%
- [ ] `ici_pool.labels` 去 NaN 后保留 ≥ 230 患者
- [ ] 每 cohort 单独查看的响应率与原文 ±5% 吻合
- [ ] pytest 绿

## A2-M3 · Config + Split（半天）

- **A2-T3.1**：`configs/ici_pool.yaml`——继承 default，覆写 `data.source=ici_pool`, `label.kind=recist_binary`（1h）
- **A2-T3.2**：`split.py::leave_one_cohort_out()` + 兼容 demo 的 `stratified_split`（3h）
- **A2-T3.3**：`tests/test_split.py` 新增 LOCO 测（1h）

**DoD**：
- [ ] `bovin-demo sanity --config configs/ici_pool.yaml` 跑通
- [ ] LOCO 6 fold 的 train/val/test 样本数打印合理

## A2-M4 · Training Run（1–2 天）

- **A2-T4.1**：`bovin-demo train --config configs/ici_pool.yaml --seeds 42,1337,2024` 跑单 70/15/15 stratified 版本（2h）
- **A2-T4.2**：LOCO 6-fold 全跑一次（~6 × 30 min = 3h compute，可并行）
- **A2-T4.3**：BaselineMLP 对照也跑一次 LOCO（为 H3）（2h）
- **A2-T4.4**：记录所有 metrics 到 `outputs/<ts>/loco/fold_<cohort>.json`（1h）

**DoD**：
- [ ] A2-DoD #1 pooled AUC ≥ 0.65 过
- [ ] A2-DoD #2 LOCO mean AUC ≥ 0.60 过
- [ ] A2-DoD #3 vs BaselineMLP gap ≥ 0.03 过（或明确失败）

## A2-M5 · External Validation on Sade-Feldman（1 天）

- **A2-T5.1**：Sade-Feldman scRNA 走 **patient-level pseudobulk**（参考 `cohorts/sade_feldman_gse120575/README.md` 已有方案）（3h）
- **A2-T5.2**：写 `eval/loco_transfer.py::run_external_validation()` 对偶 `luad_transfer.py`——加载 best ckpt、只 forward、算 AUC + CI（3h）
- **A2-T5.3**：`bovin-demo eval --run-dir <path> --external sade_feldman` 一条命令（2h）

**DoD**：
- [ ] Sade-Feldman 48 patients 通过 pipeline 不报错
- [ ] A2-DoD #4 external AUC ≥ 0.55 过（或明确失败并标注 CRT 信号缺失）

## A2-M6 · XAI + Hypothesis Tests（1 天）

- **A2-T6.1**：IG 在新训练好的 checkpoint 上跑（复用 `xai/runner.py`，零新代码）（1h）
- **A2-T6.2**：新 `eval/hypothesis_tests.py`——H1/H2 按 IG 方向性判定、H3 按 AUC bootstrap、H4 按 LOCO（3h）
- **A2-T6.3**：输出 `hypothesis_results.json`（1h）

**DoD**：
- [ ] A2-DoD #5 + #6 sanity 节点/模块过
- [ ] H1-H4 每条有 pass/fail + 证据数字

## A2-M7 · Report + Release（1 天）

- **A2-T7.1**：扩展 `eval/report.py`——新 section "Aim 2 results" 含 pooled + LOCO + external + 假设表（3h）
- **A2-T7.2**：更新 `docs/demo_card.md`——Aim 2 版本（2h）
- **A2-T7.3**：更新 `README.md` 加 A2 Quickstart 段落（1h）
- **A2-T7.4**：Git tag `v0.2-aim2` + RELEASE_NOTES（1h）

**DoD**：
- [ ] `outputs/<ts>/report.md` 含全部 A2-DoD #1–#8 + 热图
- [ ] Docker 干净 rebuild 能复现 Aim 2 训练
- [ ] `demo_card.md` 一页纸能直接插入 Research Plan §10

---

# §5 · 风险清单（Aim 2 专属）

| # | 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|---|
| R1 | Gide ENA 数据拉不下来或格式过于原始（需要跑 salmon） | 高 | 高 | 先查 Gide 2019 supplementary 有没有 author-provided processed matrix；没有就跳过 Gide，N 降到 182 |
| R2 | HGNC 对齐后 BOVIN 节点命中率 < 70% | 中 | 高 | 命中率 per-cohort 报告；< 70% 的 cohort 降级为 "available nodes subset"，不影响其他 |
| R3 | Pooled AUC < 0.55 | 中 | 高 | H3 降级——先查数据（per-cohort response rate 是否合理、label NaN 比例）；再查 batch 效应；**不**调架构 |
| R4 | LOCO mean AUC < 0.50（随机水平） | 低 | 致命 | 基本是数据错——重新核对 `recist_binary_label` 映射表 |
| R5 | BaselineMLP 比 HeteroGNN 强（H3 反向） | 低 | 中 | 说明图结构无增量，如实报告；在 demo_card 加"v2 observation: flat features sufficient for 256-patient pool"，不掩盖 |
| R6 | Sade-Feldman external AUC < 0.5 | 中 | 低 | CD45 过滤把 ICD 轴打没是预期，报告就行 |
| R7 | Cloughesy GBM 响应率极低（~20%）拉低 pool label balance | 中 | 中 | Per-cohort pos_weight 在 `lit_module.py` 里已经支持——打开 config 开关 |
| R8 | 时间线超 3 周 | 中 | 中 | 砍 Sade-Feldman external（A2-M5）；保 LOCO (A2-M4) 不能砍 |

**硬止损**：A2-M2 结束时若 pooled N < 200 或命中率 < 60%，回到 `ici_loader.py` 重查 gene mapping，不能强推训练。

---

# §6 · 评估协议 · 避免"demo 级偏差"

v2 report 必须同时报告以下指标，**缺一不可**：

## 6.1 主指标

| 指标 | 出处 | 意图 |
|---|---|---|
| Pooled 5-fold CV AUC ± bootstrap 95% CI | `eval/metrics.py` 既有函数 | 最好情况（和 demo 同范式） |
| **LOCO 6-fold AUC（per-cohort + mean）** | 新 `eval/loco_transfer.py` | 真实泛化 |
| External AUC on Sade-Feldman | 对偶 `luad_transfer.py` | 跨模态压力测试 |
| HeteroGNN – BaselineMLP gap | 复用 demo eval | H3 |

## 6.2 二级指标

- AUPRC（响应率偏差大时更稳健）
- Brier score + ECE（校准度）
- Per-cohort AUROC（防止某大 cohort 绑架整体）

## 6.3 可解释性报告

- IG top-10 节点 + top-3 模块（复用 `xai/aggregate.py`）
- SHAP on BaselineMLP（新增，用于 H3 证据——看 flat 模型关注哪些基因 vs GNN 关注哪些节点）

## 6.4 Sanity 基线（一起跑，不能砍）

- Null model（均值响应率）
- PD-L1 单基因（CD274 表达 → logistic）
- CD8A 单基因（T 细胞代理）
- BaselineMLP（graph-blind ablation，demo 已有）

---

# §7 · 如果 H3 不过怎么办（诚实预案）

H3 = "GNN 比 MLP 强 ≥ 0.03"——这是 BOVIN 的核心 claim。如果在真标签上 gap < 0.03 甚至反向：

**绝对不做**：
- 改 GNN 架构去赢 H3（这是作弊）
- 调 dropout / lr 到 AUC 最高
- 挑 cohort 子集让 H3 过

**应该做**：
1. **如实报告** gap 数字，不隐藏
2. **诊断**：是批次效应主宰？（看 LOCO vs pooled 的差距）还是样本量不足？（看 per-seed 方差）
3. **推迟 claim**：demo_card 改写成 "BOVIN pathway structure 在 256-patient bulk ICI pool 上未显示对 flat gene-set 的统计显著增量；v2.1 扩到 1,000+ 后重测"
4. **保留发现**：v2 的 LOCO + IG 报告仍然有价值——是"BOVIN 基因集在真 ICI 上的预测力"证据，只是不含"图结构额外价值"证据

**负面结果也是论文**——Ravi 2023 NSCLC integrated cohort 就是用 1,000+ 患者报告 "gene expression alone does not strongly predict ICI response"。

---

# §8 · 扩展接口（Aim 2 之后立刻能接的 3 件事）

## 8.1 Aim 2.1 · Tier B · 加 IMvigor210 (+348 患者)

- 新增 `configs/ici_pool_tier_b.yaml`（继承 `ici_pool.yaml`，加 imvigor210 到 manifest）
- `ici_loader.py::load_ici_cohort("imvigor210")` 走 R 包 `IMvigor210CoreBiologies`（pypeR 或离线 export）
- 跨瘤种——训练 pool 变成 60% urothelial + 40% melanoma
- **需要新增** `cohort_disease` 作为 stratification 列（不是协变量）

## 8.2 Aim 2.2 · Tier C · dbGaP 数据（Liu / Braun / Miao）

- DUC 申请流程（Nabe 以 UCSD 身份走）
- N 扩到 1,000+，能支撑深度模型对比（demo plan §9.1 的"多模态注入"接口就派上用场）

## 8.3 Aim 2.3 · On-treatment 时序（Riaz + Gide 都有）

- demo plan §9.3 已留接口：`HeteroData` 每 node feature 改成 `(T, d)`
- Riaz 的 pre→on Δ-signature 是 Gide 2019 原文报告 AUC 0.85 的关键
- 可以显著提 pooled AUC，但引入时序 confounder——留到 v2.3

---

# §9 · 明确不在 v2 的事（再强调一次）

- 深度模型超过 HeteroGNN（如 GraphSAGE, GIN, Transformer-based）——保持 demo 既有选型
- 多模态（WSI / H&E / mutations）——demo plan 已留接口
- Batch correction beyond z-score（ComBat / scVI）——除非 R3 触发
- 任何临床决策曲线（DCA）——留给 v3 临床合作时做
- Target-trial emulation / 因果推断——需要 RCT cohort（Braun, POPLAR-OAK），v2 只有观察性
- Per-瘤种 / per-药独立模型——N 不够，统一模型 + cohort_id stratification 即可

---

# §10 · 审阅 checklist（Nabe 过目）

先请你过一遍这 10 条，我对每条做一个快速回应，确认全绿再动工：

```
[ ] 1. §0 对 v1 的错误定性你同意（premature re-invention + 忽略 bovin-pathway-demo）
[ ] 2. §1 的 "Reuse engine, swap label" 原则你同意（不动模型代码，只加数据 + 标签 + LOCO）
[ ] 3. §2.1 的 6-cohort Tier A pool 和你的预期一致（Riaz+Hugo+Gide+Hammerman+Cloughesy+Seo = 256）
[ ] 4. §2.2 的 RECIST CR/PR vs SD/PD 二值化 映射你接受（允许 caller 通过 config 重写 mapping）
[ ] 5. §2.4 的 LOCO 6-fold 作为主评估你同意（不是普通 k-fold）
[ ] 6. §3.1 的"demo 的 0.966 不能当 Aim 2 baseline"你理解（surrogate label 本质）
[ ] 7. §3.1 的 A2-DoD #1–#8 目标数字你觉得合理（pooled ≥ 0.65, LOCO ≥ 0.60）
[ ] 8. §3.2 的 H1–H4 假设前置注册你接受（训练前固定，训练后不改）
[ ] 9. §4 的 7 个里程碑 2–3 周时间线你认可
[ ] 10. §7 "H3 不过怎么办" 你能接受诚实报告而不是调架构
```

有任何一条划叉，我们先讨论再动工；全绿我立刻进 A2-M1。

---

# 附录 A · v2 与 v1 的 diff 表

| 维度 | v1 (废弃) | v2 |
|---|---|---|
| **主模型** | LASSO + XGBoost + LightGBM + RF | **HeteroGNN**（复用 demo） |
| **Baseline** | null / PD-L1 / CD8A / IMPRES | BaselineMLP（复用 demo）+ null / PD-L1 / CD8A |
| **代码位置** | 新建 `bovin-bench/scripts/train/` | **全在 `bovin-pathway-demo/`**（只加 `ici_loader.py` + 1 个 config） |
| **工作量** | 2 周（从零搭 pipeline） | 2 周（大部分在数据层） |
| **Feature 空间** | 70 × 1 flat vector | 82 节点 × 99 边 HeteroData（+ ModuleAttentionPool） |
| **失图结构** | 是（LASSO / GBM 无法用边） | 否（HGT + GATv2 原生使用边） |
| **能测"图结构有价值"** | 否 | 是（H3） |
| **与既有 demo 兼容** | 否 | 是（share cli, share ckpt 格式, share eval/report） |

---

# 附录 B · 给评审的 3 句话预演（Aim 2 版）

> "Demo 阶段我们在 TCGA-COAD 代理标签上证明 BOVIN 的 82-node HeteroGNN 架构可跑、可解释、比 flat MLP 强 +0.03；
> Aim 2 是同一架构、同一训练循环**只换标签**——从 surrogate 换成 6 个公开 ICI cohort 的 **真 RECIST 响应** 标签，**共 256 患者**；
> 我们想测三件事：(1) 图结构在真标签上还有增量吗（H3），(2) 跨 cohort 泛化保不保得住（LOCO），(3) IG 能不能在不同癌种上自动recover 出 ICD → antigen-presentation 轴（H1/H2）。
> 不追求 SOTA，诚实给出 gap/AUC/泛化三个数字，为 Aim 2.1 的 Tier B 扩容（加 IMvigor210）打底座。"

---

*BOVIN-Bench Training Plan v2 · Aim 2 · 2026-04-22*
*v1 已作废 · v2 原则 = Reuse engine, swap label · 请 Nabe §10 全绿后动工*
