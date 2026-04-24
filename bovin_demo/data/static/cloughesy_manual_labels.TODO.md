# Cloughesy 2019 · Manual labels TODO (M2.1)

## Why this file exists

Cloughesy TF et al. *Nat Med* 2019;25(3):477-486 reports neoadjuvant vs adjuvant
pembrolizumab in recurrent glioblastoma (GBM). GEO `GSE121810` publishes the
RNA-seq but **not** per-patient response / OS / PFS. The paper's Table 1 only
gives baseline characteristics; the Figure 1b swimmer plot has per-patient
bars but no readable numerical labels. PMC supplementary Table S1
(`NIHMS1516903-supplement-Table_S1.pdf`) likewise only carries baseline
covariates.

Result: the loader has all 29 expression samples but **zero labels**, so
`load_ici_pool(require_label=True)` drops this cohort entirely. The Aim 2
pooled labeled N stays at ~166 instead of ~195 (plan §4 A2-M2 DoD target 230
assumes all 6 cohorts labeled — 195 would be the realistic cap *with* all
Cloughesy fillable).

## How to fill it

Contact the corresponding authors of Cloughesy 2019:

- **Timothy Cloughesy** — `tcloughesy@mednet.ucla.edu`
- **Robert Prins**     — `rprins@mednet.ucla.edu`

Ask for the per-patient OS / PFS / response mapping that backs Figure 1b's
swimmer plot. Either works — once you have it, fill `cloughesy_manual_labels.csv`:

| column | how to fill |
|---|---|
| `patient_id` | already populated (29 rows: Pt2, Pt3, …, Pt35) |
| `arm` | already populated (`neoadj` = pre-surgery pembro; `adjuvant` = post-surgery pembro only). Derived from sample-ID suffix `_B` / `_A` respectively. Verify against author answer. |
| `os_months` | overall survival in months (days ÷ 30.44) |
| `pfs_months` | progression-free survival in months |
| `response_raw` | either a RECIST code (CR/PR/SD/PD) if the authors used RANO/RECIST, or a plain tag like `OS>12mo` / `OS<=12mo` |
| `ici_response` | binary 0/1. Default rule (plan §2.2): **OS > 12 months → 1, else → 0**. Override if authors supply a different response call. |
| `source_note` | short pointer, e.g., "author email 2026-05-01" |

Once filled, commit the CSV. `ici_loader._load_cloughesy` already reads this
file (if non-null `ici_response` is present) and merges into the bundle's
`clinical` frame; no loader code change needed.

## Interim behaviour (before fill)

With this file as-is (all `ici_response = NaN`):

- `load_ici_cohort("cloughesy_gse121810").clinical["ici_response"]` is all NaN
- `load_ici_pool(require_label=True)` filters Cloughesy out
- `load_ici_pool()` (unfiltered) still includes Cloughesy's 29 samples for
  per-cohort z-score + gene-intersection purposes
- All existing tests pass (the filtered-pool threshold is 160, which is met
  by the 5 cohorts that do have labels)

## Scientific rationale for OS>12mo threshold

Plan §2.2 picks 12 months because:

1. Cloughesy's own median split is at 13.7 months (neoadj) / 7.5 months (adjuvant);
2. GBM's historical median OS under standard of care is ~14 months; patients
   exceeding 12 months on recurrent GBM + ICI are meaningfully "responding";
3. RECIST-equivalent (mRANO) is unreported in this cohort — OS is the only
   published endpoint per patient.

If authors supply RANO-based classifications instead, prefer those and
document the override in `source_note`.
