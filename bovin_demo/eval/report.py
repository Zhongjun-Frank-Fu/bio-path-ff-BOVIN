"""M6 · T6.1 — assemble metrics.json + XAI artifacts into ``report.md``.

``build_report(run_dir)`` is the one-stop "turn a training run directory
into something I can paste into a slide". It:

1. Reads the ``metrics.json`` M4 wrote.
2. Optionally re-runs test inference to compute bootstrap CIs with
   :func:`bovin_demo.eval.metrics.compute_metrics` (500× by default).
3. Reads ``xai/sanity.json`` + ``xai/xai_heatmap.png`` if M5 has been run.
4. Writes ``report.md`` with a DoD checklist at the top.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class InferencePayload:
    y_true: list[int]
    y_prob: list[float]
    sample_ids: list[str]


def _run_test_inference(
    run_dir: Path,
    *,
    config_path: Path,
    raw_dir_override: Path | None,
    seed: int,
) -> InferencePayload:
    """Load the best ckpt + rebuild the dataset, re-score the test fold."""
    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from bovin_demo.data import (
        icd_readiness_label,
        load_coad,
        map_to_pathway_nodes,
        stratified_split,
    )
    from bovin_demo.data.dataset import build_patient_dataset
    from bovin_demo.graph import load_graph
    from bovin_demo.model import build_classifier
    from bovin_demo.train.loop import _load_config
    from bovin_demo.xai.runner import _load_checkpoint

    cfg = _load_config(config_path)
    raw_dir = Path(raw_dir_override or cfg.paths.get("raw_dir", "data/raw"))

    graph = load_graph()
    bundle = load_coad(raw_dir)
    aligned, _ = map_to_pathway_nodes(bundle.expr, graph)
    label, _ = icd_readiness_label(bundle.expr)
    common = aligned.index.intersection(label.index)
    aligned, label = aligned.loc[common], label.loc[common]
    split = stratified_split(label, seed=seed)

    dataset = build_patient_dataset(graph, aligned, label)
    test_samples = [dataset[int(i)] for i in split.test_idx]

    probe = next(iter(PyGDataLoader(test_samples[: min(16, len(test_samples))], batch_size=16)))
    clf = build_classifier(
        probe,
        hidden_dim=int(cfg.model.hidden_dim),
        num_intra_layers=int(cfg.model.num_intra_layers),
        num_inter_layers=int(cfg.model.num_inter_layers),
        heads=int(cfg.model.attention_heads),
        dropout=float(cfg.model.dropout),
    )
    with torch.no_grad():
        clf(probe)

    ckpts = sorted((run_dir / "ckpt").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/ckpt")
    clf = _load_checkpoint(ckpts[-1], clf)
    clf.eval()

    probs: list[float] = []
    labels: list[int] = []
    sids: list[str] = []
    loader = PyGDataLoader(test_samples, batch_size=32)
    with torch.no_grad():
        for batch in loader:
            logit = clf(batch)["logit"].view(-1)
            probs.extend(torch.sigmoid(logit).cpu().tolist())
            labels.extend(batch.y.view(-1).to(torch.int64).cpu().tolist())
    for s in test_samples:
        sids.append(str(getattr(s, "sample_id", "")))
    return InferencePayload(y_true=labels, y_prob=probs, sample_ids=sids)


def _dod_checklist(metrics: dict, sanity: dict | None, baseline_gap: float) -> list[tuple[str, bool, str]]:
    val_auc = metrics.get("best_val_auc", float("nan"))
    test_auc = metrics.get("test_auc", float("nan"))

    items: list[tuple[str, bool, str]] = [
        ("DoD #2 · val-AUC ≥ 0.65", val_auc >= 0.65, f"best_val_auc = {val_auc:.4f}"),
        ("DoD #2 · test-AUC ≥ 0.60", test_auc >= 0.60, f"test_auc = {test_auc:.4f}"),
        ("DoD #3 · GNN beats baseline by ≥ 0.03", baseline_gap >= 0.03,
         f"gap = {baseline_gap:+.4f} (test_auc − baseline_test_auc)"),
    ]
    if sanity is not None:
        items.append((
            "DoD #4 · M4 DAMP in top-3 modules",
            bool(sanity.get("dod_4_m4_damp_in_top3_modules", False)),
            f"top-3 modules = {sanity.get('top3_modules', [])}",
        ))
        items.append((
            "DoD #5 · CRT/HMGB1/ST6GAL1 in top-5 nodes",
            bool(sanity.get("dod_5_landmark_in_top5_nodes", False)),
            f"landmarks found = {sanity.get('landmarks_found', [])}",
        ))
    return items


def build_report(
    run_dir: str | Path,
    *,
    config_path: str | Path = "configs/tcga_coad.yaml",
    raw_dir_override: str | Path | None = None,
    bootstrap: int = 500,
    recompute_test: bool = True,
    luad_metrics: dict | None = None,
) -> Path:
    """Produce ``run_dir/report.md`` — return the path.

    Parameters
    ----------
    run_dir : M4 output directory (must contain ``metrics.json`` and
        ``ckpt/*.ckpt``).
    config_path : path used for inference (defaults to TCGA-COAD config).
    bootstrap : passes to :func:`compute_metrics`.
    recompute_test : if True, re-score the test fold with the best ckpt and
        compute bootstrap CIs.
    luad_metrics : optional ``dict`` from ``run_luad_zero_shot`` (T6.4) —
        embedded in the report as a separate section.
    """
    from bovin_demo.eval.metrics import compute_metrics

    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found — run training first")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    sanity_path = run_dir / "xai" / "sanity.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8")) if sanity_path.exists() else None
    heatmap_rel = "xai/xai_heatmap.png" if (run_dir / "xai" / "xai_heatmap.png").exists() else None

    bootstrap_table: dict | None = None
    if recompute_test:
        payload = _run_test_inference(
            run_dir,
            config_path=Path(config_path),
            raw_dir_override=Path(raw_dir_override) if raw_dir_override else None,
            seed=int(metrics.get("seed", 42)),
        )
        bootstrap_table = compute_metrics(
            payload.y_true, payload.y_prob,
            bootstrap=bootstrap, seed=int(metrics.get("seed", 42)),
        )

    baseline_gap = float(metrics["test_auc"]) - float(metrics["baseline_test_auc"])
    dod = _dod_checklist(metrics, sanity, baseline_gap)

    lines: list[str] = []
    lines.append(f"# BOVIN-Pathway Demo · Run Report `{run_dir.name}`")
    lines.append("")
    lines.append("> **Surrogate label** — results are on a CRT/HMGB1/HSP signature, *not* "
                 "on ICI response. Aim 2 switches to IMvigor210. Read everything below "
                 "through that caveat (see `data/DATACARD.md` §Label).")
    lines.append("")

    lines.append("## 0. DoD checklist (PLAN §7)")
    lines.append("")
    lines.append("| Item | Status | Detail |")
    lines.append("|---|---|---|")
    for item, passed, detail in dod:
        mark = "✅" if passed else "❌"
        lines.append(f"| {item} | {mark} | {detail} |")
    lines.append("")

    lines.append("## 1. Run summary")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for k in ("seed", "epochs_run", "num_train", "num_val", "num_test",
              "best_val_auc", "test_auc", "test_loss", "baseline_test_auc"):
        if k in metrics:
            v = metrics[k]
            lines.append(f"| `{k}` | `{v:.4f}` |" if isinstance(v, float) else f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append(f"Baseline (pure-MLP, same split + pos_weight) test_auc: "
                 f"**{metrics['baseline_test_auc']:.4f}** · gap = **{baseline_gap:+.4f}**.")
    lines.append("")

    if bootstrap_table:
        lines.append("## 2. Test-fold metrics (bootstrap 500×, 95% CI)")
        lines.append("")
        lines.append("| Metric | Point | 95% CI |")
        lines.append("|---|---|---|")
        for k, mv in bootstrap_table.items():
            lines.append(f"| {k.upper()} | {mv['mean']:.4f} | [{mv['ci_lo']:.4f}, {mv['ci_hi']:.4f}] |")
        lines.append("")

    if sanity is not None:
        lines.append("## 3. XAI sanity (Captum Integrated Gradients)")
        lines.append("")
        lines.append(f"* Top-3 modules: {sanity['top3_modules']}")
        lines.append(f"* Top-5 nodes: {sanity['top5_nodes']}")
        lines.append(f"* Landmarks recovered: {sanity['landmarks_found']}")
        lines.append(f"* Patients in heatmap: {sanity['n_patients']}")
        if heatmap_rel:
            lines.append("")
            lines.append(f"![XAI heatmap]({heatmap_rel})")
        lines.append("")

    if luad_metrics is not None:
        lines.append("## 4. TCGA-LUAD zero-shot transfer")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in luad_metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## 5. Artifacts")
    lines.append("")
    for name in ("metrics.json", "training.log", "ckpt/best.ckpt",
                 "logs/metrics.csv", "xai/xai_heatmap.png",
                 "xai/node_attributions.csv", "xai/module_summary.csv",
                 "xai/sanity.json"):
        path = run_dir / name
        if path.exists():
            lines.append(f"- `{name}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Generated by `bovin_demo.eval.report.build_report` · "
                 "see `BOVIN-Pathway-Demo-PLAN.md` §5 for milestone context._")
    lines.append("")

    out = run_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", out)
    return out
