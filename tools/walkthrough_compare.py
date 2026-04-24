"""Run the single-patient walkthrough on a chosen patient.

Usage inside docker:
    python tools/walkthrough_compare.py --patient TCGA-CA-5256-01
    python tools/walkthrough_compare.py --pick-positive   # first label==1

Factors out the STEP 0–7 trace so we can pipe different patients through
the same forward path for side-by-side comparison.
"""

from __future__ import annotations

import argparse

import torch

from bovin_demo.data import (
    icd_readiness_label,
    load_coad,
    map_to_pathway_nodes,
)
from bovin_demo.data.dataset import build_patient_dataset
from bovin_demo.graph import load_graph
from bovin_demo.model import build_classifier
from bovin_demo.xai.runner import _load_checkpoint


def banner(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def run(patient: str, ckpt: str = "outputs/20260420_191607_seed42/ckpt/best.ckpt") -> None:
    banner(f"PATIENT: {patient}")
    bundle = load_coad("data/raw")
    label_genes = ["CALR", "HMGB1", "HSPA1A", "HSP90AA1", "CD47", "CD24"]
    raw = bundle.expr.loc[patient, label_genes]
    print("log2(RSEM+1) on 6 signature genes:")
    for g in label_genes:
        print(f"  {g:10s} = {raw[g]:.4f}")

    label, lrep = icd_readiness_label(bundle.expr)
    from bovin_demo.data.labels import icd_readiness_signature
    score, _ = icd_readiness_signature(bundle.expr)
    print(f"\ncohort threshold = {lrep.threshold:+.4f}")
    print(f"this patient's signature score = {score[patient]:+.4f}")
    print(f"label = {int(label[patient])} "
          f"({'ICD-READY' if label[patient] == 1 else 'ICD-COLD'})")

    graph = load_graph()
    aligned, _ = map_to_pathway_nodes(bundle.expr, graph)
    common = aligned.index.intersection(label.index)
    aligned, label = aligned.loc[common], label.loc[common]

    ds = build_patient_dataset(graph, aligned, label)
    idx = list(ds.sample_ids).index(patient)
    data = ds[idx]

    damp_idx = list(data["DAMP"].node_ids).index("crt")
    crt_feat = data["DAMP"].x[damp_idx]
    print(f"\nCRT's 2-d feature: [z_expr={crt_feat[0].item():+.4f}, "
          f"observed={crt_feat[1].item():.1f}]")

    from torch_geometric.loader import DataLoader
    loader = DataLoader([data], batch_size=1)
    probe = next(iter(loader))
    clf = build_classifier(
        probe, hidden_dim=64, num_intra_layers=2, num_inter_layers=1,
        heads=4, dropout=0.25,
    )
    with torch.no_grad():
        clf(probe)
    clf = _load_checkpoint(ckpt, clf)
    clf.eval()

    with torch.no_grad():
        out = clf(data)

    logit = out["logit"].item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    y = int(label[patient])
    correct = (prob >= 0.5) == (y == 1)
    print(f"\nforward pass:")
    print(f"  logit = {logit:+.4f}  prob = {prob:.4f}  "
          f"{'✅' if correct else '❌'}")

    banner("MODULE EMBEDDING (11 numbers → head)")
    module_emb = out["module_emb"]
    MODS = [("M1","ENTRY"),("M2","ISR"),("M3","MITO"),("M4","DAMP"),
            ("M5","METAB"),("M6","APC_RECV"),("M7","DC_MAT"),("M8","TCELL"),
            ("M9","ICB"),("M10","MEM"),("M11","MAC")]
    max_abs = max(abs(v.item()) for v in module_emb) or 1.0
    for i, (mid, name) in enumerate(MODS):
        v = module_emb[i].item()
        bar_len = int(abs(v) / max_abs * 30)
        bar = "█" * bar_len
        sign = "+" if v >= 0 else "−"
        print(f"  {mid:3s} {name:9s} {sign} {bar:<30s} {v:+.4f}")

    banner("HEAD CONTRIBUTIONS (weight × embedding)")
    W = clf.head.weight.detach().squeeze()
    b = clf.head.bias.detach().item()
    contrib = W * module_emb
    pairs = [(mid, name, contrib[i].item()) for i, (mid, name) in enumerate(MODS)]
    pairs.sort(key=lambda x: -abs(x[2]))
    print("ordered by |contribution|:")
    for mid, name, c in pairs[:5]:
        arrow = "↑" if c > 0 else "↓"
        print(f"  {mid:3s} {name:9s}  {arrow}  {c:+8.4f}")
    print(f"  ... (其余 6 个模块贡献之和 = {sum(c for _, _, c in pairs[5:]):+.4f})")
    print(f"  bias                      = {b:+.4f}")
    print(f"  合计 = {contrib.sum().item() + b:+.4f}  ≈  logit {logit:+.4f}")

    banner("M4 DAMP POOL ATTENTION")
    attn = out["attn"]
    damp_attn = attn.get("M4", torch.empty(0))
    damp_node_ids = [n for nt in data.node_types
                     for n, m in zip(data[nt].node_ids, data[nt].module) if m == "M4"]
    if damp_attn.numel() > 0:
        rows = sorted(zip(damp_node_ids, damp_attn.tolist()), key=lambda x: -x[1])
        for nid, w in rows:
            bar = "▓" * int(w * 40)
            print(f"  {nid:10s} {bar:<40s} {w:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patient", default=None, help="TCGA sample id; overrides --pick-*")
    p.add_argument("--pick-positive", action="store_true",
                   help="Pick the first label==1 patient")
    p.add_argument("--pick-most-confident-positive", action="store_true",
                   help="Pick the label==1 patient with highest model prob")
    args = p.parse_args()

    bundle = load_coad("data/raw")
    label, _ = icd_readiness_label(bundle.expr)

    if args.patient:
        target = args.patient
    elif args.pick_positive:
        target = next(sid for sid in bundle.expr.index if label.get(sid, 0) == 1)
    elif args.pick_most_confident_positive:
        # Lazy: loop through label==1 patients, find highest model prob
        graph = load_graph()
        aligned, _ = map_to_pathway_nodes(bundle.expr, graph)
        common = aligned.index.intersection(label.index)
        aligned, label2 = aligned.loc[common], label.loc[common]

        ds = build_patient_dataset(graph, aligned, label2)
        from torch_geometric.loader import DataLoader

        probe_i = next(i for i, sid in enumerate(ds.sample_ids) if label2[sid] == 1)
        probe = next(iter(DataLoader([ds[probe_i]], batch_size=1)))
        clf = build_classifier(probe, hidden_dim=64, num_intra_layers=2,
                              num_inter_layers=1, heads=4, dropout=0.25)
        with torch.no_grad():
            clf(probe)
        clf = _load_checkpoint(
            "outputs/20260420_191607_seed42/ckpt/best.ckpt", clf)
        clf.eval()

        best = (None, -1.0)
        for i, sid in enumerate(ds.sample_ids):
            if label2[sid] != 1:
                continue
            with torch.no_grad():
                p = torch.sigmoid(clf(ds[i])["logit"]).item()
            if p > best[1]:
                best = (sid, p)
        target = best[0]
        print(f"[auto-pick] most confident label=1: {target} (prob={best[1]:.3f})")
    else:
        target = bundle.expr.index[0]

    run(target)


if __name__ == "__main__":
    main()
