"""Re-run IG on the CRT-quartile ckpt and print the full 11-module ranking."""

from __future__ import annotations

from pathlib import Path

import torch

from bovin_demo.data import crt_quartile_label, load_coad, map_to_pathway_nodes, stratified_split
from bovin_demo.data.dataset import build_patient_dataset
from bovin_demo.graph import load_graph
from bovin_demo.model import build_classifier
from bovin_demo.xai.aggregate import aggregate_by_module, rank_nodes
from bovin_demo.xai.ig_captum import compute_node_attributions
from bovin_demo.xai.runner import _load_checkpoint
from torch_geometric.loader import DataLoader

RUN = Path("outputs/20260421_215939_crt_quartile_seed42")

bundle = load_coad("data/raw")
graph = load_graph()
aligned, _ = map_to_pathway_nodes(bundle.expr, graph)
label, _ = crt_quartile_label(bundle.expr)
keep = label.dropna().index
aligned = aligned.loc[keep]
label = label.loc[keep].astype(int)
split = stratified_split(label, seed=42)

ds = build_patient_dataset(graph, aligned, label)
val_samples = [ds[int(i)] for i in split.val_idx]
probe = next(iter(DataLoader(val_samples[:16], batch_size=16)))

clf = build_classifier(probe, hidden_dim=64, num_intra_layers=2, num_inter_layers=1,
                      heads=4, dropout=0.25)
with torch.no_grad():
    clf(probe)
clf = _load_checkpoint(RUN / "ckpt/best.ckpt", clf)
clf.eval()

attr = compute_node_attributions(clf, val_samples, n_steps=20)
roll = aggregate_by_module(attr)
rank = rank_nodes(attr)

MODS = [("M1","ENTRY"),("M2","ISR"),("M3","MITO"),("M4","DAMP"),
        ("M5","METAB"),("M6","APC_RECV"),("M7","DC_MAT"),("M8","TCELL"),
        ("M9","ICB"),("M10","MEM"),("M11","MAC")]
id2name = dict(MODS)

print("\n============ 全 11 模块 mean |IG attribution|（新 CRT-quartile label）============")
idx = {mid: i for i, mid in enumerate(roll.module_ids)}
order = sorted(roll.module_ids, key=lambda m: -roll.mean_per_module[idx[m]])
max_v = max(roll.mean_per_module) or 1.0
for r, mid in enumerate(order, 1):
    v = roll.mean_per_module[idx[mid]]
    bar = "█" * int(v * 30 / max_v)
    marker = "⭐" if r <= 3 else "  "
    print(f"  {marker} #{r:2d}  {mid:3s} {id2name[mid]:9s}  {bar:<30s}  {v:.4f}")

print("\n============ Top-10 节点 ============")
for r, (nid, v, m) in enumerate(zip(rank.node_ids[:10], rank.mean_abs_attr[:10], rank.modules[:10]), 1):
    print(f"  #{r:2d}  {nid:12s}  [{m}]  {v:.4f}")
