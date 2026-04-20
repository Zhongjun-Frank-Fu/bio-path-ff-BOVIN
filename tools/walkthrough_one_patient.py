"""Walk ONE patient through the full pipeline — print real numbers at each step.

Run inside the docker container:
    python tools/walkthrough_one_patient.py
"""

from __future__ import annotations

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


def main() -> None:
    banner("STEP 0: 原始 TCGA RNA-seq")
    bundle = load_coad("data/raw")
    print(f"全队列: {bundle.n_samples} 病人 × {bundle.n_genes} 基因")

    PATIENT = bundle.expr.index[0]
    print(f"挑的示例病人: {PATIENT}")

    label_genes = ["CALR", "HMGB1", "HSPA1A", "HSP90AA1", "CD47", "CD24"]
    print("\n这个病人在 6 个 label 基因上的 log2(RSEM+1):")
    for g in label_genes:
        v = bundle.expr.loc[PATIENT, g]
        print(f"  {g:10s} = {v:.4f}")

    banner("STEP 1: 算 surrogate label (z-score signature + 中位数切)")
    label, lrep = icd_readiness_label(bundle.expr)
    print(f"全队列: pos_rate={lrep.pos_rate:.3f}  threshold={lrep.threshold:+.4f}")
    y = int(label[PATIENT])
    print(f"\n这个病人的 label: {y}  "
          f"({'ICD-READY' if y == 1 else 'ICD-COLD'})")

    banner("STEP 2: 对齐到 BOVIN 图的 70 个 observable 节点")
    graph = load_graph()
    aligned, report = map_to_pathway_nodes(bundle.expr, graph)
    print(f"Alignment hit_rate = {report.hit_rate:.3f}")
    print(f"Miss 掉的节点: {report.misses}")

    print(f"\n这个病人在几个代表性 pathway 节点上的对齐值:")
    for nid in ["crt", "hmgb1", "cd8a", "pd1", "pdl1", "hsp70"]:
        print(f"  {nid:10s} = {aligned.loc[PATIENT, nid]:+.4f}")

    banner("STEP 3: 组成 HeteroData (每病人一张图)")
    ds = build_patient_dataset(graph, aligned, label)
    idx = list(ds.sample_ids).index(PATIENT)
    data = ds[idx]

    print(f"这张图有 {len(data.node_types)} 种节点类型:")
    for nt in data.node_types:
        x = data[nt].x
        obs = int(data[nt].observable.sum().item())
        print(f"  {nt:12s} shape={tuple(x.shape)}  observable={obs}/{x.size(0)}")

    print(f"\n共 {len(data.edge_types)} 种 (src, rel, dst) 边类型")
    print("前 5 种:")
    for et in list(data.edge_types)[:5]:
        print(f"  {et}  edges={data[et].edge_index.size(1)}")

    damp_idx = list(data["DAMP"].node_ids).index("crt")
    crt_feat = data["DAMP"].x[damp_idx]
    print(f"\nCRT 节点的 2 维特征:")
    print(f"  [z_expr = {crt_feat[0].item():+.4f},  observed_flag = {crt_feat[1].item():.1f}]")
    print(f"  data.y = {int(data.y.item())}")

    banner("STEP 4: 加载训好的 seed=42 ckpt, 跑前向")
    from torch_geometric.loader import DataLoader

    loader = DataLoader([data], batch_size=1)
    probe = next(iter(loader))
    clf = build_classifier(
        probe, hidden_dim=64, num_intra_layers=2, num_inter_layers=1,
        heads=4, dropout=0.25,
    )
    with torch.no_grad():
        clf(probe)  # warm up lazy layers
    clf = _load_checkpoint(
        "outputs/20260420_191607_seed42/ckpt/best.ckpt", clf
    )
    clf.eval()

    with torch.no_grad():
        x_dict_enc = {
            nt: clf.backbone.encoder[nt](data[nt].x)
            for nt in clf.backbone.encoder
        }
        out = clf(data)

    crt_hidden = x_dict_enc["DAMP"][damp_idx]
    print(f"\nEncoder 后 — CRT 的 hidden 表示 (dim={crt_hidden.shape[0]}):")
    print(f"  前 8 维 = {crt_hidden[:8].numpy().round(3).tolist()}")

    logit = out["logit"].item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    correct = (prob >= 0.5) == (y == 1)
    print(f"\n最终预测:")
    print(f"  logit = {logit:+.4f}")
    print(f"  prob  = {prob:.4f}  (ICD-ready 的预测概率)")
    print(f"  label = {y}   {'✅ 正确' if correct else '❌ 错'}")

    banner("STEP 5: 11 个模块的 embedding (最终送给 head 的那 11 个数)")
    module_emb = out["module_emb"]
    MODS = [
        ("M1", "ENTRY"), ("M2", "ISR"), ("M3", "MITO"), ("M4", "DAMP"),
        ("M5", "METAB"), ("M6", "APC_RECV"), ("M7", "DC_MAT"),
        ("M8", "TCELL"), ("M9", "ICB"), ("M10", "MEM"), ("M11", "MAC"),
    ]
    max_abs = max(abs(v.item()) for v in module_emb) or 1.0
    for i, (mid, name) in enumerate(MODS):
        v = module_emb[i].item()
        bar_len = int(abs(v) / max_abs * 30)
        bar = "█" * bar_len
        sign = "+" if v >= 0 else "−"
        print(f"  {mid:3s} {name:9s} {sign} {bar:<30s} {v:+.4f}")

    banner("STEP 6: Pool attention — M4 DAMP 里的权重分布")
    attn = out["attn"]
    damp_attn = attn.get("M4", torch.empty(0))
    damp_ids = [
        (nid, mod) for nt in data.node_types
        for nid, mod in zip(data[nt].node_ids, data[nt].module)
    ]
    damp_node_ids = [nid for nid, mod in damp_ids if mod == "M4"]

    if damp_attn.numel() > 0:
        pairs = sorted(zip(damp_node_ids, damp_attn.tolist()), key=lambda x: -x[1])
        print(f"M4 DAMP 模块有 {len(damp_node_ids)} 个节点, 归一化 attention (和 = 1):")
        for nid, w in pairs:
            bar = "▓" * int(w * 40)
            print(f"  {nid:10s} {bar:<40s} {w:.3f}")
        print(f"\n解读: 这个病人的 DAMP 模块 embedding 主要被 '{pairs[0][0]}' 拉动")

    banner("STEP 7: head 怎么把 11 维 → 1 个 logit")
    W = clf.head.weight.detach().squeeze()
    b = clf.head.bias.detach().item()
    print(f"head.weight (11 维): {W.numpy().round(3).tolist()}")
    print(f"head.bias:           {b:+.4f}")
    contributions = W * module_emb
    print(f"\n每个模块对 logit 的贡献 (w_i × emb_i):")
    for i, (mid, name) in enumerate(MODS):
        c = contributions[i].item()
        print(f"  {mid:3s} {name:9s} {c:+.4f}")
    print(f"\n求和 + bias = {contributions.sum().item() + b:+.4f}  "
          f"≈ logit {logit:+.4f} ✓")


if __name__ == "__main__":
    main()
