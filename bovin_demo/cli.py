"""Command-line entry point.

M0 scope: stubs only. Real subcommands (``train``, ``eval``, ``xai``, ``sanity``)
are wired in M3–M5 per PLAN §5. For now ``bovin-demo --help`` lists the planned
subcommands so the user can see the surface area.
"""

from __future__ import annotations

import argparse
import os
import sys

from bovin_demo import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bovin-demo",
        description="BOVIN-Pathway Demo · Aim 1 minimum closed loop (M0 skeleton)",
    )
    p.add_argument("--version", action="version", version=f"bovin-demo {__version__}")
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("BOVIN_SEED", "42")),
        help="Global random seed (default: 42 or $BOVIN_SEED)",
    )

    sub = p.add_subparsers(dest="command", metavar="command")

    sp_sanity = sub.add_parser("sanity", help="Import + graph loader smoke check (M1)")
    sp_sanity.add_argument("--config", default="configs/default.yaml")

    sp_train = sub.add_parser("train", help="Train HeteroGNN (M4)")
    sp_train.add_argument("--config", default="configs/tcga_coad.yaml")
    sp_train.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override cfg.train.max_epochs (handy for smoke runs)",
    )
    sp_train.add_argument(
        "--output-root",
        default=None,
        help="Override outputs/ root directory",
    )
    sp_train.add_argument(
        "--raw-dir",
        default=None,
        help="Override data/raw directory (tests point this at a fixture)",
    )
    sp_train.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list for T4.5 stability sweep, e.g. '42,1337,2024'. "
             "If omitted, falls back to --seed.",
    )

    sp_xai = sub.add_parser("xai", help="Integrated-Gradients readout (M5)")
    sp_xai.add_argument(
        "--run-dir",
        required=True,
        help="Path to an M4 run directory (contains ckpt/best.ckpt)",
    )
    sp_xai.add_argument("--config", default="configs/tcga_coad.yaml")
    sp_xai.add_argument("--raw-dir", default=None)
    sp_xai.add_argument("--top-n", type=int, default=20,
                        help="Number of top-TPR patients rendered in the heatmap")
    sp_xai.add_argument("--n-steps", type=int, default=20,
                        help="Captum IG interpolation steps (config xai.n_samples)")

    sp_eval = sub.add_parser("eval", help="Evaluate + produce report.md (M6)")
    sp_eval.add_argument("--run-dir", required=True,
                         help="Path to an M4 run directory (contains metrics.json + ckpt/best.ckpt)")
    sp_eval.add_argument("--config", default="configs/tcga_coad.yaml")
    sp_eval.add_argument("--raw-dir", default=None)
    sp_eval.add_argument("--luad-raw-dir", default=None,
                         help="Enable T6.4 zero-shot transfer: path to data/raw_luad")
    sp_eval.add_argument("--bootstrap", type=int, default=500)
    sp_eval.add_argument("--no-recompute-test", action="store_true",
                         help="Skip re-running test inference (use existing metrics only)")

    return p


def _run_sanity(seed: int) -> int:
    """Run the M1→M3 sanity probe: graph load → HeteroData → forward pass.

    Returns 0 on success. Degrades gracefully (prints and returns 0) if heavy
    dependencies (pydantic / torch / torch-geometric) aren't installed, so
    ``make smoke`` works on a bare ``pip install bovin_demo`` too.
    """
    print(f"[sanity] bovin_demo v{__version__} loaded; seed={seed}")

    try:
        from bovin_demo.graph import load_graph
    except ImportError as exc:
        print(f"[sanity] graph module unavailable ({exc}); skipping M1 probe.")
        return 0
    g = load_graph()
    n_ntypes = len({n["type"] for n in g["nodes"]})
    n_etypes = len({e["relation"] for e in g["edges"]})
    print(
        f"[sanity] M1 graph · v{g['version']}: "
        f"{len(g['nodes'])} nodes / {len(g['edges'])} edges / "
        f"{n_ntypes} node types / {n_etypes} relations / "
        f"{len(g['modules'])} modules"
    )

    try:
        import torch

        from bovin_demo.graph import to_heterodata
        from bovin_demo.model import BaselineMLP, build_classifier
    except ImportError as exc:
        print(f"[sanity] torch / PyG unavailable ({exc}); skipping M3 forward probe.")
        return 0

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    data = to_heterodata(g, feat_dim=8, generator=gen)

    clf = build_classifier(data, hidden_dim=64, num_intra_layers=2, num_inter_layers=1)
    clf.eval()
    with torch.no_grad():
        out = clf(data)
    print(
        f"[sanity] M3 HeteroGNN forward · logit={out['logit'].item():+.4f} · "
        f"module_emb shape={tuple(out['module_emb'].shape)} · "
        f"n_modules_with_attn={sum(1 for a in out['attn'].values() if a.numel() > 0)}"
    )

    obs_dim = int(sum(data[nt].observable.sum().item() for nt in data.node_types))
    baseline = BaselineMLP(in_features=max(obs_dim, 1), hidden_dim=64)
    baseline.eval()
    with torch.no_grad():
        fake_x = torch.randn(1, max(obs_dim, 1), generator=gen)
        b_logit = baseline(fake_x)
    print(
        f"[sanity] M3 BaselineMLP forward · in_features={baseline.in_features} · "
        f"logit={b_logit.item():+.4f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command is None:
        _build_parser().print_help()
        return 0

    if args.command == "sanity":
        return _run_sanity(args.seed)

    if args.command == "train":
        return _run_train(args)

    if args.command == "xai":
        return _run_xai(args)

    if args.command == "eval":
        return _run_eval(args)

    # No remaining NYI commands — sub-parsers above cover the full PLAN §5 grid.
    print(
        f"[bovin-demo] subcommand '{args.command}' is declared but not yet implemented. "
        f"See BOVIN-Pathway-Demo-PLAN.md §5 for the milestone it lands in.",
        file=sys.stderr,
    )
    return 2


def _run_train(args) -> int:
    """Drive ``run_training`` across one or more seeds (T4.5)."""
    try:
        from bovin_demo.train import run_training
    except ImportError as exc:
        print(f"[train] heavy deps missing ({exc}); aborting.", file=sys.stderr)
        return 3

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    print(f"[train] running {len(seeds)} seed(s): {seeds}")
    results = []
    for s in seeds:
        print(f"[train] ▶ seed={s} config={args.config}")
        res = run_training(
            args.config,
            seed=s,
            max_epochs_override=args.max_epochs,
            output_root=args.output_root,
            raw_dir_override=args.raw_dir,
        )
        print(
            f"[train] ✓ seed={s} "
            f"best_val_auc={res.best_val_auc:.4f} "
            f"test_auc={res.test_auc:.4f} "
            f"baseline_test_auc={res.baseline_test_auc:.4f} "
            f"→ {res.run_dir}"
        )
        results.append(res)

    if len(results) > 1:
        import statistics as _stats

        test_aucs = [r.test_auc for r in results]
        baseline_aucs = [r.baseline_test_auc for r in results]
        print(
            f"[train] stability summary over {len(results)} seeds: "
            f"test_auc = {_stats.mean(test_aucs):.4f} ± {_stats.pstdev(test_aucs):.4f} · "
            f"baseline_test_auc = {_stats.mean(baseline_aucs):.4f} ± {_stats.pstdev(baseline_aucs):.4f}"
        )
    return 0


def _run_xai(args) -> int:
    """Drive ``run_xai`` for one M4 run directory."""
    try:
        from bovin_demo.xai import run_xai
    except ImportError as exc:
        print(f"[xai] heavy deps missing ({exc}); aborting.", file=sys.stderr)
        return 3

    sanity = run_xai(
        args.run_dir,
        config_path=args.config,
        raw_dir_override=args.raw_dir,
        seed=args.seed,
        top_n_patients=args.top_n,
        n_steps=args.n_steps,
    )
    print(f"[xai] top-3 modules      : {sanity['top3_modules']}")
    print(f"[xai] top-5 nodes        : {sanity['top5_nodes']}")
    print(f"[xai] DoD #4 (M4 top-3)  : {'PASS' if sanity['dod_4_m4_damp_in_top3_modules'] else 'FAIL'}")
    print(f"[xai] DoD #5 (landmark)  : {'PASS' if sanity['dod_5_landmark_in_top5_nodes'] else 'FAIL'}"
          f" · found={sanity['landmarks_found']}")
    print(f"[xai] heatmap → {sanity['heatmap']}")
    return 0


def _run_eval(args) -> int:
    """Drive ``build_report`` (with optional LUAD zero-shot embed)."""
    try:
        from bovin_demo.eval import build_report, run_luad_zero_shot
    except ImportError as exc:
        print(f"[eval] heavy deps missing ({exc}); aborting.", file=sys.stderr)
        return 3

    luad = None
    if args.luad_raw_dir:
        print(f"[eval] running LUAD zero-shot using {args.luad_raw_dir} …")
        luad = run_luad_zero_shot(
            args.run_dir,
            config_path=args.config,
            luad_raw_dir=args.luad_raw_dir,
        )
        print(f"[eval] LUAD auc={luad['auc']:.4f} acc={luad['accuracy']:.4f} "
              f"n={luad['n_samples']} hit_rate={luad['alignment_hit_rate']:.3f}")

    path = build_report(
        args.run_dir,
        config_path=args.config,
        raw_dir_override=args.raw_dir,
        bootstrap=args.bootstrap,
        recompute_test=(not args.no_recompute_test),
        luad_metrics=luad,
    )
    print(f"[eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
