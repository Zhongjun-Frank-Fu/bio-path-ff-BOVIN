"""M1 · T1.6 — render a one-PNG sanity view of the BOVIN-Pathway graph.

This is not a publication figure; it's the cheapest possible way to eyeball
"do the 11 modules look right, do the hub nodes (wsn / crt / cd8a) actually
get lots of edges". Colors match the module palette used in the Dossier
HTML so screenshots can sit next to each other in slides.

Usage
-----
    python3 tools/render_graph_overview.py --out outputs/m1_sanity/graph.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bovin_demo.graph import load_graph

# Dossier palette (keep in sync with docs/demo_card.md when it lands).
MODULE_COLORS = {
    "M1":  "#5B8DEF",  # ENTRY       blue
    "M2":  "#7B5BEF",  # ISR         violet
    "M3":  "#EF5B5B",  # MITO        red
    "M4":  "#F2A33A",  # DAMP        amber
    "M5":  "#4FB286",  # METAB       teal
    "M6":  "#9B59B6",  # APC_RECV    purple
    "M7":  "#E67E22",  # DC_MAT      orange
    "M8":  "#2ECC71",  # TCELL       green
    "M9":  "#34495E",  # ICB         slate
    "M10": "#16A085",  # MEM         emerald
    "M11": "#C0392B",  # MAC         crimson
}


def render(out: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import networkx as nx  # noqa: PLC0415

    graph = load_graph()
    G = nx.DiGraph()
    for n in graph["nodes"]:
        G.add_node(n["id"], **n)
    for e in graph["edges"]:
        G.add_edge(e["source"], e["target"], relation=e["relation"])

    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=80)

    fig, ax = plt.subplots(figsize=(14, 10))
    for mid, color in MODULE_COLORS.items():
        ns = [n for n, d in G.nodes(data=True) if d["module"] == mid]
        nx.draw_networkx_nodes(
            G, pos, nodelist=ns, node_color=color, node_size=320,
            edgecolors="white", linewidths=1.0, label=mid, ax=ax,
        )
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowsize=8,
        edge_color="#999", width=0.6, alpha=0.55, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    ax.set_title(
        f"BOVIN-Pathway v0 · {G.number_of_nodes()} nodes / "
        f"{G.number_of_edges()} edges / 11 modules"
    )
    ax.axis("off")
    ax.legend(loc="lower left", ncol=4, fontsize=8, frameon=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_graph_overview] wrote {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/m1_sanity/graph_overview.png"))
    args = p.parse_args(argv)
    render(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
