"""M5 · T5.3 — matplotlib heatmap, colors aligned with the Dossier palette.

Layout
------
Rows = top-N patients (selected in :mod:`bovin_demo.xai.aggregate`).
Cols = 11 modules in ``M1..M11`` order.
Cell color = per-patient, per-module contribution (``|attr|`` sum over nodes).

A second axis above the heatmap carries the module palette (the same one used
in ``tools/render_graph_overview.py``) so viewers can scan the colored bar
and immediately know "which module is DAMP".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Dossier palette — identical to tools/render_graph_overview.py.
MODULE_COLORS = {
    "M1":  "#5B8DEF",
    "M2":  "#7B5BEF",
    "M3":  "#EF5B5B",
    "M4":  "#F2A33A",
    "M5":  "#4FB286",
    "M6":  "#9B59B6",
    "M7":  "#E67E22",
    "M8":  "#2ECC71",
    "M9":  "#34495E",
    "M10": "#16A085",
    "M11": "#C0392B",
}
MODULE_NAMES = {
    "M1": "ENTRY", "M2": "ISR", "M3": "MITO", "M4": "DAMP", "M5": "METAB",
    "M6": "APC_RECV", "M7": "DC_MAT", "M8": "TCELL", "M9": "ICB",
    "M10": "MEM", "M11": "MAC",
}


def plot_heatmap(
    matrix: np.ndarray,
    *,
    patient_ids: list[str],
    module_ids: list[str],
    out_path: str | Path,
    title: str = "BOVIN-Pathway · module × patient attribution",
) -> Path:
    """Render a ``(P, M)`` heatmap to ``out_path`` and return the path."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = matrix.shape
    if len(patient_ids) != n_rows:
        raise ValueError(f"patient_ids length {len(patient_ids)} != rows {n_rows}")
    if len(module_ids) != n_cols:
        raise ValueError(f"module_ids length {len(module_ids)} != cols {n_cols}")

    fig, (ax_bar, ax) = plt.subplots(
        nrows=2, figsize=(1.1 * n_cols + 2, 0.28 * n_rows + 2),
        gridspec_kw={"height_ratios": [1, max(n_rows, 4)], "hspace": 0.05},
    )

    ax_bar.set_xlim(-0.5, n_cols - 0.5)
    ax_bar.set_ylim(0, 1)
    for j, mid in enumerate(module_ids):
        ax_bar.add_patch(Rectangle((j - 0.5, 0), 1, 1, color=MODULE_COLORS.get(mid, "#cccccc")))
        label = f"{mid}\n{MODULE_NAMES.get(mid, '')}"
        ax_bar.text(j, 0.5, label, ha="center", va="center",
                    fontsize=7, color="white", weight="bold")
    ax_bar.set_axis_off()

    # ``rocket_r`` is a seaborn cmap; base matplotlib has ``magma_r`` which
    # gives a nearly identical perceptual gradient without the extra import.
    im = ax.imshow(matrix, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(module_ids, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(patient_ids, fontsize=7)
    ax.set_ylabel("Patient (top-TPR)")
    ax.set_xlabel("Pathway module")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Σ |IG attribution|")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
