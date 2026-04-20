"""M2 · T2.3 — HGNC symbol alignment between TCGA expression and graph nodes.

The graph carries per-node ``symbol`` (HGNC or equivalent) and a boolean
``observable`` flag. Only observable nodes are candidates for gene expression
input; the rest (viral / synthetic / aggregated cytokine placeholders like
``IFNA/IFNB``) receive learned constants instead.

Some ``symbol`` values are slash-separated aggregates (``IFNA/IFNB``,
``TRG/TRD``). We treat them as an "any-of" hit: if any component symbol is
present in ``expr``, the node is considered matched and its expression is
averaged across the present components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bovin_demo.graph.loader import GraphDict


@dataclass
class HitReport:
    """Summary of the alignment attempt (returned to callers for logging)."""

    hit_rate: float
    n_observable: int
    n_hits: int
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    aggregate_resolved: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "hit_rate": self.hit_rate,
            "n_observable": self.n_observable,
            "n_hits": self.n_hits,
            "hits": self.hits,
            "misses": self.misses,
            "aggregate_resolved": self.aggregate_resolved,
        }


def _candidate_symbols(symbol: str) -> list[str]:
    """Split slash/pipe-separated aggregate symbols into individual HGNC IDs."""
    cleaned = symbol.replace("|", "/").strip()
    parts = [p.strip() for p in cleaned.split("/") if p.strip()]
    return parts or [symbol]


def map_to_pathway_nodes(
    expr: pd.DataFrame,
    graph: GraphDict,
) -> tuple[pd.DataFrame, HitReport]:
    """Align ``expr`` columns (gene symbols) to observable nodes in ``graph``.

    Parameters
    ----------
    expr : pd.DataFrame
        samples × genes. Columns must be HGNC symbols.
    graph : GraphDict
        Output of ``load_graph()``. Observable = nodes that can receive
        per-patient expression input.

    Returns
    -------
    tuple[pd.DataFrame, HitReport]
        * ``aligned`` — samples × observable-nodes, one column per **node id**
          (not gene symbol), values drawn from ``expr`` (or NaN on miss).
          Aggregate symbols resolve to the mean of present components.
        * ``report`` — hit_rate + per-node hits/misses.

    Acceptance (Plan §5 · M2 DoD)
    -----------------------------
    ``report.hit_rate >= 0.70`` on TCGA-COAD.
    """
    observable = [n for n in graph["nodes"] if n["observable"]]
    aligned_cols: dict[str, pd.Series] = {}
    hits: list[str] = []
    misses: list[str] = []
    aggregate_resolved: dict[str, list[str]] = {}

    gene_set = set(expr.columns)

    for node in observable:
        candidates = _candidate_symbols(node["symbol"])
        present = [s for s in candidates if s in gene_set]

        if not present:
            misses.append(node["id"])
            # Keep the column so the HeteroData builder sees a consistent shape;
            # zero-impute and rely on downstream masking via `observable` flag.
            aligned_cols[node["id"]] = pd.Series(
                np.nan, index=expr.index, name=node["id"], dtype="float64"
            )
            continue

        if len(present) == 1:
            series = expr[present[0]].astype("float64")
        else:
            series = expr.loc[:, present].mean(axis=1)
        # Flag aggregate symbols (slash-separated) so callers can audit which
        # components actually matched, even if only one did.
        if len(candidates) > 1:
            aggregate_resolved[node["id"]] = present

        aligned_cols[node["id"]] = series.rename(node["id"]).astype("float64")
        hits.append(node["id"])

    aligned = pd.DataFrame(aligned_cols, index=expr.index)
    aligned.index.name = expr.index.name or "sample"

    n_obs = len(observable)
    hit_rate = len(hits) / n_obs if n_obs else 0.0
    report = HitReport(
        hit_rate=hit_rate,
        n_observable=n_obs,
        n_hits=len(hits),
        hits=hits,
        misses=misses,
        aggregate_resolved=aggregate_resolved,
    )
    return aligned, report
