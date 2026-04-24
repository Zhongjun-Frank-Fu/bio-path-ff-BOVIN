"""A2-M2 · T2.2 prep — one-off build of BOVIN gene alias table.

Queries mygene.info once to map each BOVIN observable-node HGNC symbol to its
Entrez gene ID and Ensembl gene ID. Output is committed to
``bovin_demo/data/static/bovin_gene_aliases.csv`` so the actual
``ici_loader.py`` has zero online dependencies at runtime.

Aggregate symbols like ``IFNA/IFNB`` or ``TRG/TRD`` expand into one row per
component symbol; the loader treats them as "any-of" (mean across present
components), matching ``gene_mapping.py::_candidate_symbols``.

Usage
-----
    python tools/build_bovin_gene_aliases.py        # produce CSV
    python tools/build_bovin_gene_aliases.py --verify  # re-run and diff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import mygene  # type: ignore[import-untyped]


ROOT = Path(__file__).resolve().parents[1]
GRAPH_JSON = ROOT / "bovin_demo" / "graph" / "bovin_pathway_v0.json"
OUT_CSV = ROOT / "bovin_demo" / "data" / "static" / "bovin_gene_aliases.csv"


def expand_aggregate(symbol: str) -> list[str]:
    """`IFNA/IFNB` → [`IFNA`, `IFNB`] ; plain symbol → [symbol]."""
    cleaned = symbol.replace("|", "/").strip()
    parts = [p.strip() for p in cleaned.split("/") if p.strip()]
    return parts or [symbol]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="rebuild and diff against existing CSV")
    args = p.parse_args()

    graph = json.loads(GRAPH_JSON.read_text())
    observable = [n for n in graph["nodes"] if n["observable"]]
    print(f"[read] {GRAPH_JSON.name} — {len(observable)} observable nodes")

    rows: list[dict] = []
    for node in observable:
        for symbol in expand_aggregate(node["symbol"]):
            rows.append({
                "node_id":     node["id"],
                "node_symbol": node["symbol"],      # keep the raw aggregate form
                "hgnc_symbol": symbol,              # single HGNC symbol
            })
    query_frame = pd.DataFrame(rows)
    unique_syms = sorted(set(query_frame["hgnc_symbol"]))
    print(f"[query] {len(unique_syms)} unique HGNC symbols "
          f"(after expanding aggregates) → mygene.info")

    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        unique_syms,
        scopes="symbol",
        fields="entrezgene,ensembl.gene",
        species="human",
        returnall=False,
        verbose=False,
    )

    alias_map: dict[str, dict] = {}
    for h in hits:
        sym = h.get("query")
        if h.get("notfound"):
            alias_map[sym] = {"entrez_id": None, "ensembl_id": None}
            continue
        ens = h.get("ensembl")
        if isinstance(ens, list):
            ens_ids = [e.get("gene") for e in ens if e.get("gene")]
            ensembl_id = ens_ids[0] if ens_ids else None
        elif isinstance(ens, dict):
            ensembl_id = ens.get("gene")
        else:
            ensembl_id = None
        alias_map[sym] = {
            "entrez_id":  h.get("entrezgene"),
            "ensembl_id": ensembl_id,
        }

    query_frame["entrez_id"]  = query_frame["hgnc_symbol"].map(lambda s: alias_map.get(s, {}).get("entrez_id"))
    query_frame["ensembl_id"] = query_frame["hgnc_symbol"].map(lambda s: alias_map.get(s, {}).get("ensembl_id"))

    missing = query_frame[query_frame["entrez_id"].isna() & query_frame["ensembl_id"].isna()]
    if len(missing):
        print(f"[warn] {len(missing)} symbols had no Entrez and no Ensembl hit:")
        for _, row in missing.iterrows():
            print(f"         {row.node_id:16s}  {row.hgnc_symbol}")

    if args.verify and OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        merged = old.merge(query_frame, on=["node_id", "hgnc_symbol"],
                           suffixes=("_old", "_new"), how="outer", indicator=True)
        diffs = merged[(merged["entrez_id_old"] != merged["entrez_id_new"])
                       | (merged["ensembl_id_old"] != merged["ensembl_id_new"])]
        if len(diffs):
            print(f"[verify] {len(diffs)} rows differ from existing CSV — review before overwriting.")
            print(diffs.to_string(index=False))
            return 1
        print("[verify] OK — re-query matches existing CSV exactly.")
        return 0

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    query_frame = query_frame.astype({"entrez_id": "Int64"})
    query_frame.to_csv(OUT_CSV, index=False)
    print(f"[write] {OUT_CSV.relative_to(ROOT)}  ·  {len(query_frame)} rows")
    print(f"         entrez hit rate:  {query_frame['entrez_id'].notna().mean()*100:.1f}%")
    print(f"         ensembl hit rate: {query_frame['ensembl_id'].notna().mean()*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
