"""Parse ``BOVIN-Pathway-Graph-v0.md`` → ``bovin_pathway_v0.json``.

M1 · T1.1. Pure stdlib, no runtime deps — we want this to be driveable from
both the host (no Docker) and from inside the bovin_demo image. The JSON it
emits is what ``bovin_demo.graph.loader.load_graph`` consumes at runtime.

Usage
-----
    python3 tools/parse_graph_v0.py \
        --src ../BOVIN-Pathway-Graph-v0.md \
        --dst bovin_demo/graph/bovin_pathway_v0.json

Guarantees on the output (asserted at the bottom)
-------------------------------------------------
* ``len(nodes) == 82``
* ``len(edges) >= 99`` after dropping edges to undefined nodes and de-duping
* Every edge's ``source`` / ``target`` is present in the node set
* Every node carries a ``module`` in ``{M1..M11}`` and a non-empty ``type``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Module ordering / names come from v0.md §1. Keeping them here as the single
# source of truth so the parser can reject typos in the source markdown.
MODULES: list[tuple[str, str]] = [
    ("M1", "ENTRY"),
    ("M2", "ISR"),
    ("M3", "MITO"),
    ("M4", "DAMP"),
    ("M5", "METAB"),
    ("M6", "APC_RECV"),
    ("M7", "DC_MAT"),
    ("M8", "TCELL"),
    ("M9", "ICB"),
    ("M10", "MEM"),
    ("M11", "MAC"),
]
VALID_MODULE_IDS = {m for m, _ in MODULES}

NODE_HEADER = ("id", "symbol", "name", "type", "side", "observable", "refs")
EDGE_HEADER = ("source", "relation", "target", "dir", "modules", "evidence")


def _cells(line: str) -> list[str]:
    """Split a markdown table row into stripped cell values.

    Strips the leading/trailing pipes and any surrounding whitespace. Does not
    try to understand inline pipes (v0.md has none inside cells).
    """
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts


def _is_separator(line: str) -> bool:
    return bool(re.match(r"^\|?\s*:?-{3,}", line.strip()))


def _match_module_heading(line: str) -> str | None:
    """Return the module id (M1..M11) if ``line`` is a §2 module heading."""
    m = re.match(r"^##\s+(M\d+)\s+·", line)
    if m and m.group(1) in VALID_MODULE_IDS:
        return m.group(1)
    return None


def _parse_observable(raw: str) -> bool:
    v = raw.strip().upper()
    if v in {"Y", "YES", "TRUE", "1"}:
        return True
    if v in {"N", "NO", "FALSE", "0"}:
        return False
    raise ValueError(f"cannot parse observable flag: {raw!r}")


def _parse_side(raw: str) -> str:
    v = raw.strip().lower()
    if v not in {"host", "therapy"}:
        raise ValueError(f"unknown side {raw!r} (expected host|therapy)")
    return v


def _parse_direction(raw: str) -> str:
    """Map the table's dir glyph ('+', '-', '±') to a normalized string."""
    v = raw.strip()
    if v in {"+", "positive", "pos"}:
        return "+"
    if v in {"-", "–", "negative", "neg"}:
        return "-"
    if v in {"±", "+/-", "both"}:
        return "±"
    raise ValueError(f"unknown direction glyph {raw!r}")


def _parse_modules_field(raw: str) -> tuple[str | None, str | None]:
    """'M1→M2' → ('M1', 'M2'); 'therapy→M2' → ('therapy', 'M2')."""
    for arrow in ("→", "->"):
        if arrow in raw:
            a, b = raw.split(arrow, 1)
            return a.strip() or None, b.strip() or None
    return None, None


def parse_graph(md_path: Path) -> dict:
    """Walk ``md_path`` line by line and pull out nodes + edges.

    The parser is deliberately simple: it classifies each table by its header
    row (matches NODE_HEADER or EDGE_HEADER), then reads data rows until the
    table ends. Nodes inherit the last seen §2 module heading.
    """
    nodes: list[dict] = []
    edges_raw: list[dict] = []
    current_module: str | None = None

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        mod = _match_module_heading(line)
        if mod is not None:
            current_module = mod
            i += 1
            continue

        # Table detection: a row that looks like | a | b | c | followed by a
        # separator row like |---|---|---|.
        if line.lstrip().startswith("|") and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            header = tuple(c.lower() for c in _cells(line))
            i += 2  # skip header + separator

            if header == NODE_HEADER:
                if current_module is None:
                    raise ValueError(
                        f"node table at line {i} has no preceding ## M<N> heading"
                    )
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    cells = _cells(lines[i])
                    if len(cells) == len(NODE_HEADER):
                        nodes.append(_node_row(cells, current_module))
                    i += 1
                continue

            if header == EDGE_HEADER:
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    cells = _cells(lines[i])
                    if len(cells) == len(EDGE_HEADER):
                        edges_raw.append(_edge_row(cells))
                    i += 1
                continue

            # Unknown table — skip its body.
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            continue

        i += 1

    return _finalize(nodes, edges_raw)


def _node_row(cells: list[str], module_id: str) -> dict:
    nid, symbol, name, ntype, side, observable, refs = cells
    return {
        "id": nid.strip(),
        "symbol": symbol.strip(),
        "name": name.strip(),
        # v0.md uses "kinase*" as a placeholder — strip the annotation; type
        # normalization to 'enzyme' etc. lands in v1 (see v0.md §2 footnote).
        "type": ntype.strip().rstrip("*") or "unknown",
        "side": _parse_side(side),
        "observable": _parse_observable(observable),
        "module": module_id,
        "refs": refs.strip(),
    }


def _edge_row(cells: list[str]) -> dict:
    source, relation, target, direction, modules, evidence = cells
    mod_from, mod_to = _parse_modules_field(modules)
    return {
        "source": source.strip(),
        "relation": relation.strip(),
        "target": target.strip(),
        "direction": _parse_direction(direction),
        "module_from": mod_from,
        "module_to": mod_to,
        "evidence": evidence.strip(),
    }


def _finalize(nodes: list[dict], edges_raw: list[dict]) -> dict:
    """Validate + clean the parsed tables.

    * drop edges whose source/target is not in the node set (v0.md §5 flags
      ``fpr1`` and ``siglec10`` as v1-only placeholders)
    * de-duplicate edges on the (source, relation, target) triple — v0.md
      §3.12 intentionally restates a few therapy-context edges that are
      already in §3.2 / §3.5
    """
    node_ids = {n["id"] for n in nodes}
    if len(node_ids) != len(nodes):
        dups = [n["id"] for n in nodes if [x["id"] for x in nodes].count(n["id"]) > 1]
        raise ValueError(f"duplicate node ids in v0.md: {sorted(set(dups))}")

    dropped: list[dict] = []
    deduped: dict[tuple[str, str, str], dict] = {}
    for e in edges_raw:
        if e["source"] not in node_ids or e["target"] not in node_ids:
            dropped.append(e)
            continue
        key = (e["source"], e["relation"], e["target"])
        deduped.setdefault(key, e)

    edges = list(deduped.values())

    return {
        "version": "v0.1",
        "modules": [{"id": mid, "name": name} for mid, name in MODULES],
        "nodes": nodes,
        "edges": edges,
        "_parse_report": {
            "nodes_parsed": len(nodes),
            "edges_parsed_raw": len(edges_raw),
            "edges_after_clean": len(edges),
            "edges_dropped_orphan": [
                {"source": e["source"], "target": e["target"], "relation": e["relation"]}
                for e in dropped
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--src",
        type=Path,
        default=repo_root.parent / "BOVIN-Pathway-Graph-v0.md",
        help="Source markdown (default: ../BOVIN-Pathway-Graph-v0.md)",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=repo_root / "bovin_demo" / "graph" / "bovin_pathway_v0.json",
        help="Destination JSON (default: bovin_demo/graph/bovin_pathway_v0.json)",
    )
    args = p.parse_args(argv)

    graph = parse_graph(args.src)

    report = graph.pop("_parse_report")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[parse_graph_v0] wrote {args.dst}")
    print(f"[parse_graph_v0] nodes          = {report['nodes_parsed']}")
    print(f"[parse_graph_v0] edges raw      = {report['edges_parsed_raw']}")
    print(f"[parse_graph_v0] edges kept     = {report['edges_after_clean']}")
    if report["edges_dropped_orphan"]:
        print("[parse_graph_v0] dropped orphan edges (undefined endpoint):")
        for e in report["edges_dropped_orphan"]:
            print(f"    - {e['source']} --{e['relation']}--> {e['target']}")

    # Hard guards — plan §5 DoD for M1.
    assert report["nodes_parsed"] == 82, (
        f"expected 82 nodes (plan DoD), got {report['nodes_parsed']}"
    )
    assert report["edges_after_clean"] >= 99, (
        f"expected >= 99 edges after cleaning, got {report['edges_after_clean']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
