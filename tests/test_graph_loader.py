"""M1 · T1.3 tests — loader round-trip + Plan §5 DoD for nodes/edges/types."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bovin_demo.graph import load_graph
from bovin_demo.graph.schema import Graph


@pytest.fixture(scope="module")
def graph() -> dict:
    return load_graph()


def test_packaged_json_exists():
    from bovin_demo.graph.loader import _default_json_path

    path = _default_json_path()
    assert path.exists(), f"packaged graph JSON missing at {path}"
    assert path.suffix == ".json"


def test_load_graph_default_returns_plan_dod(graph):
    # Plan §5 M1 DoD: 82 nodes, >= 99 edges, >= 5 node types, >= 5 edge types.
    assert len(graph["nodes"]) == 82
    assert len(graph["edges"]) >= 99
    assert len({n["type"] for n in graph["nodes"]}) >= 5
    assert len({e["relation"] for e in graph["edges"]}) >= 5


def test_all_11_modules_are_populated(graph):
    expected = {f"M{i}" for i in range(1, 12)}
    actual = {n["module"] for n in graph["nodes"]}
    assert actual == expected


def test_every_edge_endpoint_is_a_known_node(graph):
    known = {n["id"] for n in graph["nodes"]}
    for e in graph["edges"]:
        assert e["source"] in known, f"dangling source: {e}"
        assert e["target"] in known, f"dangling target: {e}"


def test_known_landmark_nodes_present(graph):
    """CRT / HMGB1 / CD8A / PD1 are the nodes the XAI sanity check (T5.4)
    keys off — if any of them disappears from v0.md, M5 will break silently."""
    ids = {n["id"] for n in graph["nodes"]}
    for landmark in {"crt", "hmgb1", "cd8a", "pd1", "pdl1"}:
        assert landmark in ids, f"landmark node missing: {landmark}"


def test_observable_nodes_cover_tcga_input_surface(graph):
    """Plan §5 M2 DoD needs observable coverage; at M1 we just sanity-check
    that at least ~60% of nodes are observable so the GNN has an input."""
    n_obs = sum(1 for n in graph["nodes"] if n["observable"])
    assert n_obs / len(graph["nodes"]) >= 0.6


def test_load_from_markdown_matches_json(graph, tmp_path):
    """Pointing ``load_graph`` at the source .md must yield an identical
    nodes/edges set to the packaged JSON (parser is deterministic)."""
    md = Path(__file__).resolve().parents[2] / "BOVIN-Pathway-Graph-v0.md"
    if not md.exists():
        pytest.skip("source markdown not available on this checkout")
    from_md = load_graph(md)
    assert {n["id"] for n in from_md["nodes"]} == {n["id"] for n in graph["nodes"]}
    assert len(from_md["edges"]) == len(graph["edges"])


def test_schema_rejects_duplicate_node_ids():
    bad = {
        "version": "v0.1",
        "modules": [{"id": "M1", "name": "ENTRY"}],
        "nodes": [
            {"id": "x", "symbol": "X", "name": "x", "type": "TF",
             "side": "host", "observable": True, "module": "M1", "refs": ""},
            {"id": "x", "symbol": "X", "name": "x2", "type": "TF",
             "side": "host", "observable": True, "module": "M1", "refs": ""},
        ],
        "edges": [],
    }
    with pytest.raises(Exception, match="duplicate"):
        Graph.model_validate(bad)


def test_schema_rejects_dangling_edge():
    bad = {
        "version": "v0.1",
        "modules": [{"id": "M1", "name": "ENTRY"}],
        "nodes": [
            {"id": "x", "symbol": "X", "name": "x", "type": "TF",
             "side": "host", "observable": True, "module": "M1", "refs": ""},
        ],
        "edges": [
            {"source": "x", "relation": "activates", "target": "GHOST",
             "direction": "+", "module_from": "M1", "module_to": "M1", "evidence": ""},
        ],
    }
    with pytest.raises(Exception, match="undefined"):
        Graph.model_validate(bad)


def test_load_graph_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graph(tmp_path / "nope.json")


def test_graph_json_is_valid_utf8_json():
    from bovin_demo.graph.loader import _default_json_path

    raw = _default_json_path().read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert set(parsed.keys()) >= {"version", "modules", "nodes", "edges"}
