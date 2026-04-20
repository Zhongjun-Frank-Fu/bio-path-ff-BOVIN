"""M1 · T1.3 — load the BOVIN-Pathway graph as a validated Python dict.

``load_graph`` is the one function the rest of the stack calls. It hides
whether the backing file is JSON (normal path) or the v0 markdown (dev
convenience): if you point it at the markdown we re-parse on the fly
using ``tools.parse_graph_v0`` so no stale copy can drift out of sync.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import TypedDict

from bovin_demo.graph.schema import Graph

DEFAULT_GRAPH_RESOURCE = "bovin_pathway_v0.json"


class GraphNode(TypedDict, total=False):
    id: str
    symbol: str
    name: str
    type: str
    side: str
    observable: bool
    module: str
    refs: str


class GraphEdge(TypedDict, total=False):
    source: str
    relation: str
    target: str
    direction: str
    module_from: str | None
    module_to: str | None
    evidence: str


class GraphDict(TypedDict):
    version: str
    modules: list[dict]
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_markdown(path: Path) -> dict:
    """Re-parse the v0 markdown via ``tools/parse_graph_v0.py``.

    Loaded via ``importlib`` rather than ``import tools.parse_graph_v0``
    so the package remains importable in environments where the ``tools``
    directory is not on ``sys.path`` (e.g. installed wheels).
    """
    import importlib.util

    repo_root = path.resolve().parent
    parser_path: Path | None = None
    for parent in [repo_root, *repo_root.parents]:
        candidate = parent / "tools" / "parse_graph_v0.py"
        if candidate.exists():
            parser_path = candidate
            break
    if parser_path is None:
        raise FileNotFoundError(
            "tools/parse_graph_v0.py not found — cannot re-parse markdown; "
            "either point load_graph at the packaged JSON or run the tool first"
        )

    spec = importlib.util.spec_from_file_location("_parse_graph_v0", parser_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = module.parse_graph(path)
    parsed.pop("_parse_report", None)
    return parsed


def _default_json_path() -> Path:
    """Return the packaged ``bovin_pathway_v0.json`` as a concrete Path."""
    with resources.as_file(
        resources.files("bovin_demo.graph").joinpath(DEFAULT_GRAPH_RESOURCE)
    ) as p:
        return Path(p)


def load_graph(path: str | Path | None = None) -> GraphDict:
    """Load + validate the BOVIN-Pathway graph.

    Parameters
    ----------
    path : str | Path | None
        * ``None`` (default) — load the JSON bundled in ``bovin_demo.graph``.
        * path to a ``.json`` — parsed and validated.
        * path to a ``.md`` — re-run the v0 markdown parser, then validate.

    Returns
    -------
    GraphDict
        ``{"version", "modules", "nodes", "edges"}`` — plain python dicts
        so downstream code (HeteroData builder, tests) doesn't need pydantic.

    Raises
    ------
    pydantic.ValidationError
        if the JSON violates the schema in ``bovin_demo.graph.schema``.
    AssertionError
        if the Plan §5 DoD (82 nodes / ≥99 edges / ≥5 types) fails.
    """
    src = _default_json_path() if path is None else Path(path)
    if not src.exists():
        raise FileNotFoundError(f"graph file not found: {src}")

    raw = _load_markdown(src) if src.suffix.lower() == ".md" else _load_json(src)
    validated = Graph.model_validate(raw)
    validated.dod_check()
    return validated.model_dump()  # type: ignore[return-value]
