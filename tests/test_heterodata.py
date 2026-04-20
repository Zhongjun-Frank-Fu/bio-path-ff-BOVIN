"""M1 · T1.4 / T1.5 tests — HeteroData shape + GATv2Conv forward smoke."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

from bovin_demo.graph import load_graph, to_heterodata  # noqa: E402


@pytest.fixture(scope="module")
def data() -> HeteroData:
    graph = load_graph()
    g = torch.Generator().manual_seed(42)
    return to_heterodata(graph, feat_dim=8, generator=g)


def test_heterodata_node_total_is_82(data):
    total = sum(data[nt].num_nodes for nt in data.node_types)
    assert total == 82


def test_heterodata_edge_total_is_at_least_99(data):
    total = sum(data[et].edge_index.size(1) for et in data.edge_types)
    assert total >= 99


def test_heterodata_has_at_least_5_node_and_edge_types(data):
    # Plan §5 DoD: "节点/边类型各 ≥ 5"
    assert len(data.node_types) >= 5
    assert len(data.edge_types) >= 5


def test_heterodata_node_features_shape(data):
    for nt in data.node_types:
        x = data[nt].x
        assert x.dim() == 2
        assert x.size(0) == data[nt].num_nodes
        assert x.size(1) == 8


def test_heterodata_edge_index_is_sane(data):
    for src, rel, dst in data.edge_types:
        ei = data[src, rel, dst].edge_index
        assert ei.dtype == torch.long
        assert ei.size(0) == 2
        assert ei[0].max().item() < data[src].num_nodes
        assert ei[1].max().item() < data[dst].num_nodes


def test_heterodata_metadata_matches_pyg_contract(data):
    # ``metadata()`` is what HGTConv consumes internally; if this breaks the
    # M3 HeteroGNN model is going to fail far from the root cause.
    ntypes, etypes = data.metadata()
    assert set(ntypes) == set(data.node_types)
    assert set(etypes) == set(data.edge_types)


def test_heterodata_self_loops_flag(data):
    graph = load_graph()
    g = torch.Generator().manual_seed(0)
    with_sl = to_heterodata(graph, feat_dim=4, add_self_loops=True, generator=g)
    self_loop_types = [et for et in with_sl.edge_types if et[1] == "self_loop"]
    assert len(self_loop_types) == len(with_sl.node_types)


def test_gatv2conv_forward_on_random_nodes():
    """T1.5: pick the largest node type, run one GATv2 layer, assert
    output shape. Proves the 82-node graph's features are shaped in a way
    PyG's attention kernels will actually accept."""
    from torch_geometric.nn import GATv2Conv

    graph = load_graph()
    g = torch.Generator().manual_seed(1)
    data = to_heterodata(graph, feat_dim=16, generator=g)

    # Pick the (src,rel,dst) edge type with the most edges + src==dst so we can
    # run a homogeneous GATv2 on that slice without detouring through HGTConv.
    same_type = [
        et for et in data.edge_types
        if et[0] == et[2] and data[et].edge_index.size(1) >= 1
    ]
    assert same_type, "expected at least one intra-node-type edge set"
    src_type, rel, _ = max(same_type, key=lambda et: data[et].edge_index.size(1))

    x = data[src_type].x
    edge_index = data[src_type, rel, src_type].edge_index

    conv = GATv2Conv(in_channels=x.size(1), out_channels=32, heads=2, concat=True)
    out = conv(x, edge_index)
    assert out.shape == (x.size(0), 64)
    assert torch.isfinite(out).all()
