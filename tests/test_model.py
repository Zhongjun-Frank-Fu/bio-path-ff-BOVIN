"""M3 tests — HeteroGNN forward shapes + grad flow + baseline MLP."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from bovin_demo.graph import load_graph, to_heterodata  # noqa: E402
from bovin_demo.model import (  # noqa: E402
    BaselineMLP,
    HeteroGNN,
    HeteroGNNClassifier,
    ModuleAttentionPool,
    build_classifier,
)


@pytest.fixture(scope="module")
def data():
    torch.manual_seed(0)
    g = load_graph()
    return to_heterodata(g, feat_dim=8, generator=torch.Generator().manual_seed(0))


# ------------------------------ T3.1 --------------------------------------
def test_hetero_gnn_forward_preserves_node_types(data):
    model = HeteroGNN.from_heterodata(data, hidden_dim=64)
    x_out = model(data)
    assert set(x_out.keys()) == set(data.node_types)
    for nt in data.node_types:
        assert x_out[nt].shape == (data[nt].num_nodes, 64)
        assert torch.isfinite(x_out[nt]).all()


def test_hetero_gnn_supports_varied_hidden_dim(data):
    """The hidden_dim should flow through encoder/intra/inter without drift."""
    model = HeteroGNN.from_heterodata(data, hidden_dim=32, heads=2)
    x_out = model(data)
    for h in x_out.values():
        assert h.size(-1) == 32


def test_hetero_gnn_encoder_is_per_node_type(data):
    model = HeteroGNN.from_heterodata(data, hidden_dim=16)
    # One encoder entry per node type — this is what gives each type its own
    # input projection, which M4/M5 needs for fair XAI attribution.
    assert set(model.encoder.keys()) == set(data.node_types)


# ------------------------------ T3.2 / T3.3 -------------------------------
def test_module_attention_pool_output_shape(data):
    model = HeteroGNN.from_heterodata(data, hidden_dim=24)
    x_out = model(data)
    pool = ModuleAttentionPool(hidden_dim=24)
    module_of_node = {nt: list(data[nt].module) for nt in data.node_types}
    emb, attn = pool(x_out, module_of_node)
    assert emb.shape == (11,)
    # Attention weights must softmax to 1 inside each populated module.
    for mid, alpha in attn.items():
        if alpha.numel() > 0:
            assert torch.isclose(alpha.sum(), torch.tensor(1.0), atol=1e-5), (mid, alpha.sum())


def test_full_classifier_forward_returns_logit(data):
    clf = build_classifier(data, hidden_dim=32)
    out = clf(data)
    assert out["logit"].shape == (1,)
    assert out["module_emb"].shape == (len(clf.pool.module_ids),)
    assert torch.isfinite(out["logit"]).all()


def test_full_classifier_grad_flows_back_to_encoder(data):
    """An AUC-loss gradient must actually update the per-node-type encoder;
    if the encoder is somehow disconnected from the logit (e.g. pool drops
    the hidden dim silently) training will look OK but never learn."""
    torch.manual_seed(7)
    clf = build_classifier(data, hidden_dim=16, dropout=0.0)
    clf.train()
    out = clf(data)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logit"], torch.tensor([1.0])
    )
    loss.backward()
    grads_nonzero = 0
    for p in clf.backbone.encoder.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            grads_nonzero += 1
    assert grads_nonzero > 0, "no gradient reached the per-node-type encoder"


def test_classifier_module_ids_match_graph_modules(data):
    clf = build_classifier(data)
    observed = set()
    for nt in data.node_types:
        observed.update(data[nt].module)
    assert set(clf.pool.module_ids) == observed


# ------------------------------ T3.4 --------------------------------------
def test_baseline_mlp_forward_shape():
    mlp = BaselineMLP(in_features=70, hidden_dim=64)
    x = torch.randn(4, 70)
    y = mlp(x)
    assert y.shape == (4, 1)
    assert torch.isfinite(y).all()


def test_baseline_mlp_handles_1d_input():
    mlp = BaselineMLP(in_features=70)
    y = mlp(torch.randn(70))
    assert y.shape == (1, 1)


def test_baseline_mlp_grad_flows():
    mlp = BaselineMLP(in_features=70, dropout=0.0)
    x = torch.randn(2, 70, requires_grad=True)
    y = mlp(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


# ------------------------------ T3.5 --------------------------------------
def test_sanity_cli_runs_through_forward(capsys):
    from bovin_demo.cli import main

    code = main(["sanity"])
    out = capsys.readouterr().out
    assert code == 0
    assert "M1 graph" in out
    assert "M3 HeteroGNN forward" in out
    assert "M3 BaselineMLP forward" in out
