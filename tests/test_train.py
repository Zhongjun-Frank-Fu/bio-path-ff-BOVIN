"""M4 smoke tests — end-to-end 3-epoch run against the synthetic Xena fixture.

We can't test a real val-AUC target (≥ 0.65 per PLAN §5) inside CI because
the dataset is synthetic; PLAN §5 DoD #2 is verified by hand on real
TCGA-COAD in Docker. These tests instead guard the *plumbing*:

* run_training returns a TrainingResult with finite numbers
* metrics.json lands under a seeded, timestamped run dir
* training.log contains key milestones
* CSV logger writes metrics_per_epoch
* a 2-seed sweep via the CLI produces two distinct run dirs
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("torchmetrics")


@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory) -> Path:
    """Write a minimal YAML config the training loop can read."""
    out = tmp_path_factory.mktemp("cfg") / "tcga_tiny.yaml"
    out.write_text(
        """
paths:
  raw_dir: data/raw
  output_root: outputs
graph:
  add_self_loops: true
model:
  hidden_dim: 16
  num_intra_layers: 1
  num_inter_layers: 1
  attention_heads: 2
  dropout: 0.1
train:
  optimizer: adamw
  lr: 5.0e-3
  weight_decay: 1.0e-4
  batch_size: 16
  max_epochs: 3
  patience: 2
  scheduler: cosine
  loss: bce_with_logits
  class_weight: balanced
""",
        encoding="utf-8",
    )
    return out


# ------------------------------ T4.2 / T4.3 -------------------------------
def test_run_training_produces_metrics_json(xena_like_raw_dir, tmp_path, tiny_config):
    from bovin_demo.train import run_training

    result = run_training(
        tiny_config,
        seed=42,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )

    # Numbers are finite (no NaN from a dead optimizer).
    assert 0.0 <= result.test_auc <= 1.0
    assert 0.0 <= result.best_val_auc <= 1.0
    assert 0.0 <= result.baseline_test_auc <= 1.0

    # Artifacts
    assert result.run_dir.exists()
    assert (result.run_dir / "metrics.json").exists()
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert metrics["seed"] == 42
    assert metrics["num_train"] > 0
    assert metrics["num_val"] > 0
    assert metrics["num_test"] > 0


def test_run_training_writes_training_log(xena_like_raw_dir, tmp_path, tiny_config):
    from bovin_demo.train import run_training

    result = run_training(
        tiny_config,
        seed=7,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )
    log_text = (result.run_dir / "training.log").read_text()
    for milestone in ("graph:", "TCGA:", "alignment hit_rate", "split sizes", "best val_auc"):
        assert milestone in log_text, f"missing milestone: {milestone}"


def test_run_training_writes_csv_log(xena_like_raw_dir, tmp_path, tiny_config):
    from bovin_demo.train import run_training

    result = run_training(
        tiny_config,
        seed=1,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )
    csvs = list((result.run_dir / "logs").rglob("metrics.csv"))
    assert csvs, f"no metrics.csv under {result.run_dir}/logs"
    # At least the header plus a few rows.
    lines = csvs[0].read_text().splitlines()
    assert len(lines) >= 2
    assert "train_loss" in lines[0] or "train_auc" in lines[0]


def test_run_training_saves_best_checkpoint(xena_like_raw_dir, tmp_path, tiny_config):
    from bovin_demo.train import run_training

    result = run_training(
        tiny_config,
        seed=0,
        output_root=tmp_path,
        raw_dir_override=xena_like_raw_dir,
    )
    ckpts = list((result.run_dir / "ckpt").glob("*.ckpt"))
    assert len(ckpts) == 1, ckpts


# ------------------------------ T4.5 --------------------------------------
def test_cli_train_multi_seed_sweep_makes_distinct_run_dirs(
    xena_like_raw_dir, tmp_path, tiny_config, capsys
):
    from bovin_demo.cli import main

    code = main(
        [
            "train",
            "--config",
            str(tiny_config),
            "--max-epochs",
            "2",
            "--output-root",
            str(tmp_path),
            "--raw-dir",
            str(xena_like_raw_dir),
            "--seeds",
            "42,1337",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "seed=42" in out
    assert "seed=1337" in out
    assert "stability summary" in out

    # Only count timestamped run dirs — the xena_like_raw_dir fixture also
    # writes under tmp_path ("data/raw") so we filter by name pattern.
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and "_seed" in p.name]
    assert len(run_dirs) == 2
