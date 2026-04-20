"""PyTorch Lightning training loop (M4)."""

from __future__ import annotations

from bovin_demo.train.lit_module import build_lit_module
from bovin_demo.train.loop import TrainingResult, run_training

__all__ = ["TrainingResult", "build_lit_module", "run_training"]
