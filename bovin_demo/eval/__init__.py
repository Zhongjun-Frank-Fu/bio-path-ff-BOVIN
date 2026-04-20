"""Evaluation metrics + report generation + LUAD zero-shot (M6)."""

from __future__ import annotations

from bovin_demo.eval.luad_transfer import run_luad_zero_shot
from bovin_demo.eval.metrics import MetricCI, compute_metrics
from bovin_demo.eval.report import build_report

__all__ = [
    "MetricCI",
    "build_report",
    "compute_metrics",
    "run_luad_zero_shot",
]
