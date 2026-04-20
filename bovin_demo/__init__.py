"""BOVIN-Pathway Demo · Aim 1 minimum closed loop.

This package provides a biology-structured HeteroGNN that consumes TCGA bulk RNA-seq
aligned to the BOVIN-Pathway graph (82 nodes / 99 edges across 11 modules) and produces:

  * an ICD-readiness logit per patient, and
  * module-level Integrated-Gradients attributions for XAI.

This is a **demo**. It exists to prove the end-to-end architecture walks,
not to claim SOTA. See docs/demo_card.md and BOVIN-Pathway-Demo-PLAN.md
for scope, surrogate-label caveats, and Definition of Done.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Nabe <z4fu@ucsd.edu>"
__all__ = ["__author__", "__version__"]
