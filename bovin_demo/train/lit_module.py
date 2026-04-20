"""M4 · T4.1 — LightningModule wrapping ``HeteroGNNClassifier``.

Loss
----
``BCEWithLogitsLoss`` with a ``pos_weight`` computed from the train fold.
This is what PLAN §4.3 ``class_weight: balanced`` resolves to: a single
scalar weight that up-weights the minority class so the model doesn't
collapse to predicting the majority.

Metric
------
``torchmetrics.classification.BinaryAUROC`` accumulates on each step, resets
at epoch end. We expose the running value as ``val_auc`` / ``test_auc`` for
early-stopping and checkpoint selection.
"""

from __future__ import annotations

import torch
from torch import nn


def build_lit_module(
    model: nn.Module,
    *,
    lr: float = 2.0e-3,
    weight_decay: float = 1.0e-4,
    max_epochs: int = 80,
    pos_weight: float = 1.0,
):
    """Build a concrete ``pl.LightningModule`` wrapping ``model``.

    The class is constructed lazily so this module stays importable when
    ``pytorch_lightning`` is not installed (eval-only, XAI-only, smoke tests).
    """
    import pytorch_lightning as pl
    from torchmetrics.classification import BinaryAUROC

    class _LitBovin(pl.LightningModule):
        def __init__(
            self,
            model: nn.Module,
            lr: float,
            weight_decay: float,
            max_epochs: int,
            pos_weight: float,
        ):
            super().__init__()
            self.save_hyperparameters(ignore=["model"])
            self.model = model
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
            )
            self.train_auc = BinaryAUROC()
            self.val_auc = BinaryAUROC()
            self.test_auc = BinaryAUROC()

        def forward(self, data):
            return self.model(data)

        def _step(self, batch, stage: str):
            out = self.model(batch)
            logit = out["logit"].view(-1)
            y = batch.y.to(torch.float32).view(-1)
            loss = self.loss_fn(logit, y)

            prob = torch.sigmoid(logit.detach())
            metric = getattr(self, f"{stage}_auc")
            metric.update(prob, y.to(torch.long))

            bs = y.size(0)
            self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), batch_size=bs)
            self.log(f"{stage}_auc", metric, prog_bar=True, batch_size=bs)
            return loss

        def training_step(self, batch, batch_idx):  # noqa: D401
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx):
            self._step(batch, "val")

        def test_step(self, batch, batch_idx):
            self._step(batch, "test")

        def configure_optimizers(self):
            opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(self.hparams.max_epochs, 1)
            )
            return {"optimizer": opt, "lr_scheduler": sched}

    return _LitBovin(model, lr, weight_decay, max_epochs, pos_weight)
