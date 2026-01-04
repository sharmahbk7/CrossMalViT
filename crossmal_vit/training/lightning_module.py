"""PyTorch Lightning module for CrossMal-ViT training."""

from typing import Dict
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models import build_crossmal_vit
from ..losses import LDAMLoss, MultiViewContrastiveLoss
from ..evaluation import compute_metrics, compute_ece


class CrossMalViTModule(pl.LightningModule):
    """Lightning module for CrossMal-ViT."""

    def __init__(self, model_config: Dict, train_config: Dict, cls_num_list: list) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = build_crossmal_vit(model_config)

        self.ldam_loss = LDAMLoss(
            cls_num_list=cls_num_list,
            max_margin=train_config.get("ldam_margin", 0.48),
            scale=train_config.get("ldam_scale", 30.0),
        )
        self.contrastive_loss = MultiViewContrastiveLoss(
            embed_dim=model_config.get("embed_dim", 768),
            temperature=train_config.get("temperature", 0.07),
        )
        self.lambda_contrast = train_config.get("lambda_contrast", 0.32)

        self.lr = train_config.get("learning_rate", 1.2e-4)
        self.weight_decay = train_config.get("weight_decay", 0.048)
        self.max_epochs = train_config.get("max_epochs", 100)

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(x, return_features=True)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        views = {
            "raw": batch["raw"],
            "entropy": batch["entropy"],
            "frequency": batch["frequency"],
        }
        targets = batch["label"]

        outputs = self.forward(views)
        logits = outputs["logits"]
        view_features = outputs["view_features"]

        cls_loss = self.ldam_loss(logits, targets)
        contrast_loss = self.contrastive_loss(view_features)
        total_loss = cls_loss + self.lambda_contrast * contrast_loss

        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/cls_loss", cls_loss)
        self.log("train/contrast_loss", contrast_loss)
        self.log("train/acc", acc, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        views = {
            "raw": batch["raw"],
            "entropy": batch["entropy"],
            "frequency": batch["frequency"],
        }
        targets = batch["label"]

        outputs = self.forward(views)
        logits = outputs["logits"]

        self.validation_step_outputs.append({"logits": logits, "targets": targets})
        cls_loss = self.ldam_loss(logits, targets)
        self.log("val/loss", cls_loss, prog_bar=True, sync_dist=True)

        return {"logits": logits, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        probs = torch.softmax(all_logits, dim=-1)
        preds = all_logits.argmax(dim=-1)

        metrics = compute_metrics(
            preds.cpu().numpy(),
            all_targets.cpu().numpy(),
            probs.cpu().numpy(),
        )

        ece = compute_ece(probs.cpu().numpy(), all_targets.cpu().numpy())

        self.log("val/accuracy", metrics["accuracy"], prog_bar=True)
        self.log("val/macro_f1", metrics["macro_f1"], prog_bar=True)
        self.log("val/weighted_f1", metrics["weighted_f1"])
        self.log("val/ece", ece)

        self.validation_step_outputs.clear()

    def test_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        views = {
            "raw": batch["raw"],
            "entropy": batch["entropy"],
            "frequency": batch["frequency"],
        }
        targets = batch["label"]

        outputs = self.forward(views)
        logits = outputs["logits"]

        self.test_step_outputs.append({"logits": logits, "targets": targets})
        return {"logits": logits, "targets": targets}

    def on_test_epoch_end(self) -> None:
        all_logits = torch.cat([x["logits"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])

        probs = torch.softmax(all_logits, dim=-1)
        preds = all_logits.argmax(dim=-1)

        metrics = compute_metrics(
            preds.cpu().numpy(),
            all_targets.cpu().numpy(),
            probs.cpu().numpy(),
        )

        ece = compute_ece(probs.cpu().numpy(), all_targets.cpu().numpy())

        self.log("test/accuracy", metrics["accuracy"])
        self.log("test/macro_f1", metrics["macro_f1"])
        self.log("test/macro_precision", metrics["macro_precision"])
        self.log("test/macro_recall", metrics["macro_recall"])
        self.log("test/weighted_f1", metrics["weighted_f1"])
        self.log("test/ece", ece)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
