"""Simple PyTorch trainer for CrossMal-ViT."""

from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..evaluation.metrics import compute_metrics


class SimpleTrainer:
    """Minimal training loop for quick experiments."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        loss_fn: torch.nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        epochs: int = 10,
    ) -> Dict[str, float]:
        model.to(self.device)
        best_metrics: Dict[str, float] = {}

        for _epoch in range(epochs):
            model.train()
            for batch in train_loader:
                views = {
                    "raw": batch["raw"].to(self.device),
                    "entropy": batch["entropy"].to(self.device),
                    "frequency": batch["frequency"].to(self.device),
                }
                targets = batch["label"].to(self.device)

                outputs = model(views)
                logits = outputs["logits"]

                loss = loss_fn(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if val_loader is not None:
                metrics = self.evaluate(model, val_loader)
                if not best_metrics or metrics["accuracy"] > best_metrics.get("accuracy", 0.0):
                    best_metrics = metrics

        return best_metrics

    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in data_loader:
                views = {
                    "raw": batch["raw"].to(self.device),
                    "entropy": batch["entropy"].to(self.device),
                    "frequency": batch["frequency"].to(self.device),
                }
                targets = batch["label"].to(self.device)
                outputs = model(views)
                preds = outputs["logits"].argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        return compute_metrics(y_pred, y_true)
