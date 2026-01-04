"""Mixup and CutMix augmentation for multi-view inputs."""

from typing import Dict, Tuple
import numpy as np
import torch


class MixupCutmix:
    """Apply Mixup or CutMix augmentation to multi-view batches."""

    def __init__(
        self,
        mixup_alpha: float = 0.42,
        cutmix_alpha: float = 1.0,
        cutmix_prob: float = 0.52,
        switch_prob: float = 0.5,
        num_classes: int = 18,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes

    def __call__(
        self, views: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        bsz = targets.shape[0]
        device = targets.device

        targets_onehot = torch.zeros(bsz, self.num_classes, device=device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

        use_cutmix = torch.rand(1).item() < self.cutmix_prob

        if use_cutmix:
            return self._cutmix(views, targets_onehot)
        return self._mixup(views, targets_onehot)

    def _mixup(
        self, views: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        bsz = targets.shape[0]
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(bsz, device=targets.device)

        mixed_views = {}
        for view_name, view_tensor in views.items():
            mixed_views[view_name] = lam * view_tensor + (1 - lam) * view_tensor[index]

        mixed_targets = lam * targets + (1 - lam) * targets[index]
        return mixed_views, mixed_targets

    def _cutmix(
        self, views: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        bsz, _, height, width = list(views.values())[0].shape

        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        index = torch.randperm(bsz, device=targets.device)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(height, width, lam)

        mixed_views = {}
        for view_name, view_tensor in views.items():
            mixed = view_tensor.clone()
            mixed[:, :, bby1:bby2, bbx1:bbx2] = view_tensor[index, :, bby1:bby2, bbx1:bbx2]
            mixed_views[view_name] = mixed

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (height * width))
        mixed_targets = lam * targets + (1 - lam) * targets[index]

        return mixed_views, mixed_targets

    def _rand_bbox(self, height: int, width: int, lam: float) -> Tuple[int, int, int, int]:
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        cy = np.random.randint(height)
        cx = np.random.randint(width)

        bbx1 = int(np.clip(cx - cut_w // 2, 0, width))
        bby1 = int(np.clip(cy - cut_h // 2, 0, height))
        bbx2 = int(np.clip(cx + cut_w // 2, 0, width))
        bby2 = int(np.clip(cy + cut_h // 2, 0, height))

        return bbx1, bby1, bbx2, bby2
