"""Multi-view generation transform for malware byteplots."""

from typing import Dict, Union
import numpy as np
from PIL import Image
import torch
from .entropy_map import compute_local_entropy
from .frequency_map import compute_frequency_energy


class MultiViewTransform:
    """Generate three complementary views from a grayscale byteplot."""

    def __init__(
        self,
        img_size: int = 224,
        entropy_window: int = 9,
        normalize: bool = True,
    ) -> None:
        self.img_size = img_size
        self.entropy_window = entropy_window
        self.normalize = normalize

    def __call__(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert("L"))
        elif isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()

        if image.max() <= 1.0:
            image = (image * 255).astype(np.float32)
        else:
            image = image.astype(np.float32)

        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            image = np.array(pil_img).astype(np.float32)

        raw = image / 255.0
        entropy = compute_local_entropy(image, window_size=self.entropy_window)
        entropy = entropy / 8.0
        frequency = compute_frequency_energy(image)

        raw_tensor = torch.from_numpy(raw).unsqueeze(0).float()
        entropy_tensor = torch.from_numpy(entropy).unsqueeze(0).float()
        frequency_tensor = torch.from_numpy(frequency).unsqueeze(0).float()

        return {"raw": raw_tensor, "entropy": entropy_tensor, "frequency": frequency_tensor}


class MultiViewAugmentation:
    """Apply consistent augmentations across all views."""

    def __init__(self, flip_prob: float = 0.5, rotate_degrees: float = 15.0) -> None:
        self.flip_prob = flip_prob
        self.rotate_degrees = rotate_degrees

    def __call__(self, views: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() < self.flip_prob:
            views = {k: torch.flip(v, dims=[-1]) for k, v in views.items()}

        if self.rotate_degrees > 0:
            angle = (torch.rand(1).item() - 0.5) * 2 * self.rotate_degrees
            views = {k: self._rotate(v, angle) for k, v in views.items()}

        return views

    def _rotate(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        return TF.rotate(x, angle)
