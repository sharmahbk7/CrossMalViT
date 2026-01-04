"""Base dataset utilities."""

from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):
    """Generic image classification dataset based on a list of samples."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        if isinstance(image, Image.Image):
            import numpy as np

            image = torch.from_numpy(np.array(image).astype("float32") / 255.0).unsqueeze(0)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }
