"""Standard image augmentations for malware byteplots."""

from torchvision import transforms


class StandardAugmentations:
    """Standard augmentations applied to single images."""

    def __init__(
        self,
        img_size: int = 224,
        rotation: float = 10.0,
        flip_prob: float = 0.5,
        jitter: float = 0.1,
    ) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=flip_prob),
                transforms.RandomRotation(degrees=rotation),
                transforms.ColorJitter(brightness=jitter, contrast=jitter),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
