"""Dataset implementations."""

from .base_dataset import ImageClassificationDataset
from .kaggle_malware import KaggleMalwareDataset

__all__ = ["ImageClassificationDataset", "KaggleMalwareDataset"]
