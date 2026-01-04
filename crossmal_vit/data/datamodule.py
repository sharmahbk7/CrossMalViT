"""PyTorch Lightning DataModule for malware datasets."""

from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets.kaggle_malware import KaggleMalwareDataset
from .samplers.balanced_sampler import BalancedSampler


class MalwareDataModule(pl.LightningDataModule):
    """Lightning DataModule for the Kaggle Malware dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        img_size: int = 224,
        transform: Optional[object] = None,
        multi_view_transform: Optional[object] = None,
        balanced_sampling: bool = False,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.transform = transform
        self.multi_view_transform = multi_view_transform
        self.balanced_sampling = balanced_sampling
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = KaggleMalwareDataset(
            root=self.data_dir,
            split="train",
            transform=self.transform,
            multi_view_transform=self.multi_view_transform,
            img_size=self.img_size,
        )
        self.val_dataset = KaggleMalwareDataset(
            root=self.data_dir,
            split="val",
            transform=self.transform,
            multi_view_transform=self.multi_view_transform,
            img_size=self.img_size,
        )
        self.test_dataset = KaggleMalwareDataset(
            root=self.data_dir,
            split="test",
            transform=self.transform,
            multi_view_transform=self.multi_view_transform,
            img_size=self.img_size,
        )

    def train_dataloader(self) -> DataLoader:
        if self.balanced_sampling and self.train_dataset is not None:
            labels = [label for _, label in self.train_dataset.samples]
            sampler = BalancedSampler(labels)
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
