from dataclasses import dataclass
from typing import List, Optional

import pytorch_lightning as pl
import torch
from comer.datamodule.dataset import CROHMEDataset, build_lazy_dataset
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from .vocab import vocab


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    return Batch(fnames, x, x_mask, seqs_y)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: Optional[str] = None,
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path or val_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

        print(f"Load train from: {self.train_path}")
        print(f"Load val from: {self.val_path}")
        if self.test_path != self.val_path:
            print(f"Load test from: {self.test_path}")

    def _make_dataset(self, data_path: str, batch_size: int, is_train: bool):
        samples, batch_indices = build_lazy_dataset(data_path, batch_size)
        return CROHMEDataset(
            samples,
            batch_indices,
            is_train=is_train,
            scale_aug=self.scale_aug,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset(
                self.train_path, self.train_batch_size, is_train=True
            )
            self.val_dataset = self._make_dataset(
                self.val_path, self.eval_batch_size, is_train=False
            )
        if stage == "test" or stage is None:
            self.test_dataset = self._make_dataset(
                self.test_path, self.eval_batch_size, is_train=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
