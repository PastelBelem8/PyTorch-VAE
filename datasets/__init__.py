import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


from base_datasets import MyCelebA, OxfordPets
from imagenet import ImageNetDataset


DATASETS = {
    "celebA": MyCelebA,
    "oxford-pets": OxfordPets,
    "imagenet": ImageNetDataset,
}



class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
    dataset_name: str
        Name of the dataset to load.

    data_dir: str
        root directory of your dataset.

    train_batch_size: int
        the batch size to use during training.

    val_batch_size: int
        the batch size to use during validation.
    patch_size: tuple(int, int)
        the size of the crop to take from the original images.

    num_workers: int
        the number of parallel workers to create to load data
        items (see PyTorch's Dataloader documentation for more details).

    pin_memory: bool, default False
        whether prepared items should be loaded into pinned memory
        or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        name: str,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dataset_class = DATASETS[name]
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.other_configs = kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset, self.val_dataset = self.dataset_class(**self.other_configs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
