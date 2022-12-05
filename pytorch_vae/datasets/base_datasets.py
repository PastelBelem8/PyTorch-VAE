from torchvision.datasets import CelebA, folder
from torchvision import transforms
from torch.utils.data import Dataset

from pathlib import Path
from typing import Callable, Sequence, Union


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True

    @staticmethod
    def get_data_splits(
        data_dir: str,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        center_crop: int = 148,
        **kwargs
    ):
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(center_crop),
            transforms.Resize(patch_size),
            transforms.ToTensor()
        ])

        val_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(center_crop),
            transforms.Resize(patch_size),
            transforms.ToTensor()
        ])

        train_dataset = MyCelebA(
            data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )

        val_dataset = MyCelebA(
            data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = folder.default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0 # dummy datat to prevent breaking


    @staticmethod
    def get_data_splits(
        data_dir: str,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        **kwargs
    ):
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(patch_size),
            # transforms.Resize(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        val_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(patch_size),
            # transforms.Resize(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset = OxfordPets(
            data_dir,
            split='train',
            transform=train_transforms,
        )

        val_dataset = OxfordPets(
            data_dir,
            split='val',
            transform=val_transforms,
        )

        return train_dataset, val_dataset