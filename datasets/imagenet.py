# Code adapted from https://www.kaggle.com/code/maunish/training-vae-on-imagenet-pytorch
from torch.utils.data import Dataset
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size, always_apply=True),
        A.Normalize(),
        ToTensorV2(p=1.0)
    ])


class ImageNetDataset(Dataset):
    """Download dataset from http://maxwell.cs.umass.edu/hsu/697l/tiny-imagenet-200.zip"""
    def __init__(self, paths, augmentations):
        self.paths = paths
        self.augmentations = augmentations

    def __getitem__(self, idx):
        path = self.paths[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image

    def __len__(self):
        return len(self.paths)


    @staticmethod
    def get_data_splits(
        data_dir: str,
        image_size,
        **kwargs
    ):
        train_transforms = get_train_transforms(image_size)
