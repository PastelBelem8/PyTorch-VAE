# Code adapted from https://www.kaggle.com/code/maunish/training-vae-on-imagenet-pytorch
from .utils import get_paths_from_folders
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

        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, 0.0 # COde expects a label, even though we do not use it

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def get_data_splits(
        train_dir: str,
        eval_dir: str,
        image_size: int=64,
        **_
    ):

        train_paths = get_paths_from_folders(train_dir)
        train_paths = [p for p in train_paths if p.suffix == ".JPEG"]

        eval_paths = get_paths_from_folders(eval_dir)
        eval_paths = [p for p in eval_paths if p.suffix == ".JPEG"]

        train_transforms = get_train_transforms(image_size)

        train_dataset = ImageNetDataset(
            train_paths,
            augmentations=train_transforms,
        )

        val_dataset = ImageNetDataset(
            eval_paths,
            augmentations=get_train_transforms(image_size),
        )

        print("#Train:", len(train_dataset))
        print("#Eval:", len(val_dataset))
        return train_dataset, val_dataset


if __name__ == "__main__":
    # Test
    # paths = get_paths_from_folders("/home/kat/Projects/PhD/coursework/PyTorch-VAE/data/tiny-imagenet-200/train")
    paths = get_paths_from_folders("/home/kat/Projects/PhD/coursework/PyTorch-VAE/data/tiny-imagenet-200/val")

    transforms = get_train_transforms(64)
    dataset = ImageNetDataset(paths, transforms)


    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    dataiter = iter(dataloader)
    sample = dataiter.next()

    import torchvision.utils
    import matplotlib.pyplot as plt

    img = torchvision.utils.make_grid(sample).permute(1,2,0).numpy()
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.savefig("./example.png")
