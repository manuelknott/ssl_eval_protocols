import os

import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config import config

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

imagenet_dir = config["datasets"]["imagenet"]["path"]
os.makedirs(imagenet_dir, exist_ok=True)

n_files = {
    "train": 1281160, # actually 1281167,but we delete a corrupted file
    "val": 50000,
}


def imagenet(split: str, augment: bool = False):

    n_files_found = sum([len(x) for _, _, x in os.walk(os.path.join(imagenet_dir, split))])
    assert n_files_found >= n_files[split], \
    f"Imagenet {split} dataset is not complete. Found {n_files_found}, expected {n_files[split]} files."

    if augment:
        transforms = [
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip()
        ]
    else:
        transforms = [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224)
        ]

    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transforms = torchvision.transforms.Compose(transforms)

    return torchvision.datasets.ImageNet(imagenet_dir, split=split, transform=transforms)
