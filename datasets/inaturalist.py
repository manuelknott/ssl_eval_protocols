from typing import Tuple
import os

import numpy as np
import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config import config

inaturalist_dir = config["datasets"]["inaturalist"]["path"]
os.makedirs(inaturalist_dir, exist_ok=True)


class INaturalist2021Mini(torchvision.datasets.INaturalist):
    """INaturalist 2021 mini dataset."""

    def __init__(self, root, split: str, target_type: Tuple[str] = ("family",), select_ids=False, **kwargs):
        if split == "train":
            split = "2021_train_mini"
        elif split == "val":
            split = "2021_valid"
        else:
            raise ValueError(f"Invalid split: {split}")

        _path = os.path.join(root, split)

        if os.path.exists(_path) and len(os.listdir(_path)) > 0:
            download = False
        else:
            download = True

        super().__init__(root, version=split, download=download, **kwargs)

        self.select_ids = select_ids
        if select_ids:
            print("Filtering IDs from file.")
            assert len(target_type) == 1
            selected_ids = np.load(f"datasets/inat_{target_type[0]}_ids.npy")
            self.index = [idx for idx in self.index if idx[0] in selected_ids]
            if target_type[0] == "species":
                present_categories = sorted(list(set([i[0] for i in self.index])))
            else:
                present_categories = sorted(list(set([self.categories_map[idx[0]][target_type[0]] for idx in self.index])))
            self.meta_cat_map = {cat: i for i, cat in enumerate(present_categories)}

        if target_type[0] == "species":
            self.target_type = ("full",)
        else:
            self.target_type = target_type

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        if self.select_ids:
            target = self.meta_cat_map[target]
        return img, target


    @property
    def classes(self):
        assert len(self.target_type) == 1
        target = self.target_type[0]

        if self.select_ids:
            return list(range(len(self.meta_cat_map)))

        if target == "full":
            return list(range(10000))
        elif target == "genus":
            return list(range(4884))
        elif target == "family":
            return list(range(1103))
        elif target == "order":
            return list(range(273))
        elif target == "class":
            return list(range(51))
        elif target == "phylum":
            return list(range(13))
        elif target == "kingdom":
            return list(range(3))


def inaturalist(split: str, augment: bool = False, inat_target="family", select_ids=False):
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

    return INaturalist2021Mini(inaturalist_dir, split=split, transform=transforms, target_type=(inat_target,),
                               select_ids=select_ids
                               )


if __name__ == '__main__':
    ds = inaturalist("train", inat_target="species", select_ids=True)
    print(len(ds))
    print(len(ds.classes))
    ds = inaturalist("val",  inat_target="species", select_ids=True)
    x, y = ds[0]
    assert x.shape == (3, 224, 224)