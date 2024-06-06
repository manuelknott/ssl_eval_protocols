import os
import tarfile
import wget
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
from PIL import Image
import torchvision
from torchvision.datasets import VisionDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_transform(augment=False, pad=False):
    resize_fun = max if pad else min

    if augment:
        transforms = [
            torchvision.transforms.Lambda(lambda x: torchvision.transforms.CenterCrop(resize_fun(x.size))(x)),
            # center crop to smaller dim
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip()
        ]
    else:
        transforms = [
            torchvision.transforms.Lambda(lambda x: torchvision.transforms.CenterCrop(resize_fun(x.size))(x)),
            # center crop to smaller dim
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224)
        ]

    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transforms = torchvision.transforms.Compose(transforms)
    return transforms


class Cub_200_2011(VisionDataset):

    def __init__(self, root, split: str, **kwargs):
        self._download(root)
        super().__init__(os.path.join(root, "CUB_200_2011"), **kwargs)

        self.split = split

        images = pd.read_csv(os.path.join(self.root, "images.txt"), sep=" ", header=None)
        images.columns = ["id", "path"]
        images = images.set_index("id")

        split_def = pd.read_csv(os.path.join(self.root, "train_test_split.txt"), sep=" ", header=None)
        split_def.columns = ["id", "is_train"]
        split_def = split_def.set_index("id")
        split_idx = split_def[split_def["is_train"] == (split == "train")].index

        classes = pd.read_csv(os.path.join(self.root, "classes.txt"), sep=" ", header=None)
        classes.columns = ["id", "name"]
        classes = classes.set_index("id")
        self.classes = classes.values.ravel().tolist()

        self.image_paths = images.loc[split_idx].values.ravel().tolist()

    def _download(self, root: str):
        if not os.path.isdir(root + "/CUB_200_2011"):
            os.makedirs(root + "/CUB_200_2011")
            filename = root + "/CUB_200_2011b.tgz"
            url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
            print("Downloading CUB_200_2011 dataset")
            wget.download(url, filename)
            print("Extracting CUB_200_2011 dataset")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(root)
            os.remove(filename)
            print("Done")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        class_name = image_path.split("/")[0]

        img = Image.open(os.path.join(self.root, "images", image_path)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.classes.index(class_name)

        return img, label


def cub(split: str, augment: bool = False):

    transform = get_transform(augment=augment, pad=True)

    return Cub_200_2011("data", split=split, transform=transform)


if __name__ == '__main__':
    ds = cub("train", augment=True)

    for x, y in ds:
        print(y)