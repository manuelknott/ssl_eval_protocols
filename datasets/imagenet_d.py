from torch.utils.data.dataset import Dataset
import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image

from config import config

datapath = config["datasets"]["imagenet_d"]["path"]


class ImagenetD(Dataset):

    def __init__(self, subset: str, split: str, transform=None):
        self.split = split
        self.transform = transform

        # read txt to list
        with open(f"{datapath}/{subset}_{'train' if split == 'train' else 'test'}.txt", "r") as f:
            self.filenames = f.readlines()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath, label = self.filenames[idx].split(" ")
        img = Image.open(f"{datapath}/{filepath}")

        if self.transform:
            img = self.transform(img)

        return img, int(label)

    @property
    def classes(self):
        return [f"c{i}" for i in range(1000)]


def imagenet_d(subset: str, split: str, augment: bool = False):
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

    return ImagenetD(subset, split=split, transform=transforms)


def imagenet_d_clipart(split: str, augment: bool = False):
    return imagenet_d("clipart", split, augment=augment)


def imagenet_d_infograph(split: str, augment: bool = False):
    return imagenet_d("infograph", split, augment=augment)


def imagenet_d_painting(split: str, augment: bool = False):
    return imagenet_d("painting", split, augment=augment)


def imagenet_d_quickdraw(split: str, augment: bool = False):
    return imagenet_d("quickdraw", split, augment=augment)


def imagenet_d_real(split: str, augment: bool = False):
    return imagenet_d("real", split, augment=augment)


def imagenet_d_sketch(split: str, augment: bool = False):
    return imagenet_d("sketch", split, augment=augment)


if __name__ == '__main__':
    ds = imagenet_d_clipart("val", augment=True)
    print(len(ds))
