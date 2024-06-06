from torchvision.datasets import CIFAR100

from datasets.cub import get_transform


def cifar100(split: str, augment: bool = False):

    transform = get_transform(augment=augment, pad=True)

    return CIFAR100(root="data", train=split=="train", transform=transform, download=True)

if __name__ == '__main__':
    ds = cifar100("val", augment=True)
    print(ds[0][0].shape)