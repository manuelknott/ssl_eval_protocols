import os

import numpy as np
import torch
from torchvision.datasets import VOCSegmentation

from datasets.cub import get_transform


class VOCClassification(VOCSegmentation):

    classes = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
               'horse', 'motorcycle', 'person', 'potted plant', 'sheep',
               'sofa', 'train', 'television']

    def __init__(self, root="data", split: str = "train", **kwargs):
        super().__init__(os.path.join(root, "VOC"), year="2012", image_set=split, download=True, **kwargs)

    def __getitem__(self, index):
        img, mask = super().__getitem__(index)

        mask = torch.from_numpy(np.asarray(mask, dtype='int64')).ravel()
        mask = mask[(mask != 0) & (mask != 255)]
        unique_values, value_counts = np.unique(mask, return_counts=True)
        class_label = unique_values[np.argmax(value_counts)]

        return img, class_label


def pascal(split: str, augment: bool = False):
    transform = get_transform(augment=augment, pad=False)

    return VOCClassification("data", split=split, transform=transform)


if __name__ == '__main__':
    ds = pascal("train")
    print(ds.classes)