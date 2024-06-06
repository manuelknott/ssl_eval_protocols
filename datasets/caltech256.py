import os
import ssl

from PIL import Image
import numpy as np
from torchvision.datasets import Caltech256
from torch.utils.data import Subset

from datasets.cub import get_transform

ssl._create_default_https_context = ssl._create_unverified_context

np.random.seed(0)
all_indices = np.arange(30607)
np.random.shuffle(all_indices)
train_indices = all_indices[:24485]
val_indices = all_indices[24485:]
train_indices.sort()
val_indices.sort()


class OurCaltech256(Caltech256):

    def __init__(self, root="data", transform=None, **kwargs):
        super().__init__(root=root, transform=transform, **kwargs)

    @property
    def classes(self):
        return self.categories

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[index]],
                f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg",
            )
        ).convert('RGB')

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def caltech256(split: str, augment: bool = False):

    transform = get_transform(augment=augment, pad=True)

    ds = OurCaltech256(root="data", transform=transform, download=True)
    _classes = ds.classes.copy()

    ds = Subset(ds, train_indices if split == 'train' else val_indices)
    ds.classes = _classes

    return ds


if __name__ == '__main__':
    ds = caltech256("train")
    print(ds.classes)
    for x, y in ds:
        assert x.shape == (3,224,224)
    ds = caltech256("val")
    for x, y in ds:
        assert x.shape == (3,224,224)