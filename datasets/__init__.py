import os

import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

from datasets.imagenet import imagenet
from datasets.inaturalist import inaturalist
from datasets.cub import cub
from datasets.pascal import pascal
from datasets.caltech256 import caltech256
from datasets.dummy import Dummy
from datasets.imagenet_d import imagenet_d_clipart, imagenet_d_infograph, imagenet_d_painting, imagenet_d_quickdraw,\
    imagenet_d_real, imagenet_d_sketch
from datasets.cifar100 import cifar100
from config import config

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Subset(torch.utils.data.Subset):

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes


class DataModule(pl.LightningDataModule):

    def __init__(self, dataset: str, batch_size: int = 64,
                 num_workers: int = config["dataloader"]["num_workers"], shuffle_train=True, augment_train=True,
                 subset_fraction: float = 1.0
                 ):
        super().__init__()
        if dataset == "dummy":
            ds_class = Dummy
        elif dataset == "imagenet":
            ds_class = imagenet
        elif dataset.startswith("inaturalist"):
            ds_class = inaturalist
        elif dataset == "cub":
            ds_class = cub
        elif dataset == "pascal":
            ds_class = pascal
        elif dataset == "caltech256":
            ds_class = caltech256
        elif dataset == "imagenet_d_clipart":
            ds_class = imagenet_d_clipart
        elif dataset == "imagenet_d_infograph":
            ds_class = imagenet_d_infograph
        elif dataset == "imagenet_d_painting":
            ds_class = imagenet_d_painting
        elif dataset == "imagenet_d_quickdraw":
            ds_class = imagenet_d_quickdraw
        elif dataset == "imagenet_d_real":
            ds_class = imagenet_d_real
        elif dataset == "imagenet_d_sketch":
            ds_class = imagenet_d_sketch
        elif dataset == "cifar100":
            ds_class = cifar100
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented for now.")


        ds_kwargs = {}
        if dataset == "inaturalist_genus":
            ds_kwargs["inat_target"] = "genus"
            ds_kwargs["select_ids"] = True
        elif dataset == "inaturalist_species":
            ds_kwargs["inat_target"] = "species"
            ds_kwargs["select_ids"] = True


        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

        self.train_set = ds_class("train", augment=augment_train, **ds_kwargs)

        if subset_fraction < 1.0:
            assert subset_fraction > 0.0

            n_per_class = int(subset_fraction * len(self.train_set) / self.num_classes)
            precalc_file = config["dataloader"]["precalculated_embeddings_path"] + f"/{dataset}_train_labels.npy"
            if os.path.exists(precalc_file):
                print("Using precalculated file. (faster)")
                labels = np.load(precalc_file)
            else:
                print("No precalculated file found. Calculating once and saving for future runs.")
                labels = None
                for _, y in tqdm(self.train_dataloader(), desc="Preparing indexes"):
                    if labels is None:
                        labels = y
                    else:
                        labels = torch.cat((labels, y), dim=0)
                labels = labels.cpu().numpy().ravel()
                np.save(precalc_file, labels)
            np.random.seed(42)
            subset_indices = np.hstack([np.random.choice(np.where(labels == l)[0], n_per_class, replace=False)
                                        for l in np.unique(labels)])

            self.train_set = Subset(self.train_set, subset_indices)

        self.val_set = ds_class("val", augment=False,  **ds_kwargs)


    @property
    def num_classes(self):
        return len(self.train_set.classes)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle_train,
                                           num_workers=self.num_workers
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers
                                           )