import os
import argparse

import numpy as np
import torch.nn as nn

from protocols.procedures.classification_sklearn import get_embeddings
from models import AVAILABLE_MODELS, get_model
from datasets import DataModule
from config import config

BASE_DIR = config["dataloader"]["precalculated_embeddings_path"]
batch_size = 512

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", default=False)
args = parser.parse_args()

for dataset in ["imagenet", "inaturalist", "cub", "pascal", "cifar100", "imagenet_d_clipart", "imagenet_d_infograph", "imagenet_d_painting", "imagenet_d_quickdraw", "imagenet_d_real", "imagenet_d_sketch"]:
    for split in ["train", "val"]:
        for i, (method, arch) in enumerate(AVAILABLE_MODELS):
            if os.path.exists(f"{BASE_DIR}/{method}_{arch}_{dataset}_{split}.npy") and args.force is False:
                print(f"Embeddings for {method} {arch} on {dataset} {split} already exist.")
                continue
            print(f"Calculate embeddings for {method} {arch} on {dataset} {split}...")

            dm = DataModule(dataset, batch_size=batch_size, shuffle_train=False, augment_train=False)
            loader = dm.train_dataloader() if split == "train" else dm.val_dataloader()
            model = get_model(method, arch, freeze=True).cuda()
            model = nn.DataParallel(model)

            X, y = get_embeddings(model, loader)
            print(X.shape, y.shape)

            os.makedirs(BASE_DIR, exist_ok=True)
            np.save(f"{BASE_DIR}/{method}_{arch}_{dataset}_{split}.npy", X)
            if not os.path.exists(f"{BASE_DIR}/{dataset}_{split}_labels.npy"):
                np.save(f"{BASE_DIR}/{dataset}_{split}_labels.npy", y)


