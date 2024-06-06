import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import config

BASE_DIR = config["dataloader"]["precalculated_embeddings_path"]


class PrecalculatedEmbeddings(Dataset):

    @classmethod
    def exists(cls, method: str, arch: str, dataset: str, split: str):
        return os.path.exists(f"{BASE_DIR}/{method}_{arch}_{dataset}_{split}.npy")

    def __init__(self, method: str, arch: str, dataset: str, split: str):
        assert self.exists(method, arch, dataset, split) is True, "Cannot find precalculated embeddings"

        data_filepath = f"{BASE_DIR}/{method}_{arch}_{dataset}_{split}.npy"
        self.data = torch.from_numpy(np.load(data_filepath))

        labels_filepath = f"{BASE_DIR}/{dataset}_{split}_labels.npy"
        self.labels = torch.from_numpy(np.load(labels_filepath))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @property
    def data_numpy(self):
        return self.data.numpy()

    @property
    def labels_numpy(self):
        return self.labels.numpy()
