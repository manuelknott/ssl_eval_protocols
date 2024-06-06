import torch
from torch.utils.data import Dataset


class Dummy(Dataset):

    def __init__(self, split="train", augment=False, len=100):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.randint(0, 1000, (1,))

    @property
    def classes(self):
        return [f"c{i}" for i in range(1000)]
