import os
import yaml
import torch

paths = yaml.safe_load(open("paths.yaml", "r"))

os.makedirs(paths["pretrained_checkpoints"], exist_ok=True)
os.makedirs(paths["precalculated_embeddings"], exist_ok=True)

config = {
    "pretrained_checkpoint_dir": paths["pretrained_checkpoints"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gpu_count": torch.cuda.device_count(),
    "dataloader": {
        "num_workers": min(10, os.cpu_count()),
        "precalculated_embeddings_path": paths["precalculated_embeddings"],
    },
    "datasets": {
        "imagenet": {
            "path": paths["datasets"]["imagenet"],
        },
        "inaturalist": {
            "path": paths["datasets"]["inaturalist"],
        },
        "imagenet_d": {
            "path": paths["datasets"]["imagenet_d"],
        }
    },
}