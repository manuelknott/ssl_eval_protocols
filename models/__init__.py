import torch
from models.util import freeze_model, unfreeze_model, partially_freeze_model
from config import config

AVAILABLE_MODELS = [

    ("jigsaw", "resnet50"),
    ("rotnet", "resnet50"),
    ("npid", "resnet50"),
    ("sela_v2", "resnet50"),
    ("npid_plusplus", "resnet50"),
    ("pirl", "resnet50"),
    ("clusterfit", "resnet50"),
    ("deepcluster_v2", "resnet50"),
    ("swav", "resnet50"),
    ("simclr", "resnet50"),
    ("moco_v2", "resnet50"),
    ("siamsiam", "resnet50"),
    ("byol", "resnet50"),
    ("dino", "resnet50"),
    ("barlowtwins", "resnet50"),
    ("densecl", "resnet50"),
    ("dino", "vitb16"),
    ("ibot", "vitb16"),
    #
    ("moco_v3", "resnet50"),
    ("moco_v3", "vitb16"),
    ("mae", "vitb16"),
    ("maskfeat", "vitb16"),
    ("beit_v2", "vitb16"),

    ("milan", "vitb16"),
    ("eva", "vitb16"),
    ("pixmim", "vitb16"),
]


def get_model(model: str, arch: str = "vitb16", freeze=True, **kwargs):
    assert freeze in [True, False, "partial"]
    out = __import__(f"models.{model}", fromlist=["get_model"]).load_model(arch, **kwargs)
    if freeze == "partial":
        partially_freeze_model(out)
    elif freeze is True:
        freeze_model(out)
    else:
        unfreeze_model(out)
    out = out.to(torch.device(config["device"]))
    return out


if __name__ == '__main__':
    print(AVAILABLE_MODELS)
