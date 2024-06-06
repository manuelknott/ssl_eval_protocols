import torch
from models.util import load_checkpoint, initialize_backbone, prepare_state_dict

checkpoints = {
    "vitb16": {
        "url": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
        "filename": "mocov3_vitb16.pth.tar",
    },
    "resnet50": {
        "url": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar",
        "filename": "mocov3_resnet50.pth.tar",
    }
}


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = prepare_state_dict(ckpt, remove_prefix="module.base_encoder.", delete_prefixes=["module.predictor."])
    ckpt = prepare_state_dict(ckpt, remove_prefix="module.momentum_encoder.")
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':

    for arch in checkpoints.keys():
        model = load_model(arch)
        print(model)
        out = model(torch.randn(2, 3, 224, 224))
        print(out.shape)
