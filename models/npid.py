import torch

from models.util import load_checkpoint, initialize_backbone, prepare_state_dict

checkpoints = {
    "resnet50": {
        "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_npid_lemniscate_neg4k_stepLR_8gpu.torch",
        "filename": "npid_resnet50.torch"
    },
}


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["model_state_dict"]
    ckpt = {k: torch.from_numpy(v) for k, v in ckpt.items()}
    ckpt = prepare_state_dict(ckpt, remove_prefix="_feature_blocks.", delete_prefixes=["fc"])
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    for arch in checkpoints.keys():
        model = load_model(arch=arch)
        print(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(out.shape)
