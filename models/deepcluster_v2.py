import torch

from models.util import initialize_backbone, prepare_state_dict, load_checkpoint

checkpoints = {
    "resnet50": {
        "url": "https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar",
        "filename": "deepcluster_resnet50.pth.tar",
    },
}


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])
    ckpt = prepare_state_dict(ckpt, remove_prefix="module.", delete_prefixes=["projection_head.", "prototypes."])
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    for arch in checkpoints.keys():
        model = load_model(arch=arch)
        print(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(out.shape)
