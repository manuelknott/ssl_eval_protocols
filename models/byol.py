import torch

from models.util import load_checkpoint, initialize_backbone, prepare_state_dict

checkpoints = {
    "resnet50": {
        "url": "https://drive.google.com/uc?id=1TLZHDbV-qQlLjkR8P0LZaxzwEE6O_7g1",
        "filename": "byol_resnet50.pth.tar"
    },
}


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["online_backbone"]
    ckpt = prepare_state_dict(ckpt, remove_prefix="module.")
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    for arch in checkpoints.keys():
        model = load_model(arch=arch)
        print(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(out.shape)