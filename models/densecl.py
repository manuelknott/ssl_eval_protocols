import torch

from models.util import initialize_backbone, load_checkpoint

checkpoints = {
    "resnet50": {
        "url": "https://cloudstor.aarnet.edu.au/plus/s/hdAg5RYm8NNM2QP/download",
        "filename": "densecl_resnet50.pth",
    },
}


def load_model(arch: str, **kwargs):
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    for arch in checkpoints.keys():
        model = load_model(arch=arch)
        print(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(out.shape)