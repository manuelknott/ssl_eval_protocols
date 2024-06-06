import torch

from models.util import MMSelfSupResnet50 as Resnet50
from models.util import load_checkpoint, prepare_state_dict

checkpoints = {
    "resnet50": {
        "url": "https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220825-57307488.pth",
        "filename": "barlowtwins_resnet50.pth",
    },
}


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = Resnet50()
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = prepare_state_dict(ckpt, remove_prefix="backbone.", delete_prefixes=["data_preprocessor.", "neck.", "head."])  # TODO what does data_preprocessor do
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    for arch in checkpoints.keys():
        model = load_model(arch=arch)
        print(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(out.shape)
