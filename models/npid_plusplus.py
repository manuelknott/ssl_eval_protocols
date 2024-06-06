import torch

from models.util import load_checkpoint, initialize_backbone, prepare_state_dict

checkpoints = {
    "resnet50": {
        "url": "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_pp/4node_800ep_32kneg_cosine_resnet_23_07_20.75432662/model_final_checkpoint_phase799.torch",
        "filename": "npid_plusplus_resnet50.torch",
    },
}


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["classy_state_dict"]["base_model"]["model"]["trunk"]
    ckpt = prepare_state_dict(ckpt, remove_prefix="_feature_blocks.")
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    for arch in checkpoints.keys():
        model = load_model(arch=arch)
        print(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(out.shape)
