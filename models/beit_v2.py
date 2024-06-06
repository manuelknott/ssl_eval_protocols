import torch

from models.util import load_checkpoint, prepare_state_dict
from models.impl_utils.beit_model import beit_base_patch16_224
from models.impl_utils.beit_state_dict import prepare_state_dict, load_state_dict

checkpoints = {
    "vitb16": {
        "url": "https://drive.google.com/uc?id=1v9MzCK4GVTKiwNA0tdICAmVPZFgV8OVx",
        "filename": "beit_v2_vitb16.pth",
    },
}


def get_model(arch: str, **kwargs):
    assert arch == "vitb16", f"Invalid arch: {arch}"
    model = beit_base_patch16_224(
        pretrained=False,
        num_classes=0,
        drop_rate=0.0,
        #drop_path_rate=0.1,
        attn_drop_rate=0.0,
  #      drop_block_rate=None,
       use_mean_pooling=True,
       init_scale=0.001,
       use_rel_pos_bias=True,
       use_abs_pos_emb=False,
       init_values=0.1,
        qkv_bias=True,
        **kwargs
    )
    model.head = torch.nn.Identity()
    return model


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = get_model(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["model"]
    ckpt, model = prepare_state_dict(ckpt, model)
    load_state_dict(model, ckpt)
    return model


if __name__ == '__main__':
    model = load_model("vitb16")
    print(model)
    out = model(torch.randn(2, 3, 224, 224))
    print(out.shape)
