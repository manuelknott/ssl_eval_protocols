import torch

from models.util import load_checkpoint, initialize_backbone, prepare_state_dict

checkpoints = {
    "vitb16": {
        "url": "https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k_20230322-e8137924.pth",
        "filename": "pixmim_vitb16.pth"
    },
}

rename_dict = {
    "layers.": "blocks.",
    "patch_embed.projection": "patch_embed.proj",
    ".ln1": ".norm1",
    ".ln2": ".norm2",
    "ln1.weight": "norm.weight",
    "ln1.bias": "norm.bias",
    "ffn.blocks.0.0.": "mlp.fc1.",
    "ffn.blocks.1.": "mlp.fc2.",
}

def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = prepare_state_dict(ckpt,
                              remove_prefix="backbone.",
                              delete_prefixes=("neck", "target_generator"))

    for k in list(ckpt.keys()):
        if k == "norm1.weight":
            print()
        old_k = k
        for key, val in rename_dict.items():
            k = k.replace(key, val)
        if k != old_k:
            ckpt[k] = ckpt[old_k]
            del ckpt[old_k]

    model.load_state_dict(ckpt)
    #model.load_state_dict(ckpt, strict=False)
    return model

if __name__ == '__main__':
    model = load_model("vitb16")
    print(model)
    out = model(torch.randn(1, 3, 224, 224))
    print(out.shape)
