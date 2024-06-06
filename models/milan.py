import torch

from models.util import load_checkpoint, initialize_backbone, prepare_state_dict

checkpoints = {
    "vitb16": {
        "url": "https://drive.google.com/uc?id=18UYGG_1r5SJyAgj1ykOfoqECdVBoFLoz", # TODO download not working via gdown
        "filename": "milan_vitb16.pth.tar"
    },
}

def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["model"]
    ckpt = prepare_state_dict(ckpt, delete_prefixes=("mask_token", "decoder"))
    model.load_state_dict(ckpt)
    #model.load_state_dict(ckpt, strict=False)
    return model

if __name__ == '__main__':
    model = load_model("vitb16")
    print(model)
    out = model(torch.randn(1, 3, 224, 224))
    print(out.shape)
