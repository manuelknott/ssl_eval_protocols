import torch
from mmcls.models import VisionTransformer as MMViT

from models.util import load_checkpoint, prepare_state_dict

GLOBAL_POOL = False

checkpoints = {
    "vitb16": {
        "url": "https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth",
        "filename": "maskfeat_vitb16.pth",
    },
}


class MMSelfSupMaskFeatViT(MMViT):
    def __init__(self):
        super().__init__(img_size=224, patch_size=16)
        self.global_pool = GLOBAL_POOL

    def forward(self, x):
        x = super().forward(x)
        if self.global_pool:
            x = x[0][0]
            return x.reshape(x.shape[0], 768, -1).mean(dim=-1)
        else:
            return x[0][1]


def load_model(arch: str, **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = MMSelfSupMaskFeatViT()
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = prepare_state_dict(ckpt, remove_prefix="backbone.", delete_prefixes=["target_generator.", "neck.", "mask_token"])
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    for arch in checkpoints.keys():
        model = load_model(arch=arch)
        print(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(out.shape)
