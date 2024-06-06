from functools import partial

import torch
import timm

from models.util import load_checkpoint

GLOBAL_POOL = False

checkpoints = {
    "vitb16": {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "filename": "mae_vitb16.pth",
    },
}


class MAE_ViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    Copied from: https://github.com/facebookresearch/mae/blob/main/models_vit.py
    """
    def __init__(self, **kwargs):
        super(MAE_ViT, self).__init__(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6, **kwargs))

        self.global_pool = GLOBAL_POOL
        if self.global_pool:
            norm_layer = partial(torch.nn.LayerNorm, eps=1e-6, **kwargs)
            embed_dim = 768
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.head = torch.nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]


        return outcome


def load_model(arch: str, **kwargs):
    assert arch == "vitb16"
    model = MAE_ViT()
    ckpt = load_checkpoint(**checkpoints[arch])["model"]
    if GLOBAL_POOL:
        ckpt["fc_norm.weight"] = ckpt["norm.weight"]
        ckpt["fc_norm.bias"] = ckpt["norm.bias"]
        del ckpt["norm.weight"]
        del ckpt["norm.bias"]

    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    for arch in checkpoints.keys():
        model = load_model(arch=arch)
        print(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(out.shape)
