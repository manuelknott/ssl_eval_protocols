from models.util import initialize_backbone, prepare_state_dict, load_checkpoint

checkpoints = {
    "resnet50": {
        "url": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
        "filename": "mocov2_resnet50.pth.tar",
    },
}


def load_model(arch: str = "resnet50", **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = prepare_state_dict(ckpt, remove_prefix="module.encoder_q.")
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    model = load_model()
    print(model)
