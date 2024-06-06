from models.util import load_checkpoint, initialize_backbone

checkpoints = {
    "vitb16": {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
        "filename": "ibot_vitb16.pth",
    },
    "vits16": {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth",
        "filename": "ibot_vits16.pth",
    },
    "vitl16": {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint_teacher.pth",
        "filename": "ibot_vitl16.pth",
    }
}


def load_model(arch: str = "vitb16", **kwargs):
    assert arch in checkpoints.keys(), f"Invalid arch: {arch}"
    model = initialize_backbone(arch, **kwargs)
    ckpt = load_checkpoint(**checkpoints[arch])["state_dict"]
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("head.")}  # remove head
    model.load_state_dict(ckpt)
    return model


if __name__ == '__main__':
    model = load_model()
    print(model)
