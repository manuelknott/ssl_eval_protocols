import torch
from models import AVAILABLE_MODELS, get_model
from config import config
device = config["device"]

for method, arch in AVAILABLE_MODELS:
    try:
        model = get_model(method, arch)
        out = model(torch.randn(1, 3, 224, 224).to(device))
        print(f"{method} {arch} { out.shape}")
    except:
        print("Failed for: ", method, arch)
