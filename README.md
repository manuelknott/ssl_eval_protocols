# Comparing evaluation protocols for self-supervised pre-training with image classification

This repository contains supplementary code for the paper [A Closer Look at Benchmarking Self-Supervised Pre-Training with Image Classification](https://arxiv.org/abs/2407.12210).

## How to use

1. Set up Python environment (3.8 or higher)
2. run `setup.sh` to install dependencies
3. Set paths in `paths.yaml` (no need to create the folders manually)
4. Manually run `download_imagenet.sh` or `download_imagenet_d.sh` if needed. Other datasets and checkpoints get downloaded on-the-fly.
5. Use one of the following scripts as your entry point:
    - `run_linear_probe.py`
    - `run_finetuning.py`
    - `run_knn_probe.py` (we recommend precalculating embeddings with `precalculate_embeddings.py` beforehand)
    - `run_fewshot_finetuning.py`

All scripts currently log results to `wandb`. You might need to adapt the scripts if you do not want to use `wandb`.

## Citation

If you find this repo useful, please consider citing us:

```
@article{marks2024benchmarking,
  title={A Closer Look at Benchmarking Self-Supervised Pre-training with Image Classification},
  author={Marks, Markus and Knott, Manuel and Kondapaneni, Neehar and Cole, Elijah and Defraeye, Thijs and Perez-Cruz, Fernando and Perona, Pietro},
  journal={arXiv preprint arXiv:2407.12210},
  year={2024}
}
```

## Acknowledgements

This repo uses code and checkpoints adapted from different repositories:

- Jigsaw (from [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md))
- Rotnet (from [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md))
- NPID (from [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md))
- SeLa-v2 (from [SwAV](https://github.com/facebookresearch/swav) repo)
- NPID++ (from [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md))
- PIRL (from [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md))
- Clusterfit (from [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md))
- DeepCluster-v2 (from [SwAV](https://github.com/facebookresearch/swav) repo)
- [SwAV](https://github.com/facebookresearch/swav)
- SIMCLR (from [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md))
- [MoCo v2](https://github.com/facebookresearch/moco)
- SiamSiam (from [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html))
- BYOL ([Inofficial Pytorch implementation](https://github.com/yaox12/BYOL-PyTorch)))
- Barlow Twins (from [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html))
- [DenseCL](https://github.com/WXinlong/DenseCL)
- [DINO](https://github.com/facebookresearch/dino)
- [MoCo v3](https://github.com/facebookresearch/moco-v3)
- [iBOT](https://github.com/bytedance/ibot)
- [MAE](https://github.com/facebookresearch/mae)
- MaskFeat (from [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html))
- [BEiT v2](https://github.com/microsoft/unilm/tree/master/beit2)
- [MILAN](https://github.com/zejiangh/milan)
- EVA (from [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html))
- PixMIM (from [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html))
