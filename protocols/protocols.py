from typing import Tuple

from protocols.procedures.classification import train_classifier
from protocols.procedures.classification_sklearn import fit_classifier as fit_sklearn_classifier

wandb_default_args = {
    "project": "ssl_eval_protocols",
    "mode": "online",
    "log_model": "all"
}


def knn_probe(pretrained_model: Tuple[str, str], dataset: str, **kwargs):
    score, top5_score = fit_sklearn_classifier(pretrained_model=pretrained_model,
                                   shallow_clf="knn",
                                   dataset=dataset,
                                   use_precalculated_embeddings=True,
                                   return_top5=True,
                                   **kwargs
                                   )
    return score, top5_score


def linear_probe(pretrained_model: Tuple[str, str], dataset: str,
                 head: str = "linear",
                 n_epochs: int = 100,
                 lr=0.001,
                 batch_size=1024,
                 seed=42,
                 final_batch_norm=False,
                 accumulate_grad_batches=1,
                 save_checkpoints=True,
                 **trainer_kwargs):
    model, arch = pretrained_model
    print(f"Running {head} probe for {model} {arch} on {dataset}")
    wandb_args = {
        "group": f"{head}_probe_{dataset}",
        "name": f"{model}_{arch}",
        "mode": "online",
        "log_model": "all" if save_checkpoints else False
    }
    if final_batch_norm:
        wandb_args["tags"] = ["final_bn"]

    train_classifier(pretrained_model=pretrained_model,
                     head=head,
                     dataset=dataset,
                     freeze_model=True,
                     batch_size=batch_size,
                     num_epochs=n_epochs,
                     optim="sgd",
                     scheduler="warmup_cosine",
                     warmup_epochs=5,
                     lr=lr,
                     wandb_args={**wandb_default_args, **wandb_args},
                     seed=seed,
                     use_final_batch_norm=final_batch_norm,
                     accumulate_grad_batches=accumulate_grad_batches,
                     save_checkpoints=save_checkpoints,
                     **trainer_kwargs
                     )


def finetuning(pretrained_model: Tuple[str, str], dataset: str, batch_size=256, seed=42, checkpoint=None,
               final_batch_norm=False, save_checkpoints=True, lr=0.1,
               **trainer_kwargs):
    model, arch = pretrained_model
    print(f"End-to-end finetuning for {model} {arch} on {dataset}.")

    wandb_args = {
        "group": f"finetuning_{dataset}",
        "name": f"{model}_{arch}",
        "mode": "online",
        "log_model": "all" if save_checkpoints else False
    }
    if final_batch_norm:
        wandb_args["tags"] = ["final_bn"]

    n_epochs = 100
    optim = "sgd"
    scheduler = "warmup_cosine"
    warmup_epochs = 5
    unfreeze_after_epoch = None  # one epoch CLF only
    lr = lr
    layerwise_lr_decay = 0.65 if arch == "vitb16" else None
    label_smoothing = 0.1 if arch == "vitb16" else 0.0
    drop_path = 0.2 if arch == "vitb16" else None

    train_classifier(pretrained_model=pretrained_model,
                     head="linear",
                     dataset=dataset,
                     num_epochs=n_epochs,
                     subset_fraction=1.0,
                     freeze_model=False,
                     unfreeze_after_epoch=unfreeze_after_epoch,
                     batch_size=batch_size,
                     optim=optim,
                     scheduler=scheduler,
                     warmup_epochs=warmup_epochs,
                     lr=lr,
                     layerwise_lr_decay=layerwise_lr_decay,
                     label_smoothing=label_smoothing,
                     drop_path=drop_path,
                     seed=seed,
                     wandb_args={**wandb_default_args, **wandb_args},
                     use_final_batch_norm=final_batch_norm,
                     continue_from_checkpoint=checkpoint,
                     save_checkpoints=save_checkpoints,
                     **trainer_kwargs
                     )


def fewshot_finetuning(pretrained_model: Tuple[str, str], dataset: str, subset_fraction: float,
                       batch_size: int = 256, seed: int = 42, setup="swav", final_batch_norm=False,
                       save_checkpoints=False,
                       **trainer_kwargs):
    assert subset_fraction in [0.1, 0.01]
    model, arch = pretrained_model
    print(f"Few-shot finetuning for {model} {arch} on {dataset}, subset fraction: {subset_fraction}.")

    wandb_args = {
        "group": f"fewshot({subset_fraction})_finetuning_{dataset}",
        "name": f"{model}_{arch}",
        "mode": "online"
    }
    if final_batch_norm:
        wandb_args["tags"] = ["final_bn"]

    # Setting 1 SwAV -like
    if setup == "swav":
        if subset_fraction == 0.1:
            lr_backbone = 0.01
            lr_head = 0.2
            n_epochs = 20
        elif subset_fraction == 0.01:
            lr_backbone = 0.02
            lr_head = 5.0
            n_epochs = 20

        else:
            raise NotImplementedError(f"Subset fraction {subset_fraction} not implemented.")
    elif setup == "barlowtwins":
        lr_backbone = 0.5
        lr_head = 0.005
        n_epochs = 20
    elif setup == "byol1":
        lr_backbone = 0.01
        lr_head = 0.01
        n_epochs = 30
    elif setup == "byol2":
        lr_backbone = 0.1
        lr_head = 0.1
        n_epochs = 30

    check_val_every_n_epoch = 1
    optim = "sgd"
    scheduler = "cosine"
    warmup_epochs = 2
    unfreeze_after_epoch = None  # one epoch CLF only
    layerwise_lr_decay = 0.65 if arch == "vitb16" else None
    label_smoothing = 0.1 if arch == "vitb16" else 0.0
    drop_path = 0.2 if arch == "vitb16" else None

    train_classifier(pretrained_model=pretrained_model,
                     head="linear",
                     dataset=dataset,
                     num_epochs=n_epochs,
                     subset_fraction=subset_fraction,
                     freeze_model=False,
                     unfreeze_after_epoch=unfreeze_after_epoch,
                     batch_size=batch_size,
                     optim=optim,
                     scheduler=scheduler,
                     lr=lr_backbone,
                     lr_head=lr_head,
                     seed=seed,
                     check_val_every_n_epoch=check_val_every_n_epoch,
                     save_checkpoints=save_checkpoints,
                     wandb_args={**wandb_default_args, **wandb_args},
                     use_final_batch_norm=final_batch_norm,
                     layerwise_lr_decay=layerwise_lr_decay,
                     label_smoothing=label_smoothing,
                     drop_path=drop_path,
                     **trainer_kwargs
                     )
