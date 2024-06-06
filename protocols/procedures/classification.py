from typing import Tuple
import warnings

import torchmetrics.functional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy
import wandb

from models import get_model
from datasets import DataModule
from protocols.procedures.util.lars import LARS
from config import config
from protocols.procedures.util.cosine_warmup import CosineWarmupScheduler
from protocols.procedures.util.vit_optim import LayerDecayValueAssigner, get_parameter_groups

device = config["device"]
world_size = max(1, config["gpu_count"])

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
torch.set_float32_matmul_precision("high")  # to utilize A5000 tensor cores


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class MLPClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000, hidden_size=1000):
        super(MLPClassifier, self).__init__()
        self.num_labels = num_labels
        self.fc1 = nn.Linear(dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels)
        self.relu = nn.ReLU()
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.weight.data.normal_(mean=0.0, std=0.01)
            fc.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class LitClassifier(pl.LightningModule):

    def __init__(self, pretrained_model: nn.Module, clf_head: nn.Module, n_classes: int, batch_size: int,
                 n_epochs: int, optim: str = "sgd", scheduler="cosine", warmup_epochs: int = 0, lr: float = 0.001,
                 lr_head: float = None, unfreeze_after_epoch: int = None, layerwise_lr_decay=None, label_smoothing=0.0
                 ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.clf_head = clf_head
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optim = optim
        self.scheduler = scheduler
        if not scheduler == "warmup_cosine" and warmup_epochs:
            warnings.warn("warmup_epochs is ingored unless scheduler is warmup_cosine")
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.layerwise_lr_decay = layerwise_lr_decay
        self.smoothing = label_smoothing
        self.lr_head = lr_head or lr
        self.unfreeze_after_epoch = unfreeze_after_epoch

        self.save_hyperparameters()

        self.highest_val_acc_top1 = torch.tensor(0.0)
        self.highest_val_acc_top5 = torch.tensor(0.0)

    def forward(self, x):
        emb = self.pretrained_model(x)
        pred = self.clf_head(emb)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = nn.functional.cross_entropy(out, y, label_smoothing=self.smoothing)
        acc_k1 = torchmetrics.functional.accuracy(out, y, task="multiclass", num_classes=self.n_classes, top_k=1)
        acc_k5 = torchmetrics.functional.accuracy(out, y, task="multiclass", num_classes=self.n_classes, top_k=5)
        self.log("train_loss", loss)
        self.log("train_accuracy", acc_k1, prog_bar=True)
        self.log("train_accuracy_top5", acc_k5)
        return loss

    def training_epoch_end(self, outputs) -> None:
        if self.unfreeze_after_epoch is not None and self.trainer.current_epoch == self.unfreeze_after_epoch:
            print(f"Reached epoch {self.trainer.current_epoch}, Unfreezing whole model.")
            self.unfreeze()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = nn.functional.cross_entropy(out, y.ravel())
        acc_k1 = torchmetrics.functional.accuracy(out, y, task="multiclass", num_classes=self.n_classes, top_k=1)
        acc_k5 = torchmetrics.functional.accuracy(out, y, task="multiclass", num_classes=self.n_classes, top_k=5)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_accuracy", acc_k1, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_accuracy_top5", acc_k5, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):

        # linear scaling rule for LR
        abs_lr = self.lr * world_size * self.batch_size / 256
        abs_head_lr = self.lr_head * world_size * self.batch_size / 256

        if self.layerwise_lr_decay:
            num_layers = 12  # TODO: get this from model, only accounts for ViT/B-16
            assigner = LayerDecayValueAssigner(
                list(self.layerwise_lr_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))[:-1])

            _params = get_parameter_groups(self.pretrained_model,
                                           weight_decay=0.00,
                                           get_num_layer=assigner.get_layer_id,
                                           get_layer_scale=assigner.get_scale
                                           )

            for p in _params:
                p["lr"] = p["lr_scale"] * abs_lr
                del p["lr_scale"]
                del p["weight_decay"]

        else:
            _params = [
                {'params': self.pretrained_model.parameters(), 'lr': abs_lr},
            ]

        _params.append({'params': self.clf_head.parameters(), 'lr': abs_head_lr})

        print(f"Learning rates: {[_p['lr'] for _p in _params]}")

        if self.optim == "lars":
            optimizer = LARS(_params, lr=abs_lr, weight_decay=0.)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(_params, lr=abs_lr, eps=1e-8)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(_params, lr=abs_lr, momentum=0.9, weight_decay=0.)
        else:
            raise NotImplementedError(f"Optimizer {self.optim} not implemented for now.")

        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs)
        elif self.scheduler == "warmup_cosine":
            scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.n_epochs)
        elif self.scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             [int(0.6 * self.n_epochs), int(0.8 * self.n_epochs)],
                                                             gamma=0.2)
        else:
            raise NotImplementedError(f"Scheduler {self.scheduler} not implemented for now.")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_classifier(
        pretrained_model: Tuple[str, str],
        head: str,
        dataset: str,
        subset_fraction: float = 1.0,
        freeze_model=True,
        num_epochs=50,
        wandb_args=None,
        batch_size=64,
        optim="sgd",
        scheduler="cosine",
        warmup_epochs=None,
        lr=0.001,
        layerwise_lr_decay=None,
        lr_head=None,
        label_smoothing=0.0,
        drop_path=None,
        unfreeze_after_epoch=None,
        seed=42,
        use_final_batch_norm=False,
        accumulate_grad_batches=1,
        continue_from_checkpoint=None,
        save_checkpoints=True,
        **trainer_kwargs
):
    method, arch = pretrained_model
    if drop_path:
        model = get_model(method, arch, freeze=freeze_model, drop_path_rate=drop_path)  # TODO unclean
    else:
        model = get_model(method, arch, freeze=freeze_model)
    with torch.no_grad():
        in_shape = model(torch.rand(1, 3, 224, 224).to(device)).shape[-1]
        print(f"Embedding shape: {in_shape}")

    batch_size_per_gpu = batch_size // world_size
    if accumulate_grad_batches == 1:
        accumulate_grad_batches = None

    if layerwise_lr_decay is not None:
        assert arch == "vitb16", "Layerwise LR decay only implemented for ViT-B/16"

    # Get data loaders
    data_module = DataModule(dataset, batch_size=batch_size_per_gpu, subset_fraction=subset_fraction)
    n_classes = data_module.num_classes

    print(f"N classes: {n_classes}")

    # initialize head
    if head == "linear":
        clf = LinearClassifier(in_shape, n_classes)
    elif head == "mlp":
        clf = MLPClassifier(in_shape, n_classes)
    else:
        raise NotImplementedError(f"Head {head} not implemented for now.")

    if use_final_batch_norm:
        clf = nn.Sequential(nn.BatchNorm1d(in_shape), clf)

    if continue_from_checkpoint is not None:
        print("loading model from checkpoint")
        checkpoint_reference = f"{wandb_args['entity']}/{wandb_args['project']}/model-{continue_from_checkpoint}:latest"

        run = wandb.init(mode="disabled")
        artifact = run.use_artifact(checkpoint_reference, type='model')
        artifact_dir = artifact.download()
        ckpt_path = artifact_dir + "/model.ckpt"
        wandb_args["name"] = f"{wandb_args['name']}-continued({continue_from_checkpoint})"
        wandb.finish()
    else:
        ckpt_path = None

    logger = WandbLogger(**wandb_args)

    full_model = LitClassifier(model, clf, n_classes, batch_size_per_gpu, num_epochs, optim=optim, scheduler=scheduler,
                               warmup_epochs=warmup_epochs, lr=lr, lr_head=lr_head, label_smoothing=label_smoothing,
                               layerwise_lr_decay=layerwise_lr_decay, unfreeze_after_epoch=unfreeze_after_epoch,
                               )

    if world_size == 1:
        strategy = SingleDeviceStrategy(device="cuda:0" if device == "cuda" else device)
    else:
        strategy = DDPStrategy(find_unused_parameters=False)

    # callbacks
    callbacks = [LearningRateMonitor(logging_interval='epoch')]
    if save_checkpoints:
        callbacks += [ModelCheckpoint(save_last=True, every_n_epochs=1, filename="last", save_top_k=1)]

    trainer = pl.Trainer(logger=logger, max_epochs=num_epochs, devices=world_size,
                         accelerator="cuda" if config["device"].startswith("cuda") else "cpu",
                         log_every_n_steps=min(len(data_module.train_set) // batch_size, 50),
                         # enable logging for very small data sets
                         strategy=strategy, callbacks=callbacks,
                         gradient_clip_val=0.5, gradient_clip_algorithm="norm",
                         accumulate_grad_batches=accumulate_grad_batches,
                         # resume_from_checkpoint=ckpt_path,
                         **trainer_kwargs)

    pl.seed_everything(seed, workers=True)
    trainer.fit(model=full_model, datamodule=data_module,
                ckpt_path=ckpt_path)
    logger.experiment.finish()


if __name__ == '__main__':
    train_classifier(("dino", "resnet50"), "linear", "dummy", subset_fraction=0.1, freeze_model=True, num_epochs=50,
                     wandb_args={
                         "project": "test",
                         "mode": "offline",
                     }
                     )
