from typing import Tuple
import warnings

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import numpy as np

from models import get_model
from datasets import DataModule
from datasets.precalculated import PrecalculatedEmbeddings
from config import config

device = config["device"]


@torch.no_grad()
def get_embeddings(model, dataloader):
    embeddings = None
    labels = None
    for x, y in tqdm(dataloader, desc="Prepare data"):
        out = model(x.to(device)).cpu().detach()
        label = y.cpu().detach()
        if embeddings is None:
            embeddings = out
        else:
            embeddings = torch.cat((embeddings, out), dim=0)
        if labels is None:
            labels = label
        else:
            labels = torch.cat((labels, label), dim=0)
    return embeddings.numpy(), labels.numpy()


def fit_classifier(
    pretrained_model: Tuple[str, str],
    shallow_clf: str,
    dataset: str,
    subset_fraction: float = 1.0,
    batch_size=64,
    use_precalculated_embeddings=True,
    normalize_embeddings=False,
    return_top5=False,
    **clf_kwargs,
):

    method, arch = pretrained_model

    if use_precalculated_embeddings and PrecalculatedEmbeddings.exists(method, arch, dataset, "train"):
        train_emb = PrecalculatedEmbeddings(method, arch, dataset, "train")
        X_train = train_emb.data_numpy
        y_train = train_emb.labels_numpy
    else:
        model = get_model(method, arch, freeze=True)
        dm = DataModule(dataset, batch_size=batch_size, subset_fraction=subset_fraction, shuffle_train=False,
                        augment_train=False)
        X_train, y_train = get_embeddings(model, dm.train_dataloader())

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0.0] = 1.0  # avoid division by zero
    if normalize_embeddings:
        X_train = (X_train - train_mean) / train_std

    if np.isnan(X_train).any():
        na_rows = np.isnan(X_train).any(axis=1)
        warnings.warn(f"Dropping {na_rows.sum()} rows with NaNs.")
        X_train = X_train[~na_rows]

    if shallow_clf == "linear":
        clf = LogisticRegression(verbose=1, solver="lbfgs", max_iter=100, **clf_kwargs)
    elif shallow_clf == "knn":
        clf = KNeighborsClassifier(n_jobs=-1, **clf_kwargs)
    else:
        raise NotImplementedError(f"Shallow classifier {shallow_clf} not implemented.")

    clf.fit(X_train, y_train)

    if use_precalculated_embeddings and PrecalculatedEmbeddings.exists(method, arch, dataset, "val"):
        val_emb = PrecalculatedEmbeddings(method, arch, dataset, "val")
        X_val = val_emb.data_numpy
        y_val = val_emb.labels_numpy
    else:
        raise NotImplementedError("Validation set not precalculated.")

    val_mean = X_val.mean(axis=0)
    val_std = X_val.std(axis=0)
    val_std[val_std == 0.0] = 1.0  # avoid division by zero

    if normalize_embeddings == "train":
        X_val = (X_val - train_mean) / train_std
    elif normalize_embeddings == "val":
        X_val = (X_val - val_mean) / val_std


    if np.isnan(X_val).any():
        na_rows = np.isnan(X_val).any(axis=1)
        warnings.warn(f"Dropping {na_rows.sum()} rows with NaNs.")
        X_val = X_val[~na_rows]

    accuracy = clf.score(X_val, y_val)

    if return_top5:
        y_score = clf.predict_proba(X_val)
        top5_accuracy = top_k_accuracy_score(y_val, y_score, k=5)
        return accuracy, top5_accuracy

    return accuracy


if __name__ == '__main__':
    fit_classifier(("dino", "vitb16"), "knn", "dummy", use_precalculated_embeddings=False)
