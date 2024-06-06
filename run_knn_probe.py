import argparse

from tqdm import tqdm
import wandb
import pandas as pd

from models import AVAILABLE_MODELS
from protocols import knn_probe

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--n_neighbors", type=int, default=20)
parser.add_argument("--weights", type=str, default="distance")
parser.add_argument("--algorithm", type=str, default="brute")
parser.add_argument("--norm", action="store_true", default=False)
parser.add_argument("--norm_val", action="store_true", default=False)


def run_all_knn_probes(dataset: str, n_neighbors: int = 20, weights: str = "distance", algorithm: str = "ball_tree",
                       norm: bool = False, norm_val: bool = False):
    assert not (norm and norm_val)
    if norm:
        _norm = "train"
    elif norm_val:
        _norm = "val"
    else:
        _norm = False
    wandb.init(project="ssl_eval_metrics", entity="vision-lab", group="knn_probes", name=f"{dataset}", mode="online")
    wandb.config.update(locals())
    results_table = wandb.Table(columns=["model", "arch", "dataset", "metric", "accuracy", "top5_accuracy"])
    for model in tqdm(AVAILABLE_MODELS, desc=f"Running KNN probes"):
        for metric in ["euclidean"]:
            acc, top5acc = knn_probe(model, dataset, metric=metric, weights=weights, n_neighbors=n_neighbors,
                            algorithm=algorithm, normalize_embeddings=_norm)
            results_table.add_data(model[0], model[1], dataset, metric, acc, top5acc)
    wandb.log({"results": results_table})
    df = pd.DataFrame(data=results_table.data, columns=results_table.columns)
    df.to_csv(f"results/knn_{dataset}{'_norm' if norm else ''}.csv", index=False)
    wandb.finish()


if __name__ == '__main__':
    run_all_knn_probes(**vars(parser.parse_args()))
