import argparse

from models import AVAILABLE_MODELS
from protocols import linear_probe

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="*", type=str, default=None)
parser.add_argument("--head", type=str, default="linear")
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--final_batch_norm", action="store_true", default=False)
parser.add_argument("--batch_accum", type=int, default=1)
parser.add_argument("--no_ckpt_log", action="store_true", default=False)


if __name__ == "__main__":

    args = parser.parse_args()

    if args.model:
        assert len(args.model) == 2
        models_to_run = [tuple(args.model)]
    else:
        models_to_run = AVAILABLE_MODELS

    for model in models_to_run:
        linear_probe(model,
                     head=args.head,
                     dataset=args.dataset,
                     lr=args.lr,
                     batch_size=args.batch_size,
                     n_epochs=args.n_epochs,
                     seed=args.seed,
                     final_batch_norm=args.final_batch_norm,
                     accumulate_grad_batches=args.batch_accum,
                     save_checkpoints=not args.no_ckpt_log
                     )
