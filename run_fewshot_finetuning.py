import argparse
from models import AVAILABLE_MODELS
from protocols import fewshot_finetuning

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="*", type=str, default=None)
parser.add_argument("fraction", type=float)
parser.add_argument("--dataset", type=str, default="imagenet")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--final_batch_norm", action="store_true", default=False)
parser.add_argument("--batch_accum", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--setup", type=str, default="byol2")
parser.add_argument("--no_ckpt_log", action="store_true", default=False)


if __name__ == "__main__":

    args = parser.parse_args()
    if args.model:
        assert len(args.model) == 2
        models_to_run = [tuple(args.model)]
    else:
        models_to_run = AVAILABLE_MODELS

    for model in models_to_run:
        fewshot_finetuning(model, args.dataset, args.fraction, batch_size=args.batch_size, seed=args.seed,
                           setup=args.setup,
                           accumulate_grad_batches=args.batch_accum,
                           final_batch_norm=args.final_batch_norm,
                           save_checkpoints=not args.no_ckpt_log
                           )
