"""Train + test one or more baselines.

Examples
    python train.py --list-models
    python train.py --model resnet50                                   # BRISC -> test BRISC + Figshare
    python train.py --model vig --mode kfold --k-folds 5
    python train.py --model all --train-dataset brisc2025 --test-datasets brisc2025 figshare
    python train.py --model swin_base --set training.lr=3e-5 --set training.batch_size=8
"""
from __future__ import annotations

import argparse
import copy
import traceback

from baselines.config import load_yaml, parse_set_args, set_by_path
from baselines.models import list_models, resolve_model_key
from baselines.runner import resolve_config, run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="NeuroSAGE baselines - train / test pipeline")
    p.add_argument("--config", default="configs/train.yaml", help="training yaml")
    p.add_argument("--models-config", default="configs/models.yaml", help="models yaml")
    p.add_argument("--model", help="model key(s), comma separated, or 'all'")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--train-dataset")
    p.add_argument("--test-datasets", nargs="+")
    p.add_argument("--mode", choices=["standard", "kfold"])
    p.add_argument("--k-folds", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--name", help="experiment name (default: auto)")
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any yaml value, e.g. --set training.lr=1e-4  (models.<key>.* edits models.yaml)")
    return p.parse_args()


def main():
    a = parse_args()
    train_cfg, models_cfg = load_yaml(a.config), load_yaml(a.models_config)
    sets = parse_set_args(a.set)
    for k, v in sets:  # model-level overrides go into models.yaml dict
        if k.startswith("models."):
            set_by_path(models_cfg, k, v)

    if a.list_models or not a.model:
        print(f"{'key':<18}{'name':<22}{'paradigm':<22}source")
        for key, s in models_cfg["models"].items():
            print(f"{key:<18}{s.get('display_name', ''):<22}{s.get('paradigm', ''):<22}"
                  f"{s.get('timm_name') or s.get('arch')}")
        print(f"\ndatasets: {list(train_cfg['data']['datasets'])}")
        return

    keys = list_models(models_cfg) if a.model.lower() == "all" else \
        [resolve_model_key(models_cfg, m) for m in a.model.split(",")]

    failed = []
    for key in keys:
        cfg, spec = resolve_config(train_cfg, models_cfg, key)
        cli = {
            "data.train_dataset": a.train_dataset, "data.test_datasets": a.test_datasets,
            "training.mode": a.mode, "training.k_folds": a.k_folds, "training.epochs": a.epochs,
            "training.batch_size": a.batch_size, "training.lr": a.lr, "experiment.seed": a.seed,
            "experiment.name": (a.name if len(keys) == 1 else (f"{a.name}__{key}" if a.name else None)),
        }
        for k, v in cli.items():
            if v is not None:
                set_by_path(cfg, k, v)
        for k, v in sets:
            if not k.startswith("models."):
                set_by_path(cfg, k, v)
        if a.no_pretrained:
            spec["pretrained"] = False
        try:
            run_experiment(copy.deepcopy(cfg), spec)
        except Exception:  # keep going with the next model when running several
            if len(keys) == 1:
                raise
            traceback.print_exc()
            failed.append(key)
    if failed:
        print(f"\nFAILED: {failed}")


if __name__ == "__main__":
    main()
