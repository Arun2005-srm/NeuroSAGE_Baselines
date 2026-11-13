"""Re-test a saved checkpoint on any dataset (in-domain test split or zero-shot cross-domain).

    python evaluate.py --checkpoint results/resnet50__train-brisc2025__standard/run/best.pt --datasets figshare
"""
from __future__ import annotations

import argparse
import json

import torch

from baselines.config import load_yaml
from baselines.data import build_transforms, load_index, make_splits, subset
from baselines.engine import get_device
from baselines.metrics import model_complexity
from baselines.models import build_model
from baselines.runner import evaluate_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--out", help="optional json output path")
    a = p.parse_args()

    cfg = load_yaml(a.config)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    spec, classes = ck["spec"], ck["classes"]
    cfg["data"]["image_size"] = ck["image_size"]
    model = build_model(spec, len(classes), img_size=ck["image_size"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    cx = model_complexity(model, ck["image_size"])
    device = get_device(cfg["experiment"].get("device", "auto"))
    model = model.to(device)
    tf = build_transforms(cfg, ck["mean"], ck["std"], False)
    print(f"{spec['display_name']} trained on {ck['train_dataset']} | Params {cx['params_m']:.2f} M | "
          f"FLOPs {cx['flops_g']:.2f} G")

    results = {}
    for ds in a.datasets:
        df = load_index(ds, cfg)
        cross = ds != ck["train_dataset"]
        if not cross:  # the exact held-out test list stored with the checkpoint
            df = subset(df, ck["test_names"] if "test_names" in ck else make_splits(ds, df, cfg)["test"])
        m = evaluate_dataset(model, classes, df, cfg, tf, device, cross, ds)
        results[ds] = m
        print(f"[{ds}] ({'cross-domain' if cross else 'in-domain test'}) acc {m['accuracy']:.2f} | precision "
              f"{m['precision']:.2f} | recall {m['recall']:.2f} | f1 {m['f1']:.2f} | spec {m['specificity']:.2f}")
        print("  confusion matrix", m["confusion_matrix"]["labels"])
        for row in m["confusion_matrix"]["matrix"]:
            print("   ", row)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"complexity": cx, "results": results}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
