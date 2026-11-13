"""Download (Kaggle, via kagglehub) and index the datasets defined in configs/train.yaml.

    python download_data.py                         # all datasets
    python download_data.py --datasets figshare --force
"""
from __future__ import annotations

import argparse

from baselines.config import load_yaml
from baselines.data import prepare_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--force", action="store_true", help="rebuild index even if it exists")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    for name in a.datasets or list(cfg["data"]["datasets"]):
        prepare_dataset(name, cfg, force=a.force)


if __name__ == "__main__":
    main()
