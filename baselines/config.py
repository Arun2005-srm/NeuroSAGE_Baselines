"""YAML config loading, merging and CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge `upd` into `base` (in place) and return base."""
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def parse_set_args(items: list[str] | None) -> list[tuple[str, object]]:
    """['training.lr=1e-4', 'data.test_datasets=[figshare]'] -> [(key, parsed_value), ...]"""
    out = []
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects key=value, got '{it}'")
        k, v = it.split("=", 1)
        out.append((k.strip(), yaml.safe_load(v)))
    return out


def as_float(x):
    return None if x is None else float(x)
