"""Model registry driven by configs/models.yaml."""
from __future__ import annotations

import copy

import torch.nn as nn

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def list_models(models_cfg: dict) -> list[str]:
    return list(models_cfg["models"])


def resolve_model_key(models_cfg: dict, name: str) -> str:
    """Accepts the yaml key or the display name (case-insensitive)."""
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    for key, spec in models_cfg["models"].items():
        if norm(name) in (norm(key), norm(spec.get("display_name", ""))):
            return key
    raise KeyError(f"Unknown model '{name}'. Available: {list_models(models_cfg)}")


def get_model_spec(models_cfg: dict, key: str) -> dict:
    spec = copy.deepcopy(models_cfg.get("defaults", {}))
    spec.update(copy.deepcopy(models_cfg["models"][key]))
    spec["key"] = key
    spec.setdefault("display_name", key)
    spec.setdefault("paradigm", "")
    return spec


def build_model(spec: dict, num_classes: int, img_size: int = 224, pretrained: bool | None = None) -> nn.Module:
    pretrained = spec.get("pretrained", True) if pretrained is None else pretrained
    if spec["source"] == "timm":
        import timm

        kw = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=float(spec.get("drop_rate", 0.0)))
        if float(spec.get("drop_path_rate", 0.0)) > 0:
            kw["drop_path_rate"] = float(spec["drop_path_rate"])
        if any(t in spec["timm_name"] for t in ("vit_", "swin_", "deit")):
            kw["img_size"] = img_size
        return timm.create_model(spec["timm_name"], **kw)
    if spec["source"] == "vig":
        from .vig import build_pvig

        url = (spec.get("pretrained_urls") or {}).get(spec["arch"]) if pretrained else None
        if pretrained and not url:
            print(f"[ViG] no pretrained URL for {spec['arch']} - training from scratch")
        return build_pvig(spec["arch"], num_classes, k=int(spec.get("k", 9)),
                          drop_path_rate=float(spec.get("drop_path_rate", 0.0)),
                          dropout=float(spec.get("drop_rate", 0.0)), pretrained_url=url, img_size=img_size)
    raise ValueError(f"Unknown model source '{spec['source']}'")


def get_normalization(spec: dict, model: nn.Module):
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD))
