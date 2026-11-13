"""Training / inference loops, optimiser + scheduler, timing and memory helpers."""
from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def reset_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_gb(device) -> float:
    """Peak GPU memory allocated since last reset (GB); process RSS on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 3
    import psutil

    return psutil.Process().memory_info().rss / 1024 ** 3


def build_optimizer(model: nn.Module, t: dict):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") or "pos_embed" in n else decay).append(p)
    groups = [{"params": decay, "weight_decay": float(t["weight_decay"])}, {"params": no_decay, "weight_decay": 0.0}]
    lr = float(t["lr"])
    if t["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if t["optimizer"].lower() == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(t.get("momentum", 0.9)), nesterov=True)
    raise ValueError(f"Unknown optimizer {t['optimizer']}")


def build_scheduler(optimizer, t: dict, steps_per_epoch: int):
    """Per-optimiser-step LambdaLR: linear warm-up then cosine / step decay."""
    total = max(1, int(t["epochs"]) * steps_per_epoch)
    warm = int(t.get("warmup_epochs", 0)) * steps_per_epoch
    base = float(t["lr"])
    floor = float(t.get("min_lr", 0.0)) / base
    kind = (t.get("scheduler") or "none").lower()

    def f(step):
        if step < warm:
            return (step + 1) / warm
        if kind == "cosine":
            prog = (step - warm) / max(1, total - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        if kind == "step":
            return float(t.get("gamma", 0.1)) ** ((step // steps_per_epoch) // int(t.get("step_size", 7)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, t: dict):
    model.train()
    accum = max(1, int(t.get("grad_accum_steps", 1)))
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    cl = bool(t.get("channels_last", False))
    loss_sum, correct, n = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = criterion(logits.float(), y)
        scaler.scale(loss / accum).backward()
        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if t.get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, 100.0 * correct / n


@torch.no_grad()
def predict(model, loader, device, t: dict, class_mask: torch.Tensor | None = None):
    """Returns (logits [N,C] float cpu, targets [N]). `class_mask` (bool [C]) disables classes."""
    model.eval()
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    cl = bool(t.get("channels_last", False))
    outs, tgts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        outs.append(logits.float().cpu())
        tgts.append(y)
    logits, targets = torch.cat(outs), torch.cat(tgts)
    if class_mask is not None:
        logits = logits.masked_fill(~class_mask.view(1, -1), -1e4)
    return logits, targets


def loss_from_logits(logits, targets) -> float:
    return F.cross_entropy(logits, targets).item()
