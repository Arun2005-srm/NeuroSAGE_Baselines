"""Classification metrics + model complexity (Params / FLOPs)."""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)


def classification_metrics(y_true, y_pred, probs=None, labels=None, class_names=None, average="macro") -> dict:
    """All numbers in %. `labels` = the class indices that exist in this evaluation set."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(set(y_true.tolist()))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=average, zero_division=0)
    pc_p, pc_r, pc_f, pc_n = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None,
                                                             zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # specificity per class: TN / (TN + FP) (predictions outside `labels` are FP for nobody)
    spec = []
    for i, c in enumerate(labels):
        tp, fp = cm[i, i], cm[:, i].sum() - cm[i, i]
        tn = len(y_true) - cm[i, :].sum() - fp
        spec.append(tn / max(tn + fp, 1))
    out = {
        "accuracy": 100 * accuracy_score(y_true, y_pred),
        "precision": 100 * p, "recall": 100 * r, "f1": 100 * f,
        "specificity": 100 * float(np.mean(spec)),
        "n_samples": int(len(y_true)),
    }
    if probs is not None and len(labels) > 1:
        try:
            pr = np.asarray(probs)[:, labels]
            pr = pr / np.clip(pr.sum(1, keepdims=True), 1e-12, None)
            out["auc"] = 100 * roc_auc_score(y_true, pr, multi_class="ovr", average="macro", labels=labels)
        except ValueError:
            pass
    names = class_names or [str(c) for c in labels]
    out["per_class"] = {names[i]: {"precision": 100 * pc_p[i], "recall": 100 * pc_r[i], "f1": 100 * pc_f[i],
                                   "specificity": 100 * spec[i], "support": int(pc_n[i])}
                        for i in range(len(labels))}
    out["confusion_matrix"] = {"labels": names, "matrix": cm.tolist()}
    return out


@torch.no_grad()
def model_complexity(model: torch.nn.Module, img_size: int = 224) -> dict:
    """Params (M) and FLOPs (G). FLOPs follow the usual paper convention (1 multiply-add = 1 FLOP),
    so ResNet-50 @224 ~= 4.1 G."""
    # learnable parameters only (ViG's fixed relative-position tables are stored as frozen Parameters)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    gflops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        m = copy.deepcopy(model).cpu().float().eval()
        counter = FlopCounterMode(display=False)
        with counter:
            m(torch.zeros(1, 3, img_size, img_size))
        gflops = counter.get_total_flops() / 2 / 1e9  # counter counts mul and add separately
        del m
    except Exception as e:  # noqa: BLE001
        print(f"[complexity] FLOP counting failed: {e}")
    return {"params_m": params, "flops_g": gflops}
