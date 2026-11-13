"""Experiment orchestration: standard / k-fold training, per-epoch logging, testing, summaries."""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import ImageListDataset, build_transforms, load_index, make_splits, subset
from .engine import (build_optimizer, build_scheduler, get_device, loss_from_logits, memory_gb, predict,
                     reset_memory, seed_everything, train_one_epoch)
from .metrics import classification_metrics, model_complexity
from .models import build_model, get_model_spec, get_normalization

SUMMARY_KEYS = ["accuracy", "precision", "recall", "f1", "specificity", "auc"]


def resolve_config(train_cfg: dict, models_cfg: dict, model_key: str) -> tuple[dict, dict]:
    """train.yaml + per-model training_overrides. CLI overrides are applied by the caller afterwards."""
    cfg = copy.deepcopy(train_cfg)
    spec = get_model_spec(models_cfg, model_key)
    cfg["training"].update(spec.get("training_overrides") or {})
    return cfg, spec


def _loader(df, classes, tf, cfg, shuffle, drop_last=False):
    labels = [classes.index(c) for c in df["label"]]
    nw = int(cfg["data"]["num_workers"])
    bs = int(cfg["training"]["batch_size"])
    return DataLoader(ImageListDataset(df["path"], labels, tf), batch_size=bs, shuffle=shuffle,
                      num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=nw > 0)


def evaluate_dataset(model, model_classes, df, cfg, tf, device, cross_domain: bool, dataset_name: str) -> dict:
    """Evaluate on an indexed dataframe. Handles label-space mismatch between domains."""
    unknown = sorted(set(df["label"]) - set(model_classes))
    if unknown:
        print(f"  [{dataset_name}] classes {unknown} unknown to the model -> {df['label'].isin(unknown).sum()} "
              f"samples excluded")
        df = df[~df["label"].isin(unknown)].reset_index(drop=True)
    present = sorted({model_classes.index(c) for c in df["label"]})
    mask = None
    if cross_domain and cfg["data"]["cross_domain"]["restrict_to_shared_classes"] and len(present) < len(model_classes):
        mask = torch.zeros(len(model_classes), dtype=torch.bool)
        mask[present] = True
    t0 = time.time()
    logits, targets = predict(model, _loader(df, model_classes, tf, cfg, False), device, cfg["training"], mask)
    elapsed = time.time() - t0
    probs = torch.softmax(logits, 1).numpy()
    m = classification_metrics(targets.numpy(), logits.argmax(1).numpy(), probs, labels=present,
                               class_names=[model_classes[i] for i in present], average=cfg["evaluation"]["average"])
    m["loss"] = loss_from_logits(logits, targets)
    m["inference_ms_per_image"] = 1000 * elapsed / max(len(df), 1)
    m["restricted_to_shared_classes"] = mask is not None
    return m


def _fmt_epoch(ep, n, tr_loss, tr_acc, vm, sec, mem):
    return (f"Ep[{ep}/{n}]: train acc {tr_acc:6.2f}% / train loss {tr_loss:.4f} | val acc {vm['accuracy']:6.2f}% | "
            f"val loss {vm['loss']:.4f} | precision {vm['precision']:6.2f}% | recall {vm['recall']:6.2f}% | "
            f"f1 {vm['f1']:6.2f}% | time {sec:6.1f}s | mem {mem:.2f}GB")


def _is_better(monitor, cur, best):
    if best is None:
        return True
    return cur < best if monitor == "val_loss" else cur > best


def run_experiment(cfg: dict, spec: dict) -> dict:
    exp, t, d = cfg["experiment"], cfg["training"], cfg["data"]
    seed_everything(int(exp["seed"]), bool(exp.get("deterministic", False)))
    device = get_device(exp.get("device", "auto"))
    train_name = d["train_dataset"]
    test_names = list(d["test_datasets"] or [train_name])
    mode, k = t["mode"], (int(t["k_folds"]) if t["mode"] == "kfold" else 1)
    if mode not in ("standard", "kfold"):
        raise ValueError("training.mode must be 'standard' or 'kfold'")
    name = exp.get("name") or f"{spec['key']}__train-{train_name}__{mode}" + (f"{k}" if mode == "kfold" else "")
    out_dir = Path(exp["output_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = list(d["datasets"][train_name]["classes"])
    df_train_ds = load_index(train_name, cfg)
    splits = make_splits(train_name, df_train_ds, cfg)
    test_dfs = {}
    for tn in test_names:
        test_dfs[tn] = subset(df_train_ds, splits["test"]) if tn == train_name else load_index(tn, cfg)

    print("=" * 110)
    print(f"Model      : {spec['display_name']}  ({spec.get('timm_name') or spec.get('arch')})")
    print(f"Train on   : {train_name} ({len(df_train_ds) - len(splits['test'])} pool imgs) | mode={mode}"
          + (f" k={k}" if mode == "kfold" else ""))
    print(f"Test on    : " + ", ".join(f"{tn} ({'in-domain test split' if tn == train_name else 'cross-domain, full'}"
                                        f": {len(test_dfs[tn])})" for tn in test_names))
    print(f"Device     : {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"Hyper-params: epochs={t['epochs']} bs={t['batch_size']}x{t.get('grad_accum_steps', 1)} lr={t['lr']} "
          f"opt={t['optimizer']} sched={t['scheduler']} amp={t['amp']}")
    print(f"Output     : {out_dir}")
    print("=" * 110)

    img_size = int(d["image_size"])
    complexity, folds_out = None, []
    for fi, fold in enumerate(splits["folds"], start=1):
        seed_everything(int(exp["seed"]) + fi - 1, bool(exp.get("deterministic", False)))
        if k > 1:
            print(f"\n----- Fold {fi}/{k} -----")
        model = build_model(spec, len(classes), img_size=img_size)
        if complexity is None:
            complexity = model_complexity(model, int(cfg["evaluation"]["flops_image_size"]))
            fl = complexity["flops_g"]
            print(f"Params: {complexity['params_m']:.2f} M | FLOPs: {fl:.2f} G" if fl is not None else
                  f"Params: {complexity['params_m']:.2f} M")
        mean, std = get_normalization(spec, model)
        tf_train, tf_eval = build_transforms(cfg, mean, std, True), build_transforms(cfg, mean, std, False)
        model = model.to(device)
        if t.get("channels_last"):
            model = model.to(memory_format=torch.channels_last)

        tr_df, va_df = subset(df_train_ds, fold["train"]), subset(df_train_ds, fold["val"])
        tr_loader = _loader(tr_df, classes, tf_train, cfg, True, drop_last=len(tr_df) > int(t["batch_size"]))
        va_loader = _loader(va_df, classes, tf_eval, cfg, False)
        print(f"train {len(tr_df)} | val {len(va_df)}")

        accum = max(1, int(t.get("grad_accum_steps", 1)))
        optimizer = build_optimizer(model, t)
        scheduler = build_scheduler(optimizer, t, -(-len(tr_loader) // accum))
        scaler = torch.amp.GradScaler("cuda", enabled=bool(t["amp"]) and device.type == "cuda")
        criterion = nn.CrossEntropyLoss(label_smoothing=float(t.get("label_smoothing", 0.0)))

        monitor = t.get("monitor", "val_f1")
        mkey = {"val_f1": "f1", "val_acc": "accuracy", "val_loss": "loss"}[monitor]
        best, best_state, best_epoch, best_val, history, bad = None, None, 0, None, [], 0
        n_ep = int(t["epochs"])
        for ep in range(1, n_ep + 1):
            reset_memory(device)
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, scheduler, scaler, device, t)
            logits, targets = predict(model, va_loader, device, t)
            vm = classification_metrics(targets.numpy(), logits.argmax(1).numpy(), None,
                                        labels=list(range(len(classes))), average=cfg["evaluation"]["average"])
            vm["loss"] = loss_from_logits(logits, targets)
            sec, mem = time.time() - t0, memory_gb(device)
            line = _fmt_epoch(ep, n_ep, tr_loss, tr_acc, vm, sec, mem)
            if _is_better(monitor, vm[mkey], best):
                best, best_epoch, bad = vm[mkey], ep, 0
                best_val = {k_: vm[k_] for k_ in ("accuracy", "precision", "recall", "f1", "loss")}
                best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
                line += "  *"
            else:
                bad += 1
            print(line, flush=True)
            history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": vm["loss"],
                            "val_acc": vm["accuracy"], "val_precision": vm["precision"], "val_recall": vm["recall"],
                            "val_f1": vm["f1"], "time_s": sec, "mem_gb": mem,
                            "lr": optimizer.param_groups[0]["lr"]})
            pat = t.get("early_stopping_patience")
            if pat and bad >= int(pat):
                print(f"early stopping (no {monitor} improvement for {pat} epochs)")
                break

        fold_dir = out_dir / (f"fold{fi}" if k > 1 else "run")
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        model.load_state_dict(best_state)
        if t.get("save_checkpoints", True):
            torch.save({"state_dict": best_state, "model_key": spec["key"], "spec": spec, "classes": classes,
                        "image_size": img_size, "mean": mean, "std": std, "epoch": best_epoch,
                        "train_dataset": train_name, "test_names": splits["test"]}, fold_dir / "best.pt")
        print(f"best epoch {best_epoch} ({monitor}={best:.4f}) -> testing")

        tests = {}
        for tn in test_names:
            m = evaluate_dataset(model, classes, test_dfs[tn], cfg, tf_eval, device, tn != train_name, tn)
            tests[tn] = m
            print(f"  TEST [{tn:>10}] acc {m['accuracy']:6.2f}% | precision {m['precision']:6.2f}% | recall "
                  f"{m['recall']:6.2f}% | f1 {m['f1']:6.2f}% | spec {m['specificity']:6.2f}%"
                  + (f" | auc {m['auc']:6.2f}%" if "auc" in m else ""))
        folds_out.append({"fold": fi, "best_epoch": best_epoch, "best_val": best_val, "history": history,
                          "test": tests, "n_train": len(tr_df), "n_val": len(va_df)})
        del model, optimizer, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summarise(cfg, spec, name, classes, complexity, folds_out, test_names, train_name)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    _print_final(summary)
    print(f"\nsaved -> {out_dir / 'summary.json'}")
    return summary


def _summarise(cfg, spec, name, classes, complexity, folds, test_names, train_name) -> dict:
    test_summary, drop = {}, {}
    for tn in test_names:
        per = {k: [f["test"][tn].get(k) for f in folds] for k in SUMMARY_KEYS}
        per = {k: v for k, v in per.items() if all(x is not None for x in v)}
        test_summary[tn] = {
            "domain": "in-domain" if tn == train_name else "cross-domain",
            "mean": {k: float(np.mean(v)) for k, v in per.items()},
            "std": {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in per.items()},
            "per_fold": per,
        }
    if train_name in test_names:
        ind = test_summary[train_name]["per_fold"]["accuracy"]
        for tn in test_names:
            if tn != train_name:
                diffs = [a - b for a, b in zip(ind, test_summary[tn]["per_fold"]["accuracy"])]
                drop[tn] = {"mean": float(np.mean(diffs)), "std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                            "per_fold": diffs}
    t = cfg["training"]
    return {
        "experiment": name, "created": datetime.now().isoformat(timespec="seconds"),
        "model_key": spec["key"], "display_name": spec["display_name"], "paradigm": spec.get("paradigm", ""),
        "train_dataset": train_name, "test_datasets": test_names, "classes": classes,
        "mode": t["mode"], "k_folds": len(folds), "seed": cfg["experiment"]["seed"],
        "complexity": complexity,
        "cv_val_accuracy": [f["best_val"]["accuracy"] for f in folds],
        "test_summary": test_summary, "performance_drop": drop,
        "folds": folds, "config": cfg, "model_spec": spec,
    }


def _ms(s, tn, k):
    m, sd = s["test_summary"][tn]["mean"].get(k), s["test_summary"][tn]["std"].get(k)
    if m is None:
        return "-"
    return f"{m:.2f} ± {sd:.2f}" if s["k_folds"] > 1 else f"{m:.2f}"


def _print_final(s: dict):
    cx = s["complexity"] or {}
    fl = f"{cx['flops_g']:.2f}" if cx.get("flops_g") is not None else "-"
    print("\n" + "=" * 110)
    print(f"FINAL RESULTS - {s['display_name']} | trained on {s['train_dataset']} | {s['mode']}"
          + (f" ({s['k_folds']} folds, mean ± std)" if s["k_folds"] > 1 else ""))
    print("=" * 110)
    hdr = f"{'Eval domain':<12}{'Type':<14}{'Acc (%)':>16}{'Prec (%)':>16}{'Rec (%)':>16}{'F1 (%)':>16}"
    print(hdr + f"{'Params(M)':>11}{'FLOPs(G)':>10}{'ΔAcc':>16}")
    for tn, ts in s["test_summary"].items():
        dr = s["performance_drop"].get(tn)
        drs = "-" if not dr else (f"{dr['mean']:.2f} ± {dr['std']:.2f}" if s["k_folds"] > 1 else f"{dr['mean']:.2f}")
        print(f"{tn:<12}{ts['domain']:<14}" + "".join(f"{_ms(s, tn, k):>16}" for k in ("accuracy", "precision", "recall", "f1"))
              + f"{cx.get('params_m', 0):>11.2f}{fl:>10}{drs:>16}")
