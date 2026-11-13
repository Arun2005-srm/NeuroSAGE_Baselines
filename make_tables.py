"""Collect results/*/summary.json into the paper tables (sheets 4.6.1, 4.6.2, 5.2) -> results/tables.xlsx

    python make_tables.py                                   # 4.6.1 + 5.2 from whatever has been run
    python make_tables.py --reference neurosage_folds.json  # + 4.6.2 significance vs. a reference model
    python make_tables.py --reference vig                   # (any baseline key can act as the reference)

Reference file format (for NeuroSAGE-Net trained in its own repo on the SAME cached folds):
    {"name": "NeuroSAGE-Net", "fold_scores": [97.1, 96.8, 97.4, 97.0, 96.9]}
or simply the summary.json produced by this pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baselines.config import load_yaml


def load_summaries(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("*/summary.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            s["_path"] = str(p)
            out.append(s)
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip {p}: {e}")
    return out


def pick(summaries, model_key, train, mode_pref):
    """Most recent run of model trained on `train`; mode 'auto' prefers kfold over standard."""
    c = [s for s in summaries if s["model_key"] == model_key and s["train_dataset"] == train]
    if mode_pref != "auto":
        c = [s for s in c if s["mode"] == mode_pref]
    if not c:
        return None
    c.sort(key=lambda s: (s["mode"] == "kfold" if mode_pref == "auto" else 0, s.get("created", "")))
    return c[-1]


def fmt(mean, std, k):
    if mean is None:
        return ""
    return f"{mean:.2f} ± {std:.2f}" if k > 1 else f"{mean:.2f}"


def table_461(summaries, models_cfg, train, mode):
    rows = []
    for key, m in models_cfg["models"].items():
        s = pick(summaries, key, train, mode)
        r = {"Model Paradigm": m.get("paradigm", ""), "Model Architecture": m.get("display_name", key)}
        if s and train in s["test_summary"]:
            ts, k = s["test_summary"][train], s["k_folds"]
            for col, met in [("Accuracy (%)", "accuracy"), ("Precision (%)", "precision"),
                             ("Recall / Sensitivity (%)", "recall"), ("F1-Score (%)", "f1"),
                             ("Specificity (%)", "specificity"), ("AUC (%)", "auc")]:
                r[col] = fmt(ts["mean"].get(met), ts["std"].get(met), k)
            cx = s.get("complexity") or {}
            r["Params (M)"] = round(cx["params_m"], 2) if cx.get("params_m") is not None else ""
            r["FLOPs (G)"] = round(cx["flops_g"], 2) if cx.get("flops_g") is not None else ""
            r["Protocol"] = f"{s['mode']}" + (f" ({k}-fold)" if k > 1 else "")
        rows.append(r)
    return pd.DataFrame(rows)


def table_52(summaries, models_cfg, train, target, mode, data_cfg):
    rows = []
    dn = lambda n: data_cfg["datasets"].get(n, {}).get("display_name", n)
    for key, m in models_cfg["models"].items():
        s = pick(summaries, key, train, mode)
        r = {"Model Architecture": m.get("display_name", key), "Training Domain": dn(train),
             "Evaluation Domain": dn(target)}
        if s and target in s["test_summary"]:
            ts, k = s["test_summary"][target], s["k_folds"]
            for col, met in [("Accuracy (%)", "accuracy"), ("Precision (%)", "precision"),
                             ("Recall / Sens. (%)", "recall"), ("F1-Score (%)", "f1")]:
                r[col] = fmt(ts["mean"].get(met), ts["std"].get(met), k)
            dr = s["performance_drop"].get(target)
            r["Performance Drop (ΔAcc)"] = fmt(dr["mean"], dr["std"], k) if dr else ""
        rows.append(r)
    return pd.DataFrame(rows)


def fold_scores(s, train, source, metric):
    if source == "cv":
        return [float(x) for x in s["cv_val_accuracy"]]
    return [float(x) for x in s["test_summary"][train]["per_fold"][metric]]


def sig_label(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "*** (p < 0.001)" if p < 0.001 else "** (p < 0.01)" if p < 0.01 else "* (p < 0.05)" if p < 0.05 else "n.s."


def table_462(summaries, models_cfg, train, ref_arg, source, metric):
    # ---- reference scores
    ref_name, ref_scores = None, None
    if Path(ref_arg).exists():
        j = json.loads(Path(ref_arg).read_text(encoding="utf-8"))
        if "fold_scores" in j:
            ref_name, ref_scores = j.get("name", "Reference"), [float(x) for x in j["fold_scores"]]
        else:
            ref_name, ref_scores = j["display_name"], fold_scores(j, train, source, metric)
            ref_key = j["model_key"]
    else:
        s = pick(summaries, ref_arg, train, "kfold")
        if not s:
            raise SystemExit(f"no k-fold run found for reference '{ref_arg}' trained on {train}")
        ref_name, ref_scores, ref_key = s["display_name"], fold_scores(s, train, source, metric), s["model_key"]
    k = len(ref_scores)
    if k < 2:
        raise SystemExit("significance tests need k-fold results (k >= 2) for the reference")

    rows = []
    # sheet order: ViG, Swin, ViT, ConvNeXt, EfficientNet, ResNet
    for key in reversed(list(models_cfg["models"])):
        m = models_cfg["models"][key]
        if m.get("display_name") == ref_name:
            continue
        s = pick(summaries, key, train, "kfold")
        r = {"Baseline Model Pair": f"{ref_name} vs. {m.get('display_name', key)}"}
        if s and s["k_folds"] == k:
            b = fold_scores(s, train, source, metric)
            a, b = np.asarray(ref_scores), np.asarray(b)
            t_stat, t_p = stats.ttest_rel(a, b)
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
            except ValueError:  # all differences zero
                w_stat, w_p = float("nan"), float("nan")
            r.update({"Mean Accuracy Difference (Δ%)": round(float(np.mean(a - b)), 3),
                      "Paired t-statistic": round(float(t_stat), 4), "Degrees of Freedom (df)": k - 1,
                      "Wilcoxon W-statistic": round(float(w_stat), 4),
                      "p-value": float(f"{t_p:.3g}"), "Wilcoxon p-value": float(f"{w_p:.3g}"),
                      "Significance Level": sig_label(float(t_p))})
        elif s:
            r["Significance Level"] = f"fold count mismatch ({s['k_folds']} vs {k})"
        rows.append(r)
    if k <= 5:
        print(f"NOTE: with k={k} folds the smallest achievable two-sided Wilcoxon p-value is {2 / 2 ** k:.4f}; "
              "use k=10 if you need Wilcoxon significance at 0.05.")
    df = pd.DataFrame(rows)
    if "Degrees of Freedom (df)" in df:
        df["Degrees of Freedom (df)"] = df["Degrees of Freedom (df)"].map(lambda v: "" if pd.isna(v) else int(v))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--train-dataset", default="brisc2025")
    p.add_argument("--target", default="figshare", help="cross-domain evaluation dataset for table 5.2")
    p.add_argument("--mode", default="auto", choices=["auto", "kfold", "standard"])
    p.add_argument("--reference", help="summary.json / fold-score json / model key for table 4.6.2")
    p.add_argument("--stat-source", default="test", choices=["test", "cv"],
                   help="per-fold scores for 4.6.2: 'test' = each fold model on the held-out test set, "
                        "'cv' = best validation-fold accuracy")
    p.add_argument("--stat-metric", default="accuracy")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    models_cfg, train_cfg = load_yaml(a.models_config), load_yaml(a.config)
    summaries = load_summaries(Path(a.results))
    print(f"found {len(summaries)} runs in {a.results}/")
    tables = {
        "4.6.1": table_461(summaries, models_cfg, a.train_dataset, a.mode),
        "5.2": table_52(summaries, models_cfg, a.train_dataset, a.target, a.mode, train_cfg["data"]),
    }
    if a.reference:
        tables["4.6.2"] = table_462(summaries, models_cfg, a.train_dataset, a.reference, a.stat_source, a.stat_metric)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, df in tables.items():
        print(f"\n===== Table {name} =====")
        print(df.fillna("").to_string(index=False))
    out = Path(a.out or Path(a.results) / "tables.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        for name in ("4.6.1", "4.6.2", "5.2"):
            if name in tables:
                tables[name].to_excel(xw, sheet_name=name, index=False)
    for name, df in tables.items():
        df.to_csv(out.with_name(f"table_{name}.csv"), index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
