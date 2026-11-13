# NeuroSAGE Baselines

Config-driven train/test pipeline for the **baseline models** compared against NeuroSAGE-Net in
Tables **4.6.1** (SOTA comparison), **4.6.2** (statistical significance) and **5.2** (cross-dataset zero-shot).

| Key | Model | Paradigm | Implementation |
|---|---|---|---|
| `resnet50` | ResNet-50 | CNN | timm `resnet50.tv_in1k` |
| `efficientnet_b0` | EfficientNet-B0 | CNN | timm `efficientnet_b0.ra_in1k` |
| `convnext_base` | ConvNeXt-Base | CNN | timm `convnext_base.fb_in1k` |
| `vit_b16` | Standard ViT-B/16 | Vision Transformer | timm `vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| `swin_base` | Swin-Base | Vision Transformer | timm `swin_base_patch4_window7_224.ms_in22k_ft_in1k` |
| `vig` | Vision GNN (ViG) | Graph Neural Net | Pyramid ViG-S, own re-implementation, official ImageNet weights load with 0 missing keys |

Datasets (downloaded automatically from Kaggle with `kagglehub`):

| Key | Kaggle | Classes | Split |
|---|---|---|---|
| `brisc2025` | `briscdataset/brisc2025` (`classification_task`) | glioma, meningioma, no_tumor, pituitary | official train / test |
| `figshare` | `ashkhagan/figshare-brain-tumor-dataset` (Cheng et al. `.mat`) | glioma, meningioma, pituitary | none: **patient-wise** stratified 80/20 hold-out |

## Setup

```bash
pip install -r requirements.txt
python download_data.py          # optional: train.py prepares missing datasets automatically
```

## Configuration: two central YAML files

* **`configs/train.yaml`**: everything about training: datasets, train/test domains, mode (`standard` | `kfold`),
  `k_folds`, `epochs` (20), batch size, optimiser, LR schedule, AMP, augmentation, checkpoint metric, etc.
* **`configs/models.yaml`**: the model registry: timm id / ViG variant, pretrained weights, drop-path, and
  per-model `training_overrides` (e.g. LR, batch size, gradient accumulation for the large models).

Precedence: `train.yaml` < `models.yaml: training_overrides` < command line (`--epochs`, `--lr`, `--set a.b=c`, …).

## Running

```bash
python train.py --list-models

# train on BRISC, test on the BRISC test split (in-domain) AND all of Figshare (zero-shot)
python train.py --model resnet50

# k-fold instead of a single train/val split
python train.py --model vig --mode kfold --k-folds 5

# several / all models, other direction, any override
python train.py --model resnet50,efficientnet_b0
python train.py --model all --mode kfold
python train.py --model swin_base --train-dataset figshare --test-datasets figshare brisc2025
python train.py --model convnext_base --set training.batch_size=8 --set training.grad_accum_steps=4
python train.py --model vig --set models.vig.arch=pvig_b
```

Every epoch prints:

```
Ep[3/20]: train acc  91.84% / train loss 0.4127 | val acc  93.10% | val loss 0.2214 | precision  93.02% | recall  92.87% | f1  92.91% | time   48.3s | mem 2.84GB  *
```
(`*` = new best checkpoint; `mem` = peak GPU memory of the epoch, or process RAM on CPU.)

After training, the best checkpoint is tested on every `test_datasets` entry and a summary is printed with
Accuracy, Precision, Recall/Sensitivity, F1 (macro), Specificity, AUC, Params (M), FLOPs (G) and the cross-domain
performance drop ΔAcc. Confusion matrices and per-class scores go into `results/<run>/summary.json`.

Re-test any checkpoint later:

```bash
python evaluate.py --checkpoint results/resnet50__train-brisc2025__standard/run/best.pt --datasets brisc2025 figshare
```

## Producing the tables

```bash
python make_tables.py                                    # 4.6.1 + 5.2  -> results/tables.xlsx (+ CSVs)
python make_tables.py --reference neurosage_folds.json   # + 4.6.2
```

* **4.6.1**: in-domain test metrics per model (mean ± std over folds for k-fold runs) + Params + FLOPs.
* **5.2**: trained on BRISC, evaluated zero-shot on all 3,064 Figshare images, with ΔAcc = BRISC-test acc − Figshare acc.
  BRISC has a `no_tumor` class that Figshare lacks; by default predictions are restricted to the 3 shared classes
  (`data.cross_domain.restrict_to_shared_classes`; set `false` to count `no_tumor` predictions as errors).
* **4.6.2**: paired t-test (df = k−1) and Wilcoxon signed-rank test of the reference model against every baseline on
  per-fold scores. The reference can be a `summary.json`, a model key, or a small JSON from the NeuroSAGE repo:
  `{"name": "NeuroSAGE-Net", "fold_scores": [97.1, 96.8, 97.4, 97.0, 96.9]}`.
  `--stat-source test` (default) pairs each fold model's accuracy on the held-out test set; `cv` uses the
  validation-fold accuracy.

## Do the baselines need K-fold training?

**Yes, for Table 4.6.2.** A paired t-test / Wilcoxon test compares *per-fold* scores of NeuroSAGE-Net and each
baseline, so every baseline needs k scores obtained on **the same folds** as NeuroSAGE-Net. A single run gives
one number per model and no test is possible. For 4.6.1, k-fold is recommended (report mean ± std); for 5.2 a
standard run is enough, but if you have k-fold runs the table reports mean ± std across the fold models.

Both options exist: `training.mode: standard | kfold`.

* Folds are **stratified** (and **patient-wise** for Figshare) and cached in `results/splits/*.json` as file
  lists. Use the same JSON in the NeuroSAGE-Net code so all models see identical folds.
* With **k = 5** the smallest possible two-sided Wilcoxon p-value is 0.0625, so Wilcoxon can never reach p < 0.05.
  Use **k = 10** if the Wilcoxon column must show significance (the t-test works with k = 5).
* For BRISC, k-fold runs on the official training set; each fold model is then tested on the official test set.

## Notes

* FLOPs follow the usual paper convention (1 multiply-accumulate = 1 FLOP, e.g. ResNet-50 ≈ 4.1 G); Params count
  learnable parameters with the task head (4 classes).
* ViG pretrained weights come from huawei-noah/Efficient-AI-Backbones releases; `arch` can be `pvig_ti/s/m/b`.
* The base models (ConvNeXt-B, ViT-B, Swin-B, ViG) use batch 16 × 2 gradient-accumulation steps so they fit a
  4 GB GPU; the effective batch size is 32 for all models.
* Kaggle downloads of public datasets usually work without credentials; otherwise set up `~/.kaggle/kaggle.json`
  or point `local_path` in `train.yaml` to an existing copy.
