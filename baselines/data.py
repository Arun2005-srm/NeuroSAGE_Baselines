"""Dataset download (Kaggle), indexing, splitting and torch Datasets.

Every dataset is turned into  data/<name>/index.csv  with columns
    path   absolute image path
    label  canonical class name (glioma | meningioma | pituitary | no_tumor)
    split  train | test | ''   ('' = no official split)
    group  patient id ('' if unknown) - used for patient-wise splits
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_ALIASES = {
    "glioma": "glioma", "glioma_tumor": "glioma", "gl": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma", "me": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary", "pi": "pituitary",
    "no_tumor": "no_tumor", "notumor": "no_tumor", "no": "no_tumor", "healthy": "no_tumor",
    "normal": "no_tumor", "nt": "no_tumor",
}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "train", "valid": "train",
                 "validation": "train", "test": "test", "testing": "test"}


def canonical_class(name: str) -> str | None:
    return CLASS_ALIASES.get(name.strip().lower().replace("-", "_").replace(" ", "_"))


# --------------------------------------------------------------------------- download / prepare
def _resolve_source(name: str, dcfg: dict) -> Path:
    if dcfg.get("local_path"):
        p = Path(dcfg["local_path"]).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"[{name}] local_path does not exist: {p}")
        return p
    slug = dcfg.get("kaggle_slug")
    if not slug:
        raise ValueError(f"[{name}] needs either `local_path` or `kaggle_slug` in configs/train.yaml")
    import kagglehub  # lazy: only needed for downloading

    print(f"[{name}] downloading Kaggle dataset '{slug}' via kagglehub (cached after first time) ...")
    return Path(kagglehub.dataset_download(slug))


def _prepare_imagefolder(src: Path, dcfg: dict) -> list[dict]:
    root = src
    if dcfg.get("subdir"):
        hits = [p for p in src.rglob(dcfg["subdir"]) if p.is_dir()]
        if not hits:
            raise FileNotFoundError(f"subdir '{dcfg['subdir']}' not found under {src}")
        root = hits[0]
    label_map = {str(k): v for k, v in (dcfg.get("label_map") or {}).items()}
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        parent = p.parent.name
        label = label_map.get(parent) or canonical_class(parent)
        if label is None:
            continue
        split = ""
        for part in p.relative_to(root).parts[:-1]:
            if part.lower() in SPLIT_ALIASES:
                split = SPLIT_ALIASES[part.lower()]
        rows.append({"path": str(p.resolve()), "label": label, "split": split, "group": ""})
    return rows


def _read_figshare_mat(path: Path):
    """Cheng et al. .mat (MATLAB v7.3 / HDF5, with a scipy fallback for older versions)."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            img = np.array(cj["image"]).T  # h5py returns MATLAB arrays transposed
            label = int(np.array(cj["label"]).squeeze())
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).flatten())
        return img, label, pid
    except OSError:
        from scipy.io import loadmat

        cj = loadmat(str(path))["cjdata"][0, 0]
        return np.array(cj["image"]), int(np.array(cj["label"]).squeeze()), str(np.array(cj["PID"]).squeeze())


def _prepare_figshare_mat(src: Path, out_dir: Path, dcfg: dict) -> list[dict]:
    label_map = {int(k): v for k, v in dcfg["label_map"].items()}
    mats = sorted(p for p in src.rglob("*.mat") if p.stem.lower() != "cvind")
    if not mats:
        raise FileNotFoundError(f"No .mat files found under {src}")
    img_dir = out_dir / "images"
    rows = []
    print(f"converting {len(mats)} .mat files to PNG -> {img_dir}")
    for i, m in enumerate(mats):
        img, lab, pid = _read_figshare_mat(m)
        img = img.astype(np.float32)
        img = (img - img.min()) / max(float(img.max() - img.min()), 1e-8) * 255.0
        cls = label_map[lab]
        dst = img_dir / cls / f"{m.stem}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            Image.fromarray(img.astype(np.uint8)).save(dst)
        rows.append({"path": str(dst.resolve()), "label": cls, "split": "", "group": pid})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(mats)}")
    return rows


def prepare_dataset(name: str, cfg: dict, force: bool = False) -> Path:
    dcfg = cfg["data"]["datasets"][name]
    out_dir = Path(cfg["data"]["root"]) / name
    index = out_dir / "index.csv"
    if index.exists() and not force:
        return index
    src = _resolve_source(name, dcfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = dcfg.get("format", "imagefolder")
    if fmt == "figshare_mat":
        rows = _prepare_figshare_mat(src, out_dir, dcfg)
    elif fmt == "imagefolder":
        rows = _prepare_imagefolder(src, dcfg)
    else:
        raise ValueError(f"[{name}] unknown format '{fmt}'")
    if not rows:
        raise RuntimeError(f"[{name}] no labelled images found under {src}")
    df = pd.DataFrame(rows)
    df.to_csv(index, index=False)
    print(f"[{name}] indexed {len(df)} images -> {index}")
    print(df.groupby(["split", "label"]).size().to_string())
    return index


def load_index(name: str, cfg: dict) -> pd.DataFrame:
    if name not in cfg["data"]["datasets"]:
        raise KeyError(f"Unknown dataset '{name}'. Defined: {list(cfg['data']['datasets'])}")
    index = Path(cfg["data"]["root"]) / name / "index.csv"
    if not index.exists():
        if not cfg["data"].get("auto_prepare", True):
            raise FileNotFoundError(f"{index} missing - run: python download_data.py --datasets {name}")
        prepare_dataset(name, cfg)
    df = pd.read_csv(index, keep_default_na=False, dtype={"group": str, "split": str})
    classes = cfg["data"]["datasets"][name]["classes"]
    df = df[df["label"].isin(classes)].reset_index(drop=True)
    # portable sample id (split/class/file) - used in the cached split files
    df["name"] = df["path"].map(lambda p: "/".join(Path(p).parts[-3:]))
    if df["name"].duplicated().any():
        raise RuntimeError(f"[{name}] duplicate sample ids in index; check {index}")
    return df


# --------------------------------------------------------------------------- splits
def _groups(df: pd.DataFrame):
    g = df["group"].astype(str)
    return g.values if (g != "").all() and g.nunique() < len(g) else None


def _holdout(df: pd.DataFrame, ratio: float, seed: int):
    y, groups = df["label"].values, _groups(df)
    idx = np.arange(len(df))
    if groups is not None:  # patient-wise stratified hold-out
        n_splits = max(2, int(round(1.0 / ratio)))
        tr, te = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(idx, y, groups))
        return tr, te
    return train_test_split(idx, test_size=ratio, stratify=y, random_state=seed)


def _kfold(df: pd.DataFrame, k: int, seed: int):
    y, groups = df["label"].values, _groups(df)
    idx = np.arange(len(df))
    if groups is not None:
        return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(idx, y, groups))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y))


def make_splits(name: str, df: pd.DataFrame, cfg: dict) -> dict:
    """Returns {'test': [file names], 'folds': [{'train': [...], 'val': [...]}, ...]}.

    Splits are cached in <output_dir>/splits/ as JSON (file names only) so every model -
    including NeuroSAGE-Net in its own codebase - is trained / tested on identical folds.
    """
    tcfg, dcfg = cfg["training"], cfg["data"]
    seed, mode = cfg["experiment"]["seed"], tcfg["mode"]
    k = int(tcfg["k_folds"]) if mode == "kfold" else 1
    tag = f"{name}__{mode}" + (f"_k{k}" if mode == "kfold" else f"_val{dcfg['val_ratio']}") + f"__seed{seed}"
    cache = Path(cfg["experiment"]["output_dir"]) / "splits" / f"{tag}.json"
    names = set(df["name"])
    if cache.exists():
        sp = json.loads(cache.read_text())
        used = set(sp["test"]) | set(sp["folds"][0]["train"]) | set(sp["folds"][0]["val"])
        if used == names:
            print(f"[{name}] reusing cached splits {cache}")
            return sp
        print(f"[{name}] cached splits {cache} do not match the current index - regenerating")

    if (df["split"] == "test").any():  # official train/test split (BRISC)
        pool = df[df["split"] != "test"].reset_index(drop=True)
        test = df[df["split"] == "test"]
    else:  # no official split (Figshare): patient-wise stratified hold-out
        tr, te = _holdout(df, float(dcfg["test_ratio"]), seed)
        pool, test = df.iloc[tr].reset_index(drop=True), df.iloc[te]

    if mode == "kfold":
        folds = _kfold(pool, k, seed)
    else:
        folds = [_holdout(pool, float(dcfg["val_ratio"]), seed)]
    sp = {
        "dataset": name, "mode": mode, "k": k, "seed": seed,
        "test": sorted(test["name"]),
        "folds": [{"train": sorted(pool["name"].iloc[a]), "val": sorted(pool["name"].iloc[b])} for a, b in folds],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sp))
    print(f"[{name}] splits saved -> {cache}  (pool {len(pool)}, test {len(test)}, folds {k})")
    return sp


# --------------------------------------------------------------------------- torch dataset
class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i], dtype=torch.long)


def build_transforms(cfg: dict, mean, std, train: bool):
    size, a = int(cfg["data"]["image_size"]), cfg["augmentation"]
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        return T.Compose([T.Resize((size, size)), *norm])
    return T.Compose([
        T.RandomResizedCrop(size, scale=tuple(a["random_resized_crop_scale"]), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(a["hflip"]),
        T.RandomRotation(a["rotation"]),
        T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        *norm,
    ])


def subset(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    s = set(names)
    return df[df["name"].isin(s)].reset_index(drop=True)
