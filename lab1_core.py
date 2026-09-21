
from __future__ import annotations

import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
IMG_SIZE = 28
N_PIXELS = IMG_SIZE * IMG_SIZE
N_CLASSES = 10
HIDDEN_DIM = 256  # размер скрытого слоя двухслойного перцептрона
BATCH_SIZE = 512  # GPU-resident: 512 даёт достаточно шагов Adam при малом overhead
EPOCHS = 12
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15
DROPOUT_GRID = [0.0, 0.2, 0.5]
ACTIVATIONS = ["relu", "tanh", "sigmoid"]
PIXEL_MIN, PIXEL_MAX = 0.0, 255.0

ROOT = Path(__file__).resolve().parent
VARIANT = os.environ.get("LAB1_VARIANT", "V2").upper()
_TRAIN_FILES = {"V1": "d1.csv", "V2": "d2.csv", "V3": "d3.csv", "V4": "d4.csv"}
if VARIANT not in _TRAIN_FILES:
    raise ValueError(f"Неизвестный вариант {VARIANT}")
DATA_TRAIN = ROOT / VARIANT / _TRAIN_FILES[VARIANT]
DATA_TEST = ROOT / VARIANT / "mnist_test.csv"
OUT_DIR = ROOT / f"outputs_{VARIANT.lower()}"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"
MODEL_DIR = OUT_DIR / "models"
ANOMALIES_CSV = ART_DIR / f"anomalies_{VARIANT.lower()}.csv"

for _d in (OUT_DIR, FIG_DIR, ART_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.family": "DejaVu Sans",
    }
)
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def set_global_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed(SEED)


def savefig(name: str) -> Path:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

# Загрузка и представление данных

def load_mnist_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"В файле {path} нет столбца label")
    pixel_cols = [c for c in df.columns if c != "label"]
    if len(pixel_cols) != N_PIXELS:
        raise ValueError(f"Ожидалось {N_PIXELS} пикселей, получено {len(pixel_cols)}")
    y = df["label"].to_numpy(dtype=np.int64)
    x = df[pixel_cols].to_numpy(dtype=np.float32)
    return x, y


def describe_dataset_structure(x: np.ndarray, y: np.ndarray, name: str) -> dict:
    info = {
        "name": name,
        "n_objects": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "image_shape": [IMG_SIZE, IMG_SIZE],
        "n_classes": int(len(np.unique(y))),
        "label_min": int(y.min()) if len(y) else None,
        "label_max": int(y.max()) if len(y) else None,
        "pixel_min": float(np.nanmin(x)),
        "pixel_max": float(np.nanmax(x)),
        "pixel_mean": float(np.nanmean(x)),
        "dtype": str(x.dtype),
    }
    return info


def to_images(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, IMG_SIZE, IMG_SIZE)


def normalize_pixels(x: np.ndarray) -> np.ndarray:
    """Линейная нормализация [0, 255] -> [0, 1]."""
    return (x / PIXEL_MAX).astype(np.float32)

# Статистики изображений и поиск аномалий
def image_statistics(x: np.ndarray, active_thr: float = 10.0) -> pd.DataFrame:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    pix_sum = x.sum(axis=1)
    n_nonzero = (x > 0).sum(axis=1)
    n_active = (x > active_thr).sum(axis=1)
    active_frac = n_active / float(N_PIXELS)
    return pd.DataFrame(
        {
            "mean": mean,
            "std": std,
            "sum": pix_sum,
            "n_nonzero": n_nonzero,
            "n_active": n_active,
            "active_frac": active_frac,
        }
    )


def class_centroids(x: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    cents = {}
    for c in range(N_CLASSES):
        mask = y == c
        cents[c] = x[mask].mean(axis=0) if mask.any() else np.zeros(x.shape[1], np.float32)
    return cents


def distances_to_centroids(x: np.ndarray, y: np.ndarray, cents: dict[int, np.ndarray]) -> np.ndarray:
    d = np.empty(len(y), dtype=np.float32)
    for c in range(N_CLASSES):
        mask = y == c
        if not mask.any():
            continue
        diff = x[mask] - cents[c]
        d[mask] = np.sqrt((diff * diff).sum(axis=1))
    return d


def find_duplicates(x: np.ndarray, y: np.ndarray) -> dict:
    """Дубликаты строк, одинаковые изображения с одной/разными метками."""
    keys = [hash(row.tobytes()) for row in np.round(x).astype(np.uint8)]
    groups: dict[int, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)

    dup_groups = {k: idxs for k, idxs in groups.items() if len(idxs) > 1}
    extra_copies = 0
    same_label_pairs = 0
    conflict_groups = []
    same_label_groups = []

    for k, idxs in dup_groups.items():
        extra_copies += len(idxs) - 1
        labels = y[idxs]
        if len(np.unique(labels)) > 1:
            conflict_groups.append(idxs)
        else:
            same_label_groups.append(idxs)
            same_label_pairs += len(idxs) - 1

    conflict_indices = sorted({i for g in conflict_groups for i in g})
    same_label_indices = sorted({i for g in same_label_groups for i in g})

    return {
        "n_duplicate_groups": len(dup_groups),
        "n_extra_copies": int(extra_copies),
        "n_same_label_extra": int(same_label_pairs),
        "n_conflict_groups": len(conflict_groups),
        "n_conflict_objects": len(conflict_indices),
        "conflict_groups": conflict_groups,
        "same_label_groups": same_label_groups,
        "conflict_indices": conflict_indices,
        "same_label_indices": same_label_indices,
    }


def iqr_outlier_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (values < lo) | (values > hi)


def detect_anomalies(x: np.ndarray, y: np.ndarray, row_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Комплексный поиск подозрительных объектов. Возвращает таблицу аномалий."""
    if row_ids is None:
        row_ids = np.arange(len(y))

    stats = image_statistics(x)
    cents = class_centroids(x, y)
    dist = distances_to_centroids(x, y, cents)
    stats["centroid_dist"] = dist
    stats["label"] = y
    stats["row_id"] = row_ids

    reasons: dict[int, list[str]] = defaultdict(list)

    # Пропуски и некорректные значения
    nan_mask = np.isnan(x).any(axis=1) | np.isinf(x).any(axis=1)
    for i in np.where(nan_mask)[0]:
        reasons[i].append("nan_or_inf")

    # Пиксели вне [0, 255]
    range_mask = (x.min(axis=1) < PIXEL_MIN) | (x.max(axis=1) > PIXEL_MAX)
    for i in np.where(range_mask)[0]:
        reasons[i].append("pixel_out_of_range")

    empty = stats["n_nonzero"].to_numpy() < 15
    almost_empty = stats["active_frac"].to_numpy() < 0.02
    almost_full = stats["active_frac"].to_numpy() > 0.60
    for i in np.where(empty)[0]:
        reasons[i].append("empty_or_too_sparse")
    for i in np.where(almost_empty & ~empty)[0]:
        reasons[i].append("almost_empty")
    for i in np.where(almost_full)[0]:
        reasons[i].append("almost_full")

    # Аномальная статистика интенсивности внутри класса (IQR)
    for c in range(N_CLASSES):
        m = y == c
        idx = np.where(m)[0]
        if len(idx) < 20:
            continue
        for col, tag in (
            ("mean", "intensity_mean_outlier"),
            ("std", "intensity_std_outlier"),
            ("active_frac", "fill_outlier"),
            ("centroid_dist", "far_from_class_centroid"),
        ):
            k = 2.2 if col == "centroid_dist" else 1.8
            mask_c = iqr_outlier_mask(stats.loc[m, col].to_numpy(), k=k)
            # для расстояния до центроида берём только правый хвост
            if col == "centroid_dist":
                q3 = np.percentile(stats.loc[m, col], 75)
                iqr = np.percentile(stats.loc[m, col], 75) - np.percentile(stats.loc[m, col], 25)
                mask_c = stats.loc[m, col].to_numpy() > (q3 + 2.2 * iqr)
            for i in idx[mask_c]:
                reasons[i].append(tag)

    # Дубликаты и конфликт меток
    dups = find_duplicates(x, y)
    for i in dups["conflict_indices"]:
        reasons[i].append("conflicting_label")
    for g in dups["same_label_groups"]:
        for i in g[1:]:  # оставляем первый экземпляр как оригинал
            reasons[i].append("duplicate_same_label")

    if not reasons:
        return pd.DataFrame(
            columns=["row_id", "label", "reasons", "mean", "std", "active_frac", "centroid_dist"]
        )

    rows = []
    for i, rs in reasons.items():
        uniq = sorted(set(rs))
        rows.append(
            {
                "row_id": int(row_ids[i]),
                "local_index": int(i),
                "label": int(y[i]),
                "reasons": ";".join(uniq),
                "n_reasons": len(uniq),
                "mean": float(stats.loc[i, "mean"]),
                "std": float(stats.loc[i, "std"]),
                "active_frac": float(stats.loc[i, "active_frac"]),
                "n_nonzero": int(stats.loc[i, "n_nonzero"]),
                "centroid_dist": float(stats.loc[i, "centroid_dist"]),
            }
        )
    anom = pd.DataFrame(rows).sort_values(["n_reasons", "row_id"], ascending=[False, True])
    return anom


def class_distribution(y: np.ndarray) -> pd.DataFrame:
    counts = pd.Series(y).value_counts().sort_index()
    for c in range(N_CLASSES):
        if c not in counts.index:
            counts.loc[c] = 0
    counts = counts.sort_index()
    total = counts.sum() if counts.sum() else 1
    df = pd.DataFrame(
        {
            "class": counts.index.astype(int),
            "count": counts.values.astype(int),
            "share": counts.values / total,
        }
    )
    return df


def imbalance_metrics(y: np.ndarray) -> dict:
    dist = class_distribution(y)
    counts = dist["count"].to_numpy(dtype=float)
    counts = np.maximum(counts, 1e-12)
    ir = float(counts.max() / counts.min())
    shares = counts / counts.sum()
    entropy = float(-(shares * np.log(shares)).sum())
    max_entropy = math.log(N_CLASSES)
    return {
        "n": int(len(y)),
        "min_class": int(dist.loc[dist["count"].idxmin(), "class"]),
        "max_class": int(dist.loc[dist["count"].idxmax(), "class"]),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "imbalance_ratio": ir,
        "entropy": entropy,
        "normalized_entropy": float(entropy / max_entropy),
        "majority_share": float(shares.max()),
    }


def brightness_by_class(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    stats = image_statistics(x)
    stats["label"] = y
    g = stats.groupby("label")[["mean", "std", "active_frac", "n_nonzero", "sum"]].agg(
        ["mean", "std", "min", "max"]
    )
    g.columns = ["_".join(c) for c in g.columns]
    return g.reset_index()


def representativeness_report(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Количественная оценка репрезентативности train относительно test."""
    dist_tr = class_distribution(y_train)
    dist_te = class_distribution(y_test)
    share_diff = (dist_tr["share"] - dist_te["share"]).abs()
    cents_tr = class_centroids(x_train, y_train)
    cents_te = class_centroids(x_test, y_test)
    centroid_shift = {
        int(c): float(np.linalg.norm(cents_tr[c] - cents_te[c])) for c in range(N_CLASSES)
    }
    return {
        "max_class_share_diff": float(share_diff.max()),
        "mean_class_share_diff": float(share_diff.mean()),
        "mean_pixel_train": float(x_train.mean()),
        "mean_pixel_test": float(x_test.mean()),
        "std_pixel_train": float(x_train.std()),
        "std_pixel_test": float(x_test.std()),
        "centroid_shift_by_class": centroid_shift,
        "mean_centroid_shift": float(np.mean(list(centroid_shift.values()))),
    }

# Балансировка и очистка
def oversample_balance(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Доводит каждый класс до размера самого частого класса повторной выборкой."""
    rng = np.random.default_rng(seed)
    target = max(int((y == c).sum()) for c in range(N_CLASSES))
    xs, ys, src = [], [], []
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=target, replace=len(idx) < target)
        xs.append(x[take])
        ys.append(y[take])
        src.append(take)
    order = rng.permutation(target * N_CLASSES)
    x_b = np.concatenate(xs, axis=0)[order]
    y_b = np.concatenate(ys, axis=0)[order]
    src_b = np.concatenate(src, axis=0)[order]
    return x_b, y_b, src_b


def clean_dataset(
    x: np.ndarray,
    y: np.ndarray,
    row_ids: np.ndarray,
    anomalies: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Удаляет дубликаты, конфликты меток и статистические аномалии."""
    drop = set(anomalies["local_index"].tolist()) if len(anomalies) else set()
    keep = np.array([i for i in range(len(y)) if i not in drop], dtype=np.int64)
    return x[keep], y[keep], row_ids[keep]


def stratified_split(x, y, row_ids, val_fraction=VAL_FRACTION, seed=SEED):
    x_tr, x_va, y_tr, y_va, id_tr, id_va = train_test_split(
        x, y, row_ids, test_size=val_fraction, random_state=seed, stratify=y
    )
    return x_tr, y_tr, id_tr, x_va, y_va, id_va

# Визуализации части 1
def plot_image_grid(images: np.ndarray, labels: np.ndarray, title: str, fname: str, ncols: int = 10, preds=None):
    n = len(images)
    ncols = min(ncols, n) if n else 1
    nrows = int(math.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.3 * ncols, 1.5 * nrows + 0.4))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        lab = str(int(labels[i]))
        if preds is not None:
            lab = f"y={lab} ŷ={int(preds[i])}"
        ax.set_title(lab, fontsize=8)
    fig.suptitle(title, y=1.02)
    return savefig(fname)


def plot_class_hist(y: np.ndarray, title: str, fname: str, color: str = "steelblue"):
    dist = class_distribution(y)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dist["class"], dist["count"], color=color, edgecolor="black", width=0.8)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xlabel("Класс")
    ax.set_ylabel("Число объектов")
    ax.set_title(title)
    for i, v in enumerate(dist["count"]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    return savefig(fname)


def plot_dimred(z: np.ndarray, y: np.ndarray, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=8, alpha=0.75, vmin=0, vmax=9)
    cb = plt.colorbar(sc, ax=ax, ticks=range(10))
    cb.set_label("Класс")
    ax.set_title(title)
    ax.set_xlabel("компонента 1")
    ax.set_ylabel("компонента 2")
    return savefig(fname)


def plot_intensity_distributions(x: np.ndarray, y: np.ndarray, fname: str):
    stats = image_statistics(x)
    stats["label"] = y.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=stats, x="label", y="mean", ax=axes[0], palette="tab10")
    axes[0].set_title("Средняя интенсивность по классам")
    axes[0].set_xlabel("Класс")
    axes[0].set_ylabel("mean(pixel)")
    sns.boxplot(data=stats, x="label", y="active_frac", ax=axes[1], palette="tab10")
    axes[1].set_title("Доля активных пикселей по классам")
    axes[1].set_xlabel("Класс")
    axes[1].set_ylabel("active_frac")
    return savefig(fname)


def plot_set_sizes(sizes: dict, fname: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(sizes.keys())
    vals = list(sizes.values())
    ax.bar(names, vals, color=["#4c72b0", "#dd8452", "#55a868"], edgecolor="black")
    ax.set_ylabel("Число объектов")
    ax.set_title("Сравнение объёмов исходного, сбалансированного и очищенного наборов")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.xticks(rotation=15)
    return savefig(fname)


def plot_class_compare(ys: dict[str, np.ndarray], fname: str):
    rows = []
    for name, y in ys.items():
        d = class_distribution(y)
        d["set"] = name
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=df, x="class", y="count", hue="set", ax=ax)
    ax.set_title("Распределение классов: исходный / сбалансированный / очищенный")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Число объектов")
    return savefig(fname)


def subsample_stratified(x, y, n=4000, seed=SEED):
    n = min(n, len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return x[idx], y[idx], idx

# Метрики части 2–3

def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    prec_m = precision_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    rec_m = recall_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    f1_m = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    wrong = y_true != y_pred
    n_wrong = int(wrong.sum())
    conf = proba.max(axis=1)
    correct = ~wrong
    report = classification_report(
        y_true, y_pred, labels=list(range(N_CLASSES)), digits=4, zero_division=0, output_dict=True
    )
    return {
        "accuracy": acc,
        "precision_per_class": prec_m.tolist(),
        "recall_per_class": rec_m.tolist(),
        "f1_per_class": f1_m.tolist(),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n_errors": n_wrong,
        "error_rate": float(n_wrong / max(len(y_true), 1)),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(conf[wrong].mean()) if wrong.any() else None,
        "report": report,
    }


def plot_confusion(cm: np.ndarray, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    return savefig(fname)


def plot_history(histories: dict, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{name} train")
        axes[0].plot(h["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(h["train_acc"], label=f"{name} train")
        axes[1].plot(h["val_acc"], linestyle="--", label=f"{name} val")
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Эпоха")
        ax.legend(fontsize=8)
    return savefig(fname)

# Интерпретация: окклюзия, карты весов, saliency

def occlusion_map(
    predict_fn,
    image: np.ndarray,
    kernel: int = 1,
    fill_value: float = 0.0,
    mode: str = "pred",
    true_label: int | None = None,
) -> np.ndarray:
    """Карта падения уверенности при закрашивании окна kernel x kernel.

    predict_fn(x_batch[N,784]) -> proba[N,10]
    Возвращает карту 28x28: среднее падение уверенности по всем окнам, накрывшим пиксель.
    """
    img = image.reshape(IMG_SIZE, IMG_SIZE).astype(np.float32)
    base = predict_fn(img.reshape(1, -1))[0]
    if mode == "true" and true_label is not None:
        cls = int(true_label)
    else:
        cls = int(np.argmax(base))
    base_conf = float(base[cls])

    positions = []
    batch = []
    for i in range(0, IMG_SIZE - kernel + 1):
        for j in range(0, IMG_SIZE - kernel + 1):
            occ = img.copy()
            occ[i : i + kernel, j : j + kernel] = fill_value
            positions.append((i, j))
            batch.append(occ.reshape(-1))
    batch = np.stack(batch, axis=0)
    probs = []
    bs = 4096
    for s in range(0, len(batch), bs):
        probs.append(predict_fn(batch[s : s + bs]))
    probs = np.concatenate(probs, axis=0)

    drops = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    counts = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for (i, j), p in zip(positions, probs):
        drop = base_conf - float(p[cls])
        drops[i : i + kernel, j : j + kernel] += drop
        counts[i : i + kernel, j : j + kernel] += 1
    counts = np.maximum(counts, 1.0)
    return drops / counts


def plot_occlusion_triplet(maps: dict[int, np.ndarray], image: np.ndarray, title: str, fname: str, vmin=None, vmax=None):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    axes[0].imshow(image.reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    axes[0].set_title("Исходное")
    axes[0].axis("off")
    for ax, k in zip(axes[1:], sorted(maps)):
        im = ax.imshow(maps[k], cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"ядро {k}×{k}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, y=1.05)
    return savefig(fname)


def plot_weight_maps(W: np.ndarray, fname: str, n: int = 16):
    """W: (hidden, 784) — веса первого слоя."""
    norms = np.linalg.norm(W, axis=1)
    pick = np.argsort(norms)[::-1][:n]
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    lim = np.percentile(np.abs(W[pick]), 99)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(pick):
            continue
        nid = pick[i]
        im = ax.imshow(W[nid].reshape(IMG_SIZE, IMG_SIZE), cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(f"нейрон {nid}\n||w||={norms[nid]:.2f}", fontsize=8)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    fig.suptitle("Карты весов нейронов первого полносвязного слоя (28×28)", y=1.01)
    return savefig(fname)


def dump_json(obj, path: Path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding="utf-8")
