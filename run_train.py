
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lab1_core import (
    ACTIVATIONS,
    ANOMALIES_CSV,
    ART_DIR,
    DATA_TRAIN,
    DROPOUT_GRID,
    FIG_DIR,
    MODEL_DIR,
    N_CLASSES,
    SEED,
    classification_metrics,
    dump_json,
    load_mnist_csv,
    normalize_pixels,
    occlusion_map,
    plot_confusion,
    plot_history,
    plot_image_grid,
    plot_occlusion_triplet,
    plot_weight_maps,
    savefig,
    to_images,
)
from lab1_models import (
    jax_first_layer_weights,
    jax_predict_fn,
    jax_saliency,
    load_jax,
    load_torch,
    predict_jax,
    predict_torch,
    save_jax,
    save_torch,
    torch_predict_fn,
    torch_saliency,
    train_jax,
    train_torch,
)


def load_splits():
    z = np.load(ART_DIR / "splits.npz", allow_pickle=True)
    data = {k: z[k] for k in z.files}
    for k in list(data):
        if k.startswith("x_"):
            data[k] = normalize_pixels(data[k])
    return data


def run_grid(framework: str, data: dict, datasets: list[str], activations, dropouts) -> pd.DataFrame:
    rows = []
    x_va, y_va = data["x_va"], data["y_va"]
    x_te, y_te = data["x_test"], data["y_test"]
    train_x = {"original": data["x_tr"], "balanced": data["x_bal"], "cleaned": data["x_cln"]}
    train_y = {"original": data["y_tr"], "balanced": data["y_bal"], "cleaned": data["y_cln"]}

    for ds, act, p in itertools.product(datasets, activations, dropouts):
        tag = f"{framework}_{ds}_{act}_p{p}"
        print(f"\n>>> train {tag} | n={len(train_y[ds])}")
        if framework == "pytorch":
            bundle = train_torch(train_x[ds], train_y[ds], x_va, y_va, activation=act, dropout_p=p)
            pred_va, proba_va = predict_torch(bundle["model"], x_va)
            pred_te, proba_te = predict_torch(bundle["model"], x_te)
            save_torch(bundle["model"], MODEL_DIR / f"{tag}.pt")
            hist = bundle["history"]
            extra = {k: bundle[k] for k in ("n_params", "train_time_sec", "epochs_run", "best_val_loss", "device")}
        else:
            bundle = train_jax(train_x[ds], train_y[ds], x_va, y_va, activation=act, dropout_p=p)
            pred_va, proba_va = predict_jax(bundle["params"], x_va, act)
            pred_te, proba_te = predict_jax(bundle["params"], x_te, act)
            save_jax(bundle, MODEL_DIR / f"{tag}.npz")
            hist = bundle["history"]
            extra = {k: bundle[k] for k in ("n_params", "train_time_sec", "epochs_run", "best_val_loss", "device")}

        m_va = classification_metrics(y_va, pred_va, proba_va)
        m_te = classification_metrics(y_te, pred_te, proba_te)
        dump_json({"val": m_va, "test": m_te, "history": hist, **extra}, ART_DIR / f"metrics_{tag}.json")
        np.savez(ART_DIR / f"history_{tag}.npz", **{k: np.array(v) for k, v in hist.items()})

        row = {
            "framework": framework,
            "dataset": ds,
            "activation": act,
            "dropout_p": p,
            "tag": tag,
            "val_acc": m_va["accuracy"],
            "val_f1_macro": m_va["f1_macro"],
            "val_f1_weighted": m_va["f1_weighted"],
            "test_acc": m_te["accuracy"],
            "test_f1_macro": m_te["f1_macro"],
            "test_f1_weighted": m_te["f1_weighted"],
            "test_error_rate": m_te["error_rate"],
            "mean_conf_correct": m_te["mean_confidence_correct"],
            "mean_conf_incorrect": m_te["mean_confidence_incorrect"],
            **extra,
        }
        rows.append(row)
        print(
            f"    val_acc={row['val_acc']:.4f}  test_acc={row['test_acc']:.4f}  "
            f"time={row['train_time_sec']:.1f}s  epochs={row['epochs_run']}"
        )
    return pd.DataFrame(rows)


def pick_best(df: pd.DataFrame, dataset: str, framework: str) -> pd.Series:
    sub = df[(df["dataset"] == dataset) & (df["framework"] == framework)]
    return sub.sort_values(["val_f1_macro", "val_acc"], ascending=False).iloc[0]


def load_predict_bundle(framework: str, tag: str, activation: str):
    if framework == "pytorch":
        model = load_torch(MODEL_DIR / f"{tag}.pt")
        pred_fn = torch_predict_fn(model)
        W = model.first_layer_weights()
        sal_fn = lambda img, cls=None: torch_saliency(model, img, cls)
        predict = lambda x: predict_torch(model, x)
        return {"pred_fn": pred_fn, "W": W, "sal_fn": sal_fn, "predict": predict, "model": model}
    bundle = load_jax(MODEL_DIR / f"{tag}.npz")
    params, act = bundle["params"], bundle["activation"]
    pred_fn = jax_predict_fn(params, act)
    W = jax_first_layer_weights(params)
    sal_fn = lambda img, cls=None, a=act, p=params: jax_saliency(p, img, a, cls)
    predict = lambda x, a=act, p=params: predict_jax(p, x, a)
    return {"pred_fn": pred_fn, "W": W, "sal_fn": sal_fn, "predict": predict, "params": params}


def plot_metrics_heatmap(df: pd.DataFrame, framework: str, dataset: str, fname: str):
    sub = df[(df["framework"] == framework) & (df["dataset"] == dataset)]
    piv = sub.pivot(index="activation", columns="dropout_p", values="val_f1_macro")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(piv, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"{framework} / {dataset}: val macro-F1")
    return savefig(fname)


def plot_saliency_compare(image, maps: dict, title: str, fname: str):
    keys = list(maps)
    fig, axes = plt.subplots(1, 1 + len(keys), figsize=(3.2 * (1 + len(keys)), 3.0))
    axes[0].imshow(image.reshape(28, 28), cmap="gray")
    axes[0].set_title("вход")
    axes[0].axis("off")
    for ax, k in zip(axes[1:], keys):
        im = ax.imshow(maps[k], cmap="hot")
        ax.set_title(k)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, y=1.04)
    return savefig(fname)


def qualitative_panels(x, y, pred, proba, title_prefix, fname_prefix, n=8):
    conf = proba.max(axis=1)
    correct = pred == y
    groups = {
        "highconf_correct": np.where(correct & (conf >= np.quantile(conf[correct], 0.9) if correct.any() else 1))[0],
        "lowconf_correct": np.where(correct & (conf <= np.quantile(conf[correct], 0.1) if correct.any() else 0))[0],
        "highconf_wrong": np.where((~correct) & (conf >= (np.quantile(conf[~correct], 0.8) if (~correct).any() else 1)))[0],
        "lowconf_wrong": np.where((~correct) & (conf <= (np.quantile(conf[~correct], 0.2) if (~correct).any() else 0)))[0],
    }
    titles = {
        "highconf_correct": "Верно, высокая уверенность",
        "lowconf_correct": "Верно, низкая уверенность",
        "highconf_wrong": "Ошибка, высокая уверенность",
        "lowconf_wrong": "Ошибка, низкая уверенность",
    }
    rng = np.random.default_rng(SEED)
    for key, idx in groups.items():
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=min(n, len(idx)), replace=False)
        plot_image_grid(
            to_images(x[take] * 255.0),
            y[take],
            f"{title_prefix}: {titles[key]}",
            f"{fname_prefix}_{key}.png",
            ncols=8,
            preds=pred[take],
        )


def confused_pairs(cm: np.ndarray, k: int = 6):
    cm = np.array(cm).copy()
    np.fill_diagonal(cm, 0)
    pairs = []
    for _ in range(k):
        i, j = np.unravel_index(cm.argmax(), cm.shape)
        if cm[i, j] <= 0:
            break
        pairs.append((int(i), int(j), int(cm[i, j])))
        cm[i, j] = 0
    return pairs


def part3_for_framework(framework: str, df: pd.DataFrame, data: dict):
    x_te, y_te = data["x_test"], data["y_test"]
    models = {}
    preds = {}
    for ds in ("original", "balanced", "cleaned"):
        best = pick_best(df, ds, framework)
        tag = best["tag"]
        print(f"Part3 {framework} {ds}: {tag}")
        pack = load_predict_bundle(framework, tag, best["activation"])
        pred, proba = pack["predict"](x_te)
        models[ds] = {**pack, "tag": tag, "row": best, "pred": pred, "proba": proba}
        preds[ds] = pred
        m = classification_metrics(y_te, pred, proba)
        plot_confusion(np.array(m["confusion_matrix"]), f"{framework} {ds} — матрица ошибок (test)", f"cm_{framework}_{ds}.png")
        qualitative_panels(x_te, y_te, pred, proba, f"{framework}/{ds}", f"qual_{framework}_{ds}")
        plot_weight_maps(pack["W"], f"weights_{framework}_{ds}.png", n=16)

        # истории лучшей модели
        hpath = ART_DIR / f"history_{tag}.npz"
        if hpath.exists():
            h = np.load(hpath)
            plot_history(
                {tag: {k: h[k].tolist() for k in h.files}},
                f"hist_{framework}_{ds}.png",
            )

    # объекты, где три модели расходятся
    disagree = (preds["original"] != preds["balanced"]) | (preds["original"] != preds["cleaned"])
    idx_dis = np.where(disagree)[0]
    print(f"  disagree among 3 models: {len(idx_dis)}")
    if len(idx_dis):
        take = idx_dis[:18]
        fig, axes = plt.subplots(3, 6, figsize=(12, 6.5))
        axes = axes.ravel()
        for ax, i in zip(axes, take):
            ax.imshow(x_te[i].reshape(28, 28), cmap="gray")
            ax.set_title(
                f"y={int(y_te[i])}\nO={preds['original'][i]} B={preds['balanced'][i]} C={preds['cleaned'][i]}",
                fontsize=7,
            )
            ax.axis("off")
        fig.suptitle(f"{framework}: объекты с расхождением трёх моделей")
        savefig(f"disagree_{framework}.png")

    cm = np.array(
        classification_metrics(y_te, models["original"]["pred"], models["original"]["proba"])["confusion_matrix"]
    )
    pairs = confused_pairs(cm, k=6)
    dump_json({"confused_pairs": pairs}, ART_DIR / f"confused_{framework}.json")
    for a, b, cnt in pairs[:3]:
        mask = (y_te == a) & (models["original"]["pred"] == b)
        idx = np.where(mask)[0][:8]
        if len(idx):
            plot_image_grid(
                to_images(x_te[idx] * 255.0),
                y_te[idx],
                f"{framework}: часто путаемая пара {a}→{b} (n={cnt})",
                f"pair_{framework}_{a}_{b}.png",
                ncols=8,
                preds=models["original"]["pred"][idx],
            )

    anom = pd.read_csv(ANOMALIES_CSV)

    x_full, y_full = load_mnist_csv(DATA_TRAIN)
    x_an = normalize_pixels(x_full[anom["local_index"].to_numpy()])
    y_an = y_full[anom["local_index"].to_numpy()]
    anom_pred_table = anom.copy()
    for ds, pack in models.items():
        pr, pb = pack["predict"](x_an)
        anom_pred_table[f"{ds}_pred"] = pr
        anom_pred_table[f"{ds}_conf"] = pb.max(axis=1)
        anom_pred_table[f"{ds}_ok"] = pr == y_an
    anom_pred_table.to_csv(ART_DIR / f"anomalies_preds_{framework}.csv", index=False)

    show_n = min(12, len(y_an))
    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    for ax, i in zip(axes.ravel(), range(show_n)):
        ax.imshow(x_an[i].reshape(28, 28), cmap="gray")
        msg = " ".join(f"{ds[0].upper()}={int(anom_pred_table.loc[i, ds + '_pred'])}" for ds in models)
        ax.set_title(f"y={int(y_an[i])} {msg}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"{framework}: предсказания трёх моделей на аномалиях")
    savefig(f"anom_preds_{framework}.png")

    rng = np.random.default_rng(SEED)
    conf_o = models["original"]["proba"].max(axis=1)
    ok = np.where((models["original"]["pred"] == y_te) & (conf_o > 0.95))[0]
    bad = np.where(models["original"]["pred"] != y_te)[0]
    chosen = []
    if len(ok):
        chosen.append(int(rng.choice(ok)))
    if len(bad):
        chosen.append(int(rng.choice(bad)))
    if len(idx_dis):
        chosen.append(int(idx_dis[0]))
    chosen = list(dict.fromkeys(chosen))[:3]

    for img_i in chosen:
        img = x_te[img_i]
        all_maps = {}
        vmax = 0.0
        for ds, pack in models.items():
            maps = {
                k: occlusion_map(pack["pred_fn"], img, kernel=k, fill_value=0.0, mode="pred")
                for k in (1, 2, 3)
            }
            all_maps[ds] = maps
            vmax = max(vmax, max(float(np.percentile(m, 99)) for m in maps.values()))
        vmax = max(vmax, 1e-6)
        for ds, maps in all_maps.items():
            plot_occlusion_triplet(
                maps,
                img * 255.0,
                title=f"{framework}/{ds} окклюзия, idx={img_i}, y={int(y_te[img_i])}",
                fname=f"occ_{framework}_{ds}_{img_i}.png",
                vmin=0.0,
                vmax=vmax,
            )
        sal_maps = {ds: pack["sal_fn"](img) for ds, pack in models.items()}
        plot_saliency_compare(
            img,
            sal_maps,
            title=f"{framework}: |∇_x logit| idx={img_i} y={int(y_te[img_i])}",
            fname=f"sal_{framework}_{img_i}.png",
        )

    return models


def plot_framework_compare(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(
        axes, ["test_acc", "train_time_sec"], ["Test accuracy", "Время обучения, с"]
    ):
        sns.barplot(data=df, x="dataset", y=metric, hue="framework", ax=ax)
        ax.set_title(title)
    savefig("compare_framework_bars.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=df, x="train_time_sec", y="test_acc", hue="framework", style="dataset", s=80, ax=ax)
    ax.set_title("Сопоставление: время обучения vs test accuracy")
    savefig("compare_time_vs_acc.png")


def main():
    data = load_splits()
    print(
        "sizes:",
        {k: (data[k].shape if hasattr(data[k], "shape") else None) for k in ("x_tr", "x_bal", "x_cln", "x_va", "x_test")},
    )

    df_pt = run_grid("pytorch", data, ["original", "balanced", "cleaned"], ACTIVATIONS, DROPOUT_GRID)
    df_pt.to_csv(ART_DIR / "grid_pytorch.csv", index=False)

    df_jx_a = run_grid("jax", data, ["original", "balanced"], ACTIVATIONS, DROPOUT_GRID)
    df_jx_b = run_grid("jax", data, ["cleaned"], ["relu"], DROPOUT_GRID)
    best_cl_pt = pick_best(df_pt, "cleaned", "pytorch")
    extra = []
    if best_cl_pt["activation"] != "relu":
        extra.append(
            run_grid("jax", data, ["cleaned"], [best_cl_pt["activation"]], [float(best_cl_pt["dropout_p"])])
        )
    df_jx = pd.concat([df_jx_a, df_jx_b, *extra], ignore_index=True)
    df_jx.to_csv(ART_DIR / "grid_jax.csv", index=False)

    df = pd.concat([df_pt, df_jx], ignore_index=True)
    df.to_csv(ART_DIR / "grid_all.csv", index=False)

    for fw, dsf in [("pytorch", df_pt), ("jax", df_jx)]:
        for ds in dsf["dataset"].unique():
            plot_metrics_heatmap(dsf, fw, ds, f"heat_{fw}_{ds}.png")

    hist_pack = {}
    for fw in ("pytorch",):
        for p in (0.0, 0.5):
            tag = f"{fw}_original_relu_p{p}"
            hp = ART_DIR / f"history_{tag}.npz"
            if hp.exists():
                h = np.load(hp)
                hist_pack[tag] = {k: h[k].tolist() for k in h.files}
    if hist_pack:
        plot_history(hist_pack, "hist_dropout_effect.png")

    hist_act = {}
    for act in ACTIVATIONS:
        tag = f"pytorch_original_{act}_p0.2"
        hp = ART_DIR / f"history_{tag}.npz"
        if hp.exists():
            h = np.load(hp)
            hist_act[tag] = {k: h[k].tolist() for k in h.files}
    if hist_act:
        plot_history(hist_act, "hist_activation_effect.png")

    print("\n=== Part 3 PyTorch ===")
    part3_for_framework("pytorch", df_pt, data)
    print("\n=== Part 3 JAX ===")
    part3_for_framework("jax", df_jx, data)

    best_rows = []
    for fw, dsf in [("pytorch", df_pt), ("jax", df_jx)]:
        for ds in ("original", "balanced", "cleaned"):
            sub = dsf[dsf["dataset"] == ds]
            if len(sub):
                best_rows.append(pick_best(dsf, ds, fw))
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(ART_DIR / "best_models.csv", index=False)
    plot_framework_compare(best_df)

    dump_json(
        {
            "n_experiments": int(len(df)),
            "best": best_df.to_dict(orient="records"),
        },
        ART_DIR / "part2_summary.json",
    )
    print("Готово. Экспериментов:", len(df))


if __name__ == "__main__":
    main()
