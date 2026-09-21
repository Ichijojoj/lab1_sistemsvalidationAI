
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lab1_core import (
    ART_DIR,
    DATA_TEST,
    DATA_TRAIN,
    FIG_DIR,
    N_CLASSES,
    SEED,
    VARIANT,
    ANOMALIES_CSV,
    brightness_by_class,
    class_distribution,
    clean_dataset,
    describe_dataset_structure,
    detect_anomalies,
    dump_json,
    find_duplicates,
    image_statistics,
    imbalance_metrics,
    load_mnist_csv,
    oversample_balance,
    plot_class_compare,
    plot_class_hist,
    plot_dimred,
    plot_image_grid,
    plot_intensity_distributions,
    plot_set_sizes,
    representativeness_report,
    stratified_split,
    subsample_stratified,
    to_images,
)


def try_umap(x_sub, n_components=2, seed=SEED):
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, random_state=seed)
        return reducer.fit_transform(x_sub), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def main():
    print(f"=== Загрузка {VARIANT} ===")
    x_all, y_all = load_mnist_csv(DATA_TRAIN)
    x_test, y_test = load_mnist_csv(DATA_TEST)
    row_ids = np.arange(len(y_all))

    info_train = describe_dataset_structure(x_all, y_all, f"{DATA_TRAIN.name} (train)")
    info_test = describe_dataset_structure(x_test, y_test, "mnist_test.csv")
    print(json.dumps({"train": info_train, "test": info_test}, indent=2, ensure_ascii=False))

    print("=== Дубликаты ===")
    dups = find_duplicates(x_all, y_all)
    dups_pub = {k: v for k, v in dups.items() if k not in {"conflict_groups", "same_label_groups"}}
    print(json.dumps(dups_pub, indent=2, ensure_ascii=False))

    print("=== Аномалии ===")
    anomalies = detect_anomalies(x_all, y_all, row_ids=row_ids)
    anomalies.to_csv(ANOMALIES_CSV, index=False)
    print(f"аномалий: {len(anomalies)}  доля: {len(anomalies)/len(y_all):.4f}")
    print(anomalies["reasons"].str.split(";").explode().value_counts().head(20).to_string())

    print("=== Дисбаланс ===")
    dist = class_distribution(y_all)
    dist.to_csv(ART_DIR / "class_distribution_original.csv", index=False)
    imb = imbalance_metrics(y_all)
    print(json.dumps(imb, indent=2, ensure_ascii=False))

    bright = brightness_by_class(x_all, y_all)
    bright.to_csv(ART_DIR / "brightness_by_class.csv", index=False)

    print("=== Репрезентативность train vs test ===")
    rep = representativeness_report(x_all, y_all, x_test, y_test)
    dump_json(rep, ART_DIR / "representativeness.json")
    print(json.dumps({k: v for k, v in rep.items() if k != "centroid_shift_by_class"}, indent=2))

    print("=== Визуализации примеров ===")
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(y_all), size=30, replace=False)
    plot_image_grid(to_images(x_all[sample_idx]), y_all[sample_idx], "Примеры исходных изображений 28×28", "01_examples.png")

    # по одному примеру на класс
    one_per_class = [np.where(y_all == c)[0][0] for c in range(N_CLASSES)]
    plot_image_grid(
        to_images(x_all[one_per_class]),
        y_all[one_per_class],
        "По одному примеру на класс",
        "02_one_per_class.png",
        ncols=10,
    )

    plot_class_hist(y_all, f"Распределение объектов по классам (исходный набор {VARIANT})", "03_class_hist.png")
    plot_intensity_distributions(x_all, y_all, "04_intensity_boxplots.png")

    print("=== Понижение размерности (подвыборка 4000) ===")
    x_sub, y_sub, _ = subsample_stratified(x_all / 255.0, y_all, n=4000, seed=SEED)
    pca = PCA(n_components=2, random_state=SEED)
    z_pca = pca.fit_transform(x_sub)
    plot_dimred(z_pca, y_sub, f"PCA (explained {pca.explained_variance_ratio_.sum():.1%})", "05_pca.png")
    dump_json(
        {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "n": 4000},
        ART_DIR / "pca_info.json",
    )

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED, init="pca")
    z_tsne = tsne.fit_transform(x_sub)
    plot_dimred(z_tsne, y_sub, "t-SNE (perplexity=30)", "06_tsne.png")

    z_umap, umap_err = try_umap(x_sub)
    if z_umap is not None:
        plot_dimred(z_umap, y_sub, "UMAP (n_neighbors=15, min_dist=0.1)", "07_umap.png")
        print("UMAP: OK")
    else:
        print("UMAP недоступен:", umap_err)

    print("=== Примеры дубликатов / конфликтов / аномалий ===")
    if dups["same_label_groups"]:
        g = dups["same_label_groups"][0]
        show = g[: min(8, len(g))]
        plot_image_grid(to_images(x_all[show]), y_all[show], "Дубликаты с одинаковой меткой", "08_duplicates_same.png", ncols=8)
    if dups["conflict_groups"]:
        g = dups["conflict_groups"][0]
        show = g[: min(8, len(g))]
        plot_image_grid(
            to_images(x_all[show]),
            y_all[show],
            "Одинаковые изображения с разными метками",
            "09_duplicates_conflict.png",
            ncols=8,
        )
    else:
        print("Конфликтующих меток не найдено")

    if len(anomalies):
        pick = anomalies.head(20)
        idx = pick["local_index"].to_numpy()
        plot_image_grid(
            to_images(x_all[idx]),
            y_all[idx],
            "Примеры аномальных объектов (топ по числу причин)",
            "10_anomalies.png",
            ncols=10,
        )

        for tag, fname, title in [
            ("empty_or_too_sparse", "11_empty.png", "Пустые / слишком разреженные"),
            ("almost_full", "12_almost_full.png", "Почти полностью заполненные"),
            ("far_from_class_centroid", "13_far_centroid.png", "Далеко от центроида класса"),
            ("conflicting_label", "14_conflicts.png", "Конфликт меток"),
        ]:
            sub = anomalies[anomalies["reasons"].str.contains(tag)]
            if len(sub):
                idx = sub["local_index"].head(12).to_numpy()
                plot_image_grid(to_images(x_all[idx]), y_all[idx], title, fname, ncols=6)

    print("=== Разделение train/val + балансировка + очистка ===")
    x_tr, y_tr, id_tr, x_va, y_va, id_va = stratified_split(x_all, y_all, row_ids)
    anom_tr = detect_anomalies(x_tr, y_tr, row_ids=id_tr)
    anom_tr.to_csv(ART_DIR / "anomalies_train_split.csv", index=False)

    x_bal, y_bal, _ = oversample_balance(x_tr, y_tr, seed=SEED)
    x_cln, y_cln, id_cln = clean_dataset(x_tr, y_tr, id_tr, anom_tr)

    np.savez(
        ART_DIR / "splits.npz",
        x_tr=x_tr,
        y_tr=y_tr,
        id_tr=id_tr,
        x_va=x_va,
        y_va=y_va,
        id_va=id_va,
        x_bal=x_bal,
        y_bal=y_bal,
        x_cln=x_cln,
        y_cln=y_cln,
        id_cln=id_cln,
        x_test=x_test,
        y_test=y_test,
    )

    plot_class_hist(y_tr, "Классы: обучающая выборка (исходная)", "15_hist_train.png")
    plot_class_hist(y_bal, "Классы: сбалансированная обучающая выборка", "16_hist_balanced.png", color="#dd8452")
    plot_class_hist(y_cln, "Классы: очищенная обучающая выборка", "17_hist_cleaned.png", color="#55a868")
    plot_class_compare({"исходный train": y_tr, "сбалансированный": y_bal, "очищенный": y_cln}, "18_class_compare.png")
    plot_set_sizes(
        {
            "исходный train": len(y_tr),
            "сбалансированный": len(y_bal),
            "очищенный": len(y_cln),
        },
        "19_set_sizes.png",
    )

    summary = {
        "train_file": str(DATA_TRAIN),
        "test_file": str(DATA_TEST),
        "structure_train": info_train,
        "structure_test": info_test,
        "duplicates": dups_pub,
        "n_anomalies_full": int(len(anomalies)),
        "anomaly_share_full": float(len(anomalies) / len(y_all)),
        "anomaly_reason_counts": anomalies["reasons"].str.split(";").explode().value_counts().to_dict()
        if len(anomalies)
        else {},
        "imbalance_full": imb,
        "imbalance_train": imbalance_metrics(y_tr),
        "imbalance_balanced": imbalance_metrics(y_bal),
        "imbalance_cleaned": imbalance_metrics(y_cln),
        "split": {
            "train": int(len(y_tr)),
            "val": int(len(y_va)),
            "test": int(len(y_test)),
            "balanced_train": int(len(y_bal)),
            "cleaned_train": int(len(y_cln)),
            "val_fraction": 0.15,
            "stratified": True,
            "seed": SEED,
        },
        "representativeness": {k: v for k, v in rep.items() if k != "centroid_shift_by_class"},
        "centroid_shift_by_class": rep["centroid_shift_by_class"],
        "normalization": "x / 255.0 -> [0, 1]",
        "balancing": "oversampling меньшинства до размера majority с возвращением",
        "cleaning": "удаление дубликатов (кроме первого), конфликтов меток и статистических выбросов",
        "umap_error": umap_err,
        "pca_explained": pca.explained_variance_ratio_.tolist(),
    }
    dump_json(summary, ART_DIR / "part1_summary.json")
    print("Сохранено:", ART_DIR / "part1_summary.json")
    print("Фигуры:", FIG_DIR)
    print("Готово.")


if __name__ == "__main__":
    main()
