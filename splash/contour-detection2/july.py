#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import hog

from src.loader import load_data
from src.registration import rigid_registration
from src.report import center_crop


def hog_feature_map(img, cell_size=8, block_size=2, orientations=9):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0

    feat = hog(
        img,
        orientations=orientations,
        pixels_per_cell=(cell_size, cell_size),
        cells_per_block=(block_size, block_size),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=False  # shape: (H_blocks, W_blocks, 2, 2, 9)
    )

    # reshape: (H_blocks, W_blocks, 2, 2, 9) → (H_blocks, W_blocks, 36)
    H, W, c0, c1, o = feat.shape
    feat = feat.reshape(H, W, -1)
    return feat  # shape: (H, W, 36)


def draw_histogram(df, out_path):
    ok_scores = df[df["label"] == "OK"]["score"]
    ng_scores = df[df["label"] == "NG"]["score"]

    plt.figure(figsize=(8, 5))
    plt.hist(ok_scores, bins=20, alpha=0.6, label="OK", color="blue")
    plt.hist(ng_scores, bins=20, alpha=0.6, label="NG", color="red")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Anomaly Score Histogram")
    plt.legend()
    plt.grid(True)
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(out_path, "histogram.png"))
    plt.close()


def save_anomaly_map(anomaly_map, save_path):
    plt.imshow(anomaly_map, cmap="jet")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    base_dir = "/net/nfs3/export/home/hasegawa/workspace/"
    data_root = os.path.join(base_dir, "data", "bestex", "splash")
    out_dir = os.path.join(base_dir, "bestex", "splash", "contour-detection2", "_out")
    os.makedirs(out_dir, exist_ok=True)

    max_train_ok = 50
    max_test_ok = 50
    max_test_ng = 100

    data = load_data(data_root)
    template_path = os.path.join(data_root, "20250326", "OK", "1.png")
    ref = cv2.imread(template_path)

    train_feat_maps = []
    test_results = []
    count_train_ok = 0
    count_test_ok = 0
    count_test_ng = 0

    for item in data:
        path = item["path"]
        label = item["label"]
        name = os.path.splitext(os.path.basename(path))[0]

        if label == "OK":
            if count_train_ok < max_train_ok:
                img = cv2.imread(path)
                reg = rigid_registration(img, ref)
                reg = center_crop(reg, 3500)
                fmap = hog_feature_map(reg)
                train_feat_maps.append(fmap)
                count_train_ok += 1
                continue
            elif count_test_ok >= max_test_ok:
                continue
        elif label == "NG":
            if count_test_ng >= max_test_ng:
                continue

        # --- テスト対象処理 ---
        img = cv2.imread(path)
        reg = rigid_registration(img, ref)
        reg = center_crop(reg, 3500)
        test_map = hog_feature_map(reg)

        ref_avg = np.mean(train_feat_maps, axis=0)  # shape: (H, W, 36)
        anomaly_map = np.linalg.norm(test_map - ref_avg, axis=2)  # shape: (H, W)

        score = float(np.max(anomaly_map))  # または np.mean(anomaly_map)

        test_results.append({
            "filename": name,
            "label": label,
            "score": score
        })

        if label == "OK":
            count_test_ok += 1
        else:
            count_test_ng += 1

        # --- ヒートマップ保存 ---
        anomaly_dir = os.path.join(out_dir, "anomaly_map", label)
        os.makedirs(anomaly_dir, exist_ok=True)
        save_anomaly_map(anomaly_map, os.path.join(anomaly_dir, f"{name}.png"))

    # --- 結果保存 ---
    df = pd.DataFrame(test_results)
    df.to_csv(os.path.join(out_dir, "score.csv"), index=False)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"train_ok = {count_train_ok}\n")
        f.write(f"test_ok  = {count_test_ok}\n")
        f.write(f"test_ng  = {count_test_ng}\n")
        f.write(f"total_test = {len(test_results)}\n")

    print(f"[INFO] train_ok = {count_train_ok}")
    print(f"[INFO] test_ok  = {count_test_ok}")
    print(f"[INFO] test_ng  = {count_test_ng}")
    print(f"[INFO] total_test = {len(test_results)}")

    draw_histogram(df, out_dir)


if __name__ == "__main__":
    main()
