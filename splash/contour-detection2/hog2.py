#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import hog
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

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
        feature_vector=False
    )
    H, W, c0, c1, o = feat.shape
    feat = feat.reshape(H, W, -1)
    return feat

def save_anomaly_map(anomaly_map, save_path, cell_size=8, block_size=2, scale=10.0):
    plt.figure(figsize=(6, 6))
    h_blocks, w_blocks = anomaly_map.shape
    block_px = cell_size * block_size
    img_w = w_blocks * block_px
    img_h = h_blocks * block_px
    plt.imshow(anomaly_map,
               cmap="jet",
               interpolation="nearest",
               extent=[0, img_w, img_h, 0],
               zorder=1)
    plt.colorbar()
    plt.axis("off")
    plt.xlim(0, img_w)
    plt.ylim(img_h, 0)
    h_idx, w_idx = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
    center_x = (w_idx + 0.5) * block_px
    center_y = (h_idx + 0.5) * block_px
    rect_w = block_px * scale
    rect_h = block_px * scale
    rect_x = center_x - rect_w / 2
    rect_y = center_y - rect_h / 2
    rect = Rectangle(
        (rect_x, rect_y),
        rect_w,
        rect_h,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
        zorder=10
    )
    plt.gca().add_patch(rect)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

def compute_anomaly_score(test_map, train_feat_maps):
    dist_maps = [np.linalg.norm(test_map - ref_map, axis=2) for ref_map in train_feat_maps]
    stacked = np.stack(dist_maps, axis=0)
    anomaly_map = np.min(stacked, axis=0)
    score = float(np.max(anomaly_map))
    return anomaly_map, score

def eval_threshold(val_df):
    y_true = (val_df["label"] == "NG").astype(int).values
    y_score = val_df["score"].values
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    best_f1 = -1
    best_thresh = None
    for t in thresholds:
        y_pred = (y_score > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"[AUC] {auc_score:.3f}, Best F1 = {best_f1:.3f} at threshold = {best_thresh:.3f}")
    return best_thresh

def main():
    base_dir = "/net/nfs3/export/home/hasegawa/workspace/"
    data_root = os.path.join(base_dir, "data", "bestex", "splash")
    out_dir = os.path.join(base_dir, "bestex", "splash", "contour-detection2", "_out")
    os.makedirs(out_dir, exist_ok=True)

    data = load_data(data_root)
    template_path = os.path.join(data_root, "20250326", "OK", "1.png")
    ref = cv2.imread(template_path)

    ok_items = [item for item in data if item["label"] == "OK"]
    ng_items = [item for item in data if item["label"] == "NG"]

    ok_train, ok_temp = train_test_split(ok_items, train_size=50, random_state=42)
    ok_val, ok_test = train_test_split(ok_temp, test_size=0.5, random_state=42)
    ng_val, ng_test = train_test_split(ng_items, test_size=0.5, random_state=42)

    train_feat_maps = []
    for item in ok_train:
        img = cv2.imread(item["path"])
        reg = center_crop(rigid_registration(img, ref), 3500)
        fmap = hog_feature_map(reg)
        train_feat_maps.append(fmap)

    def process(items, tag):
        results = []
        for item in items:
            img = cv2.imread(item["path"])
            reg = center_crop(rigid_registration(img, ref), 3500)
            fmap = hog_feature_map(reg)
            anomaly_map, score = compute_anomaly_score(fmap, train_feat_maps)
            results.append({"filename": os.path.basename(item["path"]), "label": item["label"], "score": score})
            save_dir = os.path.join(out_dir, f"anomaly_map_{tag}", item["label"])
            save_anomaly_map(anomaly_map, os.path.join(save_dir, f"{os.path.splitext(os.path.basename(item['path']))[0]}.png"))
        return pd.DataFrame(results)

    df_val = process(ok_val + ng_val, "val")
    df_test = process(ok_test + ng_test, "test")
    df_val.to_csv(os.path.join(out_dir, "val_score.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "test_score.csv"), index=False)

    threshold = eval_threshold(df_val)

    # apply threshold to test set
    df_test["pred"] = (df_test["score"] > threshold).astype(int)
    df_test["label_bin"] = (df_test["label"] == "NG").astype(int)
    cm = confusion_matrix(df_test["label_bin"], df_test["pred"])
    print("Confusion Matrix (test):\n", cm)

    # save misclassified samples
    err_df = df_test[df_test["pred"] != df_test["label_bin"]]
    err_df.to_excel(os.path.join(out_dir, "misclassified.xlsx"), index=False)

    draw_histogram(df_test, out_dir, threshold)

def draw_histogram(df, out_path, threshold):
    ok_scores = df[df["label"] == "OK"]["score"]
    ng_scores = df[df["label"] == "NG"]["score"]
    plt.figure(figsize=(8, 5))
    plt.hist(ok_scores, bins=20, alpha=0.6, label="OK", color="blue")
    plt.hist(ng_scores, bins=20, alpha=0.6, label="NG", color="red")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold: {threshold:.2f}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Anomaly Score Histogram")
    plt.legend()
    plt.grid(True)
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(out_path, "histogram.png"))
    plt.close()

if __name__ == "__main__":
    main()