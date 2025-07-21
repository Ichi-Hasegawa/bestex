#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2

from src.loader import load_data
from src.registration import rigid_registration
from src.feature import hog_feature
from src.scorer import anomaly_score
from src.report import center_crop, save_overlay, save_hog_visual


def main():

    base_dir = "/net/nfs3/export/home/hasegawa/workspace/"
    data_root = os.path.join(base_dir, "data", "bestex", "splash")
    out_dir = os.path.join(base_dir, "bestex", "splash", "contour-detection2", "_out")
    os.makedirs(out_dir, exist_ok=True)

    data = load_data(data_root)

    # テンプレート画像（OK画像1枚）
    template_path = os.path.join(data_root, "20250326", "OK", "1.png")
    ref = cv2.imread(template_path)

    normal_feats = []
    results = []

    for item in data:
        path = item["path"]
        label = item["label"]
        name = os.path.splitext(os.path.basename(path))[0]

        img = cv2.imread(path)
        reg = rigid_registration(img, ref)
        img = center_crop(img, 3500)
        reg = center_crop(reg, 3500)
        # overlay_dir = os.path.join(out_dir, "overlay", label)
        # os.makedirs(overlay_dir, exist_ok=True)
        # overlay_path = os.path.join(overlay_dir, f"{name}.png")
        # save_overlay(reg, ref, overlay_path)

        
        feat = hog_feature(reg)

        if label == "OK":
            normal_feats.append(feat)
            score = 0.0
        else:
            score = anomaly_score(feat, normal_feats)

        results.append({
            "filename": name,
            "label": label,
            "score": score
        })

        hog_dir = os.path.join(out_dir, "hog", label)
        os.makedirs(hog_dir, exist_ok=True)
        save_hog_visual(reg, os.path.join(hog_dir, f"{name}.png"))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "score.csv"), index=False)
    print(f"[INFO] Saved {len(df)} results")

if __name__ == "__main__":
    main()