#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

def load_data(root: str) -> list:
    """
    Returns: list of dicts: { "path": str, "label": "OK" or "NG" }
    """
    categories = ["OK", "NG"]
    image_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    data = []

    # 日付ディレクトリを走査
    for date_dir in sorted(os.listdir(root)):
        date_path = os.path.join(root, date_dir)
        if not os.path.isdir(date_path):
            continue

        for label in categories:
            class_dir = os.path.join(date_path, label)
            if not os.path.isdir(class_dir):
                print(f"[WARN] Skipped missing: {class_dir}")
                continue

            for ext in image_exts:
                for path in glob.glob(os.path.join(class_dir, ext)):
                    abs_path = os.path.abspath(path)
                    data.append({
                        "path": abs_path,
                        "label": label
                    })

    print(f"[INFO] Loaded {len(data)} images from {root}")
    return data