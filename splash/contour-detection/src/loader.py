#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def load_image_paths(base_dir):
    """
    base_dir配下の OK / NG フォルダから画像パスを読み込みリストで返す
    Args:
        base_dir (str): ベースディレクトリパス（例：.../data/20250326）
    Returns:
        tuple: (ok_paths, ng_paths)
    """
    ok_dir = os.path.join(base_dir, "OK")
    ng_dir = os.path.join(base_dir, "NG")

    def get_image_list(folder):
        return [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

    ok_paths = get_image_list(ok_dir)
    ng_paths = get_image_list(ng_dir)

    print(f"[INFO] 読み込み完了: OK={len(ok_paths)}枚, NG={len(ng_paths)}枚")
    return ok_paths, ng_paths
