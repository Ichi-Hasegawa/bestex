#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

def compute_anomaly_score(edge_img, template_edge):
    """
    テンプレートとエッジ差分を取って異常スコアを計算
    Args:
        edge_img (np.ndarray): 入力画像のエッジ
        template_edge (np.ndarray): テンプレート画像のエッジ
    Returns:
        int: 異常スコア（差分のピクセル数）
    """
    # 差分マスク
    diff = cv2.absdiff(edge_img, template_edge)
    anomaly_score = np.sum(diff > 0)
    return int(anomaly_score)


def save_results(img_path, label, edge_img, score, output_dir):
    """
    エッジ画像を保存し、結果情報を辞書で返す
    Args:
        img_path (str): 元画像パス
        label (str): OK or NG
        edge_img (np.ndarray): エッジ画像
        score (int): 異常スコア
        output_dir (str): 保存先のルート
    Returns:
        dict: 結果データ
    """
    fname = os.path.basename(img_path)
    fname_noext = os.path.splitext(fname)[0]
    out_dir = os.path.join(output_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{fname_noext}_score{score}.png")
    cv2.imwrite(out_path, edge_img)

    return {
        "filename": fname,
        "label": label,
        "score": score,
        "saved_path": out_path
    }
