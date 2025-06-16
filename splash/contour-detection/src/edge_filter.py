#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

def extract_edges(img, use_clahe=True, low_thresh=30, high_thresh=150):
    """
    エッジ抽出（前処理込み）
    Args:
        img (np.ndarray): 位置合わせ後のグレースケール画像
        use_clahe (bool): コントラスト強調の有無
        low_thresh (int): Canny下限しきい値
        high_thresh (int): Canny上限しきい値
    Returns:
        edges (np.ndarray): エッジ抽出後の画像（2値）
    """
    # ノイズ除去
    blurred = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHEで局所コントラスト強調
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(blurred)
    else:
        processed = blurred

    # Cannyでエッジ抽出
    edges = cv2.Canny(processed, threshold1=low_thresh, threshold2=high_thresh)

    return edges
