#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def align_to_template(img_path, template_path, max_features=500, good_match_ratio=0.15):
    """
    ORB特徴点＋Homographyで位置合わせを行う
    Args:
        img_path (str): 入力画像のファイルパス
        template_path (str): テンプレート画像のファイルパス
    Returns:
        aligned_img (np.ndarray): テンプレートに位置合わせされた画像（グレースケール）
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if img is None or template is None:
        raise ValueError(f"画像読み込みエラー: {img_path} または {template_path}")

    # ORB特徴点抽出
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    if des1 is None or des2 is None:
        raise ValueError("特徴点が検出できませんでした")

    # マッチング
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 上位マッチを使用
    num_good_matches = int(len(matches) * good_match_ratio)
    good_matches = matches[:num_good_matches]

    if len(good_matches) < 4:
        raise ValueError("十分なマッチが見つかりません")

    # Homography推定
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    height, width = template.shape
    aligned_img = cv2.warpPerspective(img, H, (width, height))

    return aligned_img

def overlay_images(template, aligned, threshold=20, red_alpha=0.5):
    """
    基準画像と位置合わせ後の画像を重ね、ズレた部分を半透明赤で強調表示。

    Args:
        template (np.ndarray): テンプレート画像（グレースケール）
        aligned (np.ndarray): 位置合わせ後画像（グレースケール）
        threshold (int): 差分ピクセル値の閾値
        red_alpha (float): 赤ハイライトの透明度（0.0〜1.0）

    Returns:
        overlay (np.ndarray): 可視化用オーバーレイ画像（BGR）
    """
    # 差分マスク
    diff = cv2.absdiff(template, aligned)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # カラー変換
    template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    aligned_color = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)

    # 背景：αブレンドで重ねる
    blend = cv2.addWeighted(template_color, 0.5, aligned_color, 0.5, 0)

    # 赤マスク画像を作成
    red_layer = np.zeros_like(blend)
    red_layer[:, :] = [0, 0, 255]  # BGR: 赤

    # 赤をマスク領域にのみブレンド
    mask_3ch = cv2.merge([mask, mask, mask])
    blend = np.where(mask_3ch > 0, cv2.addWeighted(blend, 1 - red_alpha, red_layer, red_alpha, 0), blend)

    return blend

def crop_center(img, crop_border=100):
    """
    上下左右から指定ピクセルを除外して中央を切り出す
    Args:
        img (np.ndarray): 入力画像
        crop_border (int): 上下左右の切り落としサイズ
    Returns:
        np.ndarray: クロップ済み画像
    """
    h, w = img.shape[:2]
    return img[crop_border:h - crop_border, crop_border:w - crop_border]
