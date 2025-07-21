#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

def center_crop(img, size):
    h, w = img.shape[:2]
    ch, cw = h // 2, w // 2
    half = size // 2
    return img[ch - half:ch + half, cw - half:cw + half]


def save_overlay(
    aligned, ref, out_path, crop_size=3500
):
    """
    Args:
        aligned (np.ndarray): 位置合わせ後の画像
        ref (np.ndarray): テンプレート画像
        out_path (str): 保存パス
        crop_size (int or None): 出力時に中央クロップするサイズ
    """
    # グレースケール変換
    if len(aligned.shape) == 3:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    if len(ref.shape) == 3:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(aligned, ref)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    overlay = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    overlay[mask == 255] = [0, 0, 255]

    if crop_size is not None:
        overlay = center_crop(overlay, crop_size)

    cv2.imwrite(out_path, overlay)


def save_hog_visual(img, out_path, cell_size=8, block_size=2, orientations=9):
    """
    HOGの特徴マップを保存（視覚化用）

    Args:
        img (np.ndarray): 入力画像（BGR or Grayscale）
        out_path (str): 保存先パス
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, hog_img = hog(
        img,
        orientations=orientations,
        pixels_per_cell=(cell_size, cell_size),
        cells_per_block=(block_size, block_size),
        visualize=True,
        feature_vector=True
    )
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    hog_img = (hog_img * 255).astype("uint8")
    cv2.imwrite(out_path, hog_img)