#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.feature import hog


def hog_feature(img, cell_size=8, block_size=2, orientations=9):
    """
    HOG特徴量を抽出

    Args:
        img (np.ndarray): 入力画像（BGRまたはGrayscale）
        cell_size (int): セルのサイズ（ピクセル）
        block_size (int): ブロックのサイズ（セル数）
        orientations (int): 勾配方向の分割数

    Returns:
        np.ndarray: 1次元のHOG特徴量ベクトル
    """

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
        feature_vector=True
    )
    return feat