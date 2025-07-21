#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def rigid_registration(
    src, ref, n_feat=500, keep=0.15
):
    """
    ORB + Homography 

    Args:
        src (np.ndarray): 入力画像
        ref (np.ndarray): 基準画像
        n_feat (int): ORB特徴点の最大数
        keep (float): マッチの上位割合（0〜1）

    Returns:
        np.ndarray: 位置合わせ後の画像
    """
    src_gray = src if len(src.shape) == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ref_gray = ref if len(ref.shape) == 2 else cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_feat)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(src_gray, None)

    if des1 is None or des2 is None:
        raise ValueError("特徴点が見つかりません")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)
    good = matches[:int(len(matches) * keep)]

    if len(good) < 4:
        raise ValueError("十分なマッチがありません")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h, w = ref.shape[:2]
    reg = cv2.warpPerspective(src, H, (w, h))

    return reg
