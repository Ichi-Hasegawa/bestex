#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance


def anomaly_score(feat, normal_feats):
    """
    入力特徴量が正常特徴量群からどれだけ離れているかを計算

    Args:
        feat (np.ndarray): 対象画像の特徴ベクトル（1次元）
        normal_feats (list of np.ndarray): 正常画像の特徴ベクトル群

    Returns:
        float: 異常スコア（距離ベース）
    """
    if len(normal_feats) == 0:
        raise ValueError("正常特徴量が空です")

    # 正常データとのユークリッド距離を計算
    dists = [distance.euclidean(feat, ref) for ref in normal_feats]
    return float(np.min(dists))  # 最も近い正常サンプルとの距離を異常スコアとする