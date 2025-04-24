#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def histogram(anomaly_score_list: list[str], normal_score_list: list[str], threshold: float, out_dir: str) -> None:

    all_scores = anomaly_score_list + normal_score_list
    min_score, max_score = min(all_scores), max(all_scores)

    plt.figure(figsize=(10, 6))
    plt.title("Anomaly Score Hist")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")

    bin_wise = 100

    plt.hist(anomaly_score_list, bins=bin_wise, range=(min_score, max_score), alpha=0.3, histtype="stepfilled", color="red", label="Anomaly")
    plt.hist(normal_score_list, bins=bin_wise, range=(min_score, max_score), alpha=0.3, histtype="stepfilled", color="blue", label="Normal")

    plt.axvline(threshold, color="green", linestyle="--", label="Threshold {0:.5f}".format(threshold))
    plt.legend()
    plt.savefig(out_dir + "histogram.png", dpi=300)
    plt.clf()
    plt.close()


# def rank_histogram(anomaly_dict, normal_dict, threshold: float, out_dir: str) -> None:
#     anomaly_scores = anomaly_dict["anomaly_scores"]
#     anomaly_ranks = anomaly_dict["anomaly_ranks"]
#     normal_scores = normal_dict["normal_scores"]

#     # 全スコアの範囲を計算
#     all_scores = anomaly_scores + normal_scores
#     min_score, max_score = min(all_scores), max(all_scores)

#     # ランクごとの色マッピング
#     anomaly_colors = {1: "purple", 2: "orange", 3: "black"}
#     normal_color = "blue"
#     nan_color = "red"

#     # ヒストグラムプロット設定
#     plt.figure(figsize=(10, 6))
#     plt.title("Anomaly Score Histogram by Rank Score")
#     plt.xlabel("Anomaly Score")
#     plt.ylabel("Frequency")
#     bin_wise = 100

#     # Normalデータのプロット
#     plt.hist(
#         normal_scores,
#         bins=bin_wise,
#         range=(min_score, max_score),
#         alpha=0.3,
#         histtype="stepfilled",
#         color=normal_color,
#         label="Normal Rank 0",
#     )

#     # Anomalyデータのランクごとのプロット
#     for rank, color in anomaly_colors.items():
#         rank_scores = [score for score, r in zip(anomaly_scores, anomaly_ranks) if r == rank]
#         plt.hist(
#             rank_scores,
#             bins=bin_wise,
#             range=(min_score, max_score),
#             alpha=0.5,
#             histtype="stepfilled",
#             color=color,
#             label=f"Anomaly Rank {rank}",
#         )

#     # nanランクのスコアのプロット
#     nan_scores = [score for score, r in zip(anomaly_scores, anomaly_ranks) if r is None or np.isnan(r)]
#     if nan_scores:
#         plt.hist(
#             nan_scores,
#             bins=bin_wise,
#             range=(min_score, max_score),
#             alpha=0.3,
#             histtype="stepfilled",
#             color=nan_color,
#             label="Anomaly Rank nan",
#         )

#     # 閾値のライン
#     plt.axvline(threshold, color="green", linestyle="--", label=f"Threshold {threshold:.5f}")

#     # 凡例と保存
#     plt.legend()
#     plt.savefig(out_dir + "rank_histogram.png", dpi=300)
#     plt.clf()
#     plt.close()
