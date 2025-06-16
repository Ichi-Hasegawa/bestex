#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from src.loader import load_image_paths
from src.registration import align_to_template, overlay_images, crop_center
from src.edge_filter import extract_edges
from src.classifier import compute_anomaly_score, save_results

def plot_score_distribution(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # 散布図
    plt.figure(figsize=(8, 4))
    colors = {'OK': 'blue', 'NG': 'red'}
    for label in ['OK', 'NG']:
        subset = df[df['label'] == label]
        plt.scatter(range(len(subset)), subset['score'], label=label, c=colors[label], alpha=0.6)

    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Scatter Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_score_scatter.png"))
    plt.close()

    # ヒストグラム
    plt.figure(figsize=(8, 4))
    for label in ['OK', 'NG']:
        subset = df[df['label'] == label]
        plt.hist(subset['score'], bins=30, alpha=0.5, label=label, color=colors[label])

    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Anomaly Score Histogram")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_score_histogram.png"))
    plt.close()

def main():
    # パス設定
    base_dir = "/net/nfs3/export/home/hasegawa/workspace/bestex/splash/contour-detection"
    input_dir = os.path.join(base_dir, "data/20250326")
    output_dir = os.path.join(base_dir, "_out")
    template_path = os.path.join(input_dir, "OK", "1.png")

    # データ読み込み
    ok_paths, ng_paths = load_image_paths(input_dir)

    # テンプレート画像読み込み・位置合わせ・エッジ抽出
    print("[INFO] テンプレート読み込み中...")
    template_aligned = align_to_template(template_path, template_path)
    template_aligned = crop_center(template_aligned, crop_border=300)
    template_edge = extract_edges(template_aligned)

    all_results = []

    for label, paths in [("OK", ok_paths), ("NG", ng_paths)]:
        for img_path in paths:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[INFO] 処理中: {label}/{fname}")

            try:
                # 位置合わせ
                aligned_img = align_to_template(img_path, template_path)
                aligned_img = crop_center(aligned_img, crop_border=300)
                # オーバーレイ保存（位置確認用）
                template_for_overlay = crop_center(cv2.imread(template_path, 0), crop_border=300)
                overlay = overlay_images(template_for_overlay, aligned_img)
                overlay_dir = os.path.join(output_dir, label, "align_overlay")
                os.makedirs(overlay_dir, exist_ok=True)
                overlay_path = os.path.join(overlay_dir, f"{fname}_overlay.png")
                cv2.imwrite(overlay_path, overlay)
                # 差分画像保存
                raw_diff = cv2.absdiff(aligned_img, template_aligned)
                raw_diff_dir = os.path.join(output_dir, label, "raw_diff")
                os.makedirs(raw_diff_dir, exist_ok=True)
                raw_diff_path = os.path.join(raw_diff_dir, f"{fname}_rawdiff.png")
                cv2.imwrite(raw_diff_path, raw_diff)
                # エッジ抽出
                edge_img = extract_edges(aligned_img)
                edge_dir = os.path.join(output_dir, label, "edge")
                os.makedirs(edge_dir, exist_ok=True)
                edge_path = os.path.join(edge_dir, f"{fname}_edge.png")
                cv2.imwrite(edge_path, edge_img)

                # スコア計算（テンプレートのエッジと比較）
                score = compute_anomaly_score(edge_img, template_edge)

                # 差分マスク保存（エッジ差分）
                diff = cv2.absdiff(edge_img, template_edge)
                diff_dir = os.path.join(output_dir, label, "diff_mask")
                os.makedirs(diff_dir, exist_ok=True)
                diff_path = os.path.join(diff_dir, f"{fname}_diff.png")
                cv2.imwrite(diff_path, diff)

                # エッジ保存＋CSV用記録
                result = save_results(img_path, label, edge_img, score, output_dir)
                all_results.append(result)

            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

    # 結果CSV出力
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "score", "saved_path"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"[INFO] 全処理完了: 結果 → {csv_path}")

    # スコア可視化
    plot_score_distribution(csv_path, output_dir)
    print(f"[INFO] スコア可視化完了（scatter, histogram） → {output_dir}")

if __name__ == "__main__":
    main()



