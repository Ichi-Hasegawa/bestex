#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.heatmap import heatmap


def test_model(test_dataloader, expt_fold, criterion, device, threshold, patch_split):
    """Test
    Args:
        test_dataloader: DataLoader yielding (N_patches, 3, H, W)
        expt_fold: experiment output directory
        criterion: loss function (e.g., MSELoss)
        device: torch.device
        threshold: float for anomaly decision
        patch_split: int (e.g., 5 for 5x5 split)

    Returns:
        anomaly_dict, normal_dict
    """

    best_model = torch.load(f"{expt_fold}/best_model.pth", weights_only=False)
    best_model.eval()

    anomalies = []
    anomaly_scores = []
    normals = []
    normal_scores = []

    true_labels = []
    pred_labels = []
    scores = []

    with torch.no_grad():
        for inputs, labels, path in test_dataloader:
            # inputs: (B=1, N, 3, H, W) → reshape to (N, 3, H, W)
            if inputs.dim() == 5:
                inputs = inputs.squeeze(0)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = best_model(inputs)
            diff = criterion(outputs, inputs).item()  # 平均MSEでスコア化

            scores.append(diff)
            true_labels.append(labels.item())
            pred_labels.append(1 if diff > threshold else 0)

            # パッチ群を1枚の画像に再構成
            C, H, W = inputs.shape[1:]

            def stitch(patches):
                rows = []
                for i in range(patch_split):
                    row = torch.cat([patches[i * patch_split + j] for j in range(patch_split)], dim=2)
                    rows.append(row)
                return torch.cat(rows, dim=1)  # (C, H×patch_split, W×patch_split)

            stitched_input = stitch(inputs.cpu())
            stitched_output = stitch(outputs.cpu())

            # Heatmap保存
            heatmap(stitched_input, stitched_output, path[0], out_dir=expt_fold)

            # スコア記録
            if labels.item() == 1:
                anomalies.append(path[0])
                anomaly_scores.append(diff)
            else:
                normals.append(path[0])
                normal_scores.append(diff)

    # Metrics保存
    metrics = {
        "Accuracy": accuracy_score(true_labels, pred_labels),
        "Precision": precision_score(true_labels, pred_labels),
        "Recall": recall_score(true_labels, pred_labels),
        "ROC-AUC": roc_auc_score(true_labels, scores),
        "F1 Score": f1_score(true_labels, pred_labels),
    }
    pd.DataFrame(metrics, index=[0]).to_csv(f"{expt_fold}/metrics.csv", index=False)

    # スコアCSV保存
    pd.DataFrame({"anomalies": anomalies, "anomaly_scores": anomaly_scores}).to_csv(f"{expt_fold}/anomalies.csv", index=False)
    pd.DataFrame({"normals": normals, "normal_scores": normal_scores}).to_csv(f"{expt_fold}/normals.csv", index=False)

    return {"anomalies": anomalies, "anomaly_scores": anomaly_scores}, {"normals": normals, "normal_scores": normal_scores}


