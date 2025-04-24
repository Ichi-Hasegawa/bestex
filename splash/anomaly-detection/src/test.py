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


def test_model(test_dataloader, expt_fold, criterion, device, threshold):
    """Test
    Args:
        test_dataloader (_type_): _description_
        expt_fold (_type_): _description_
        criterion (_type_): _description_
        device (_type_): _description_
        threshold (_type_): _description_

    Returns:
        anomaly_dict: ["anomalies": str, "anomaly_scores": float]
        normal_dict: ["normals": str, "normals_scores": float]
    """

    # Best model
    #best_model = torch.load("{}best_model.pth".format(expt_fold))
    #best_model = torch.load(f"{out_dir}/best_model.pth", weights_only=False)
    # 修正後
    best_model = torch.load(f"{expt_fold}/best_model.pth", weights_only=False)


    best_model.eval()

    anomalies = []
    anomaly_scores = []
    #anomaly_ranks = []
    normals = []
    normal_scores = []
    #normal_ranks = []

    true_labels = []
    pred_labels = []
    scores = []

    with torch.no_grad():

        #for inputs, labels, ranks, path in test_dataloader:
        for inputs, labels, path in test_dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = best_model(inputs).to(device)
            diff = criterion(outputs, inputs)

            scores.append(diff.item())
            true_labels.append(labels.item())
            pred_labels.append(1 if diff.item() > threshold else 0)

            # Save Fig
            heatmap(inputs, outputs, path[0], out_dir=expt_fold)

            if labels.item() == 1:
                anomalies.append(path[0])
                anomaly_scores.append(diff.item())
                #anomaly_ranks.append(ranks.item())
            else:
                normals.append(path[0])
                normal_scores.append(diff.item())
                #normal_ranks.append(ranks.item())

    # Save results to CSV
    metrics = {
        "Accuracy": accuracy_score(true_labels, pred_labels),
        "Precision": precision_score(true_labels, pred_labels),
        "Recall": recall_score(true_labels, pred_labels),
        "ROC-AUC": roc_auc_score(true_labels, scores),
        "F1 Score": f1_score(true_labels, pred_labels),
    }

    df_metrics = pd.DataFrame(metrics, index=[0])
    df_metrics.to_csv(f"{expt_fold}/metrics.csv", index=False)

    #anomaly_dict = {"anomalies": anomalies, "anomaly_scores": anomaly_scores, "anomaly_ranks": anomaly_ranks}
    #normal_dict = {"normals": normals, "normal_scores": normal_scores, "normal_ranks": normal_ranks}
    anomaly_dict = {"anomalies": anomalies, "anomaly_scores": anomaly_scores}
    normal_dict = {"normals": normals, "normal_scores": normal_scores}

    # Save all to CSV
    #df_anomalies = pd.DataFrame(anomaly_dict, columns=["anomalies", "anomaly_scores", "anomaly_ranks"])
    #df_anomalies.to_csv("{}/anomalies.csv".format(expt_fold), index=False)

    #df_normals = pd.DataFrame(normal_dict, columns=["normals", "normal_scores", "normal_ranks"])
    #df_normals.to_csv("{}/normals.csv".format(expt_fold), index=False)

    df_anomalies = pd.DataFrame(anomaly_dict, columns=["anomalies", "anomaly_scores"])
    df_anomalies.to_csv(f"{expt_fold}/anomalies.csv", index=False)

    df_normals = pd.DataFrame(normal_dict, columns=["normals", "normal_scores"])
    df_normals.to_csv(f"{expt_fold}/normals.csv", index=False)

    return anomaly_dict, normal_dict
