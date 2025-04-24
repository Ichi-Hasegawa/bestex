#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from src.dataset import img_crop, img_crop2, mask_processing


def get_threshold(anomaly_val_data, out_dir, criterion, device, crop_flag, mask_flag):
    print(f"[DEBUG] anomaly_val_data rows: {len(anomaly_val_data)}")

    #best_model = torch.load(f"{out_dir}/best_model.pth")
    best_model = torch.load(f"{out_dir}/best_model.pth", weights_only=False)

    best_model.eval()

    ano_scores = []
    
    for i in range(len(anomaly_val_data)):
        img_path = anomaly_val_data["path"].values[i]
        print(f"[DEBUG] Processing image: {img_path}", flush=True)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Failed to read image: {img_path}", flush=True)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if crop_flag == "on":
            img = img_crop2(img)
        #elif crop_flag == "off":
            #img = img_crop(img)
        if mask_flag == "on":
            img = mask_processing(img)

        # 💡 resize は transpose より前！
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

        img = img.astype("float32") / 255.0
        img = np.clip(img, 0, 1)
        img = img.transpose(2, 0, 1)

        inputs = torch.tensor(img, dtype=torch.float32).to(device).unsqueeze(0)
        outputs = best_model(inputs).to(device)
        anomaly_score = criterion(outputs, inputs)

        ano_scores.append(anomaly_score.item())

    return min(ano_scores)
