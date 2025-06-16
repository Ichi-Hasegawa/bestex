#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
import math
from src.dataset import img_crop, img_crop2, mask_processing


def split_into_patches(img, patch_split):
    H, W = img.shape[:2]
    patch_h = math.ceil(H / patch_split)
    patch_w = math.ceil(W / patch_split)
    target_h = patch_h * patch_split
    target_w = patch_w * patch_split
    img = cv2.resize(img, (target_w, target_h))

    patches = []
    for i in range(patch_split):
        for j in range(patch_split):
            y1 = i * patch_h
            y2 = (i + 1) * patch_h
            x1 = j * patch_w
            x2 = (j + 1) * patch_w
            patch = img[y1:y2, x1:x2]
            patches.append(patch)
    return patches, (patch_h, patch_w)

def stitch_patches(patches, patch_split):
    rows = []
    for i in range(patch_split):
        row = np.concatenate(patches[i * patch_split:(i + 1) * patch_split], axis=1)
        rows.append(row)
    stitched = np.concatenate(rows, axis=0)
    return stitched

# def get_threshold(anomaly_val_data, out_dir, criterion, device, crop_flag, mask_flag, patch_split):
#     print(f"[DEBUG] anomaly_val_data rows: {len(anomaly_val_data)}")

#     best_model = torch.load(f"{out_dir}/best_model.pth", weights_only=False)
#     best_model.eval()

#     ano_scores = []

#     for i in range(len(anomaly_val_data)):
#         img_path = anomaly_val_data["path"].values[i]
#         print(f"[DEBUG] Processing image: {img_path}", flush=True)

#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[ERROR] Failed to read image: {img_path}", flush=True)
#             continue

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         if crop_flag == "on":
#             img = img_crop2(img)
#         if mask_flag == "on":
#             img = mask_processing(img)

#         patches, (patch_h, patch_w) = split_into_patches(img, patch_split)

#         input_tensors = []
#         for patch in patches:
#             patch = patch.astype("float32") / 255.0
#             patch = np.clip(patch, 0, 1)
#             patch = patch.transpose(2, 0, 1)
#             input_tensors.append(patch)

#         inputs = torch.tensor(np.stack(input_tensors), dtype=torch.float32).to(device)
#         outputs = best_model(inputs).to(device)

#         inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1) * 255
#         outputs_np = outputs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255

#         input_recon = stitch_patches(inputs_np, patch_split)
#         output_recon = stitch_patches(outputs_np, patch_split)

#         input_tensor = torch.tensor(input_recon.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
#         output_tensor = torch.tensor(output_recon.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)

#         score = criterion(output_tensor, input_tensor)
#         ano_scores.append(score.item())

#     return min(ano_scores)


# def get_threshold(anomaly_val_data, out_dir, criterion, device, crop_flag, mask_flag):
#     print(f"[DEBUG] anomaly_val_data rows: {len(anomaly_val_data)}")

#     #best_model = torch.load(f"{out_dir}/best_model.pth")
#     best_model = torch.load(f"{out_dir}/best_model.pth", weights_only=False)

#     best_model.eval()

#     ano_scores = []
    
#     for i in range(len(anomaly_val_data)):
#         img_path = anomaly_val_data["path"].values[i]
#         print(f"[DEBUG] Processing image: {img_path}", flush=True)

#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[ERROR] Failed to read image: {img_path}", flush=True)
#             continue

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         if crop_flag == "on":
#             img = img_crop2(img)
#         #elif crop_flag == "off":
#             #img = img_crop(img)
#         if mask_flag == "on":
#             img = mask_processing(img)

#         # 💡 resize は transpose より前！
#         img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

#         img = img.astype("float32") / 255.0
#         img = np.clip(img, 0, 1)
#         img = img.transpose(2, 0, 1)

#         inputs = torch.tensor(img, dtype=torch.float32).to(device).unsqueeze(0)
#         outputs = best_model(inputs).to(device)
#         anomaly_score = criterion(outputs, inputs)

#         ano_scores.append(anomaly_score.item())

#     return min(ano_scores)

def get_threshold(anomaly_val_data, out_dir, criterion, device, crop_flag, mask_flag, patch_split):
    import math
    print(f"[DEBUG] anomaly_val_data rows: {len(anomaly_val_data)}")

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
        if mask_flag == "on":
            img = mask_processing(img)

        # 分割前の画像を保存（比較用）
        #original_img = cv2.resize(img, None)  # resizeされる前の状態を保持
        original_img = img.copy()
        # --- パッチ分割 ---
        patches, (patch_h, patch_w) = split_into_patches(img, patch_split)

        # --- 入力テンソル化 ---
        input_tensors = []
        for patch in patches:
            patch = patch.astype(np.float32) / 255.0
            patch = patch.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
            input_tensors.append(patch)

        inputs = torch.from_numpy(np.stack(input_tensors)).float().to(device)

        # --- 推論 ---
        with torch.no_grad():
            outputs = best_model(inputs)

        # --- 出力画像結合 ---
        outputs_np = outputs.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        recon_img = stitch_patches(outputs_np, patch_split).astype(np.uint8)

        # --- 入出力比較（分割前画像 vs 結合出力）---
        input_tensor = torch.from_numpy(original_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        output_tensor = torch.from_numpy(recon_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

        score = criterion(output_tensor, input_tensor)
        ano_scores.append(score.item())

    return min(ano_scores)