import os
import random
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Cropping Image (off)
def img_crop(img, top=256):
    # デフォルトはcropしない
    return img

def img_crop2(img, top=500, bottom=4000, left=580, right=4000):
    img = img[top:bottom, left:right]
    return img

# Mask processing (off)
def mask_processing(img, rectangles=[(0, 500, 600, 850), (650, 500, 1200, 800), (1700, 200, 2304, 600)]):
    mask = np.zeros_like(img, dtype=np.uint8)
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        if mask.ndim == 2:
            raise ValueError("Input image must be a color image.")
        else:
            mask[y1:y2, x1:x2] = (255, 255, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Data Augmentation (rotation)
def rotate_img(img, angle: int):
    angle_y = random.uniform(-angle, angle)
    angle_x = random.uniform(-angle, angle)
    angle_center = random.uniform(-angle, angle)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix_y = cv2.getRotationMatrix2D(center, angle_y, 1.0)
    img_y_rotated = cv2.warpAffine(img, rotation_matrix_y, (width, height))
    rotation_matrix_x = cv2.getRotationMatrix2D(center, angle_x, 1.0)
    img_x_rotated = cv2.warpAffine(img_y_rotated, rotation_matrix_x, (width, height))
    rotation_matrix_center = cv2.getRotationMatrix2D(center, angle_center, 1.0)
    rotated_img = cv2.warpAffine(img_x_rotated, rotation_matrix_center, (width, height))
    return rotated_img

# Data Augmentation (translation)
def shift_img(img, shift_range: int):
    shift_x = random.randint(-shift_range, shift_range)
    shift_y = random.randint(-shift_range, shift_range)
    height, width = img.shape[:2]
    shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(img, shift_matrix, (width, height))
    return shifted_img

# Dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file, crop_flag, mask_flag, rotate_flag, rotate_angle, shift_flag, shift_range, patch_split):
        self.data = pd.read_csv(csv_file)
        self.patch_split = patch_split
        self.printed_image_size = False

        if crop_flag == "on":
            self.img_crop = img_crop2
        elif crop_flag == "off":
            self.img_crop = img_crop
        else:
            raise ValueError("crop_flag is invalid.")

        self.rotate_img = rotate_img if rotate_flag == "on" else None
        self.shift_img = shift_img if shift_flag == "on" else None
        self.mask_img = mask_processing if mask_flag == "on" else None

        self.rotate_angle = rotate_angle
        self.shift_range = shift_range

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        import math

        img_path = self.data.iloc[item, 0]
        label = self.data.iloc[item, 1]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[ERROR] Image path does not exist: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"[ERROR] Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_crop(img)

        if self.mask_img is not None:
            img = self.mask_img(img)
        if self.shift_img is not None:
            img = self.shift_img(img, self.shift_range)
        if self.rotate_img is not None:
            img = self.rotate_img(img, self.rotate_angle)

        # --- サイズ揃え：patch_split × patch_split にリサイズ ---
        height, width = img.shape[:2]
        patch_size_h = math.ceil(height / self.patch_split)
        patch_size_w = math.ceil(width / self.patch_split)

        target_h = patch_size_h * self.patch_split
        target_w = patch_size_w * self.patch_split

        if (target_h != height) or (target_w != width):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # --- 正規化とチャンネル順変換 ---
        img = img.astype("float32") / 255.0
        img = np.clip(img, 0, 1)
        img = img.transpose(2, 0, 1)  # (C, H, W)

        if not self.printed_image_size:
            print(f"[INFO] Input image shape (resized): {img.shape[1]}x{img.shape[2]}")
            self.printed_image_size = True

        # --- パッチ分割 ---
        C, H, W = img.shape
        patch_size_h = H // self.patch_split
        patch_size_w = W // self.patch_split

        patches = []
        for i in range(self.patch_split):
            for j in range(self.patch_split):
                top = i * patch_size_h
                left = j * patch_size_w
                patch = img[:, top:top + patch_size_h, left:left + patch_size_w]
                patches.append(patch)

        img = np.stack(patches)  # → (N, C, h, w)

        return img, label, img_path

