#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import os
import cv2
import numpy as np
import pandas as pd
from src.registration import rigid_registration
from torch.utils.data import Dataset


# Cropping Image (off)
def img_crop(img, top=256):
    """Crop Time Message
    Args:
        img (_type_): _description_
        top (int, optional): _description_. Defaults to 128.

    Returns:
        _type_: _description_
    """

    img = img[top : img.shape[0], 0 : img.shape[1]]  # (1792, 3072, 3)
    return img


def img_crop2(img, top=472, bottom=1496, left=768):
    """Crop Weld Only (Defined by User)
    Args:
        img (_type_): _description_
        top (int, optional): _description_. Defaults to 472.
        bottom (int, optional): _description_. Defaults to 1496.
        left (int, optional): _description_. Defaults to 768.

    Returns:
        img (_type_): _description_
    """

    img = img[top:bottom, left : img.shape[1]]  # (1024, 2304, 3)
    # img = cv2.resize(img, (1024, 1024))

    return img

#off
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


# Data Augmentation(回転)
def rotate_img(img, angle: int):

    angle_y = random.uniform(-1 * angle, angle)  # 縦軸を軸とした回転角度
    angle_x = random.uniform(-1 * angle, angle)  # 横軸を軸とした回転角度
    angle_center = random.uniform(-1 * angle, angle)  # 重心を軸とした回転角度

    # 画像のサイズを取得
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # 1. 縦軸を軸とした回転（左右方向の回転、Y軸）
    rotation_matrix_y = cv2.getRotationMatrix2D(center, angle_y, 1.0)
    img_y_rotated = cv2.warpAffine(img, rotation_matrix_y, (width, height))

    # 2. 横軸を軸とした回転（上下方向の回転、X軸）
    rotation_matrix_x = cv2.getRotationMatrix2D(center, angle_x, 1.0)
    img_x_rotated = cv2.warpAffine(img_y_rotated, rotation_matrix_x, (width, height))

    # 3. 画像の重心を軸とした回転
    rotation_matrix_center = cv2.getRotationMatrix2D(center, angle_center, 1.0)
    rotated_img = cv2.warpAffine(img_x_rotated, rotation_matrix_center, (width, height))

    return rotated_img


# Data Augmentation(左右移動)
def shift_img(img, shift_range: int):

    shift_x = random.randint(-1 * shift_range, shift_range)
    shift_y = random.randint(-1 * shift_range, shift_range)

    height, width = img.shape[:2]
    shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(img, shift_matrix, (width, height))

    return shifted_img


class MyDataset(Dataset):

    def __init__(self, csv_file, crop_flag, mask_flag, rotate_flag, rotate_angle, shift_flag, shift_range):

        self.data = pd.read_csv(csv_file)

        if crop_flag == "on":
            self.img_crop = img_crop2
        elif crop_flag == "off":
            self.img_crop = img_crop
        else:
            raise ValueError("crop_flag is invalid.")

        if rotate_flag == "on":
            self.rotate_img = rotate_img
        elif rotate_flag == "off":
            self.rotate_img = None
        else:
            raise ValueError("rotate_flag is invalid.")

        if shift_flag == "on":
            self.shift_img = shift_img
        elif shift_flag == "off":
            self.shift_img = None
        else:
            raise ValueError("shift_flag is invalid.")

        if mask_flag == "on":
            self.mask_img = mask_processing
        elif mask_flag == "off":
            self.mask_img = None

        self.rotate_angle = rotate_angle
        self.shift_range = shift_range

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        img_path = self.data.iloc[item, 0]
        label = self.data.iloc[item, 1]
        # rank = self.data.iloc[item, 2]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"[ERROR] Failed to read image: {img_path}")
        
        # image registration
        #ref_img = cv2.imread("/net/nfs3/export/dataset/morita/tlo/bestex-weld2/20240514/2024-05-14_20.59.21.721(DA2143910).png")
        #float_img = cv2.imread(img_path)

        #img = rigid_registration(ref_img, float_img, scale_factor=0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        
        # Crop
        img = self.img_crop(img)

        # Mask
        if self.mask_img is not None:
            img = self.mask_img(img)

        # Random Shift
        if self.shift_img is not None:
            img = self.shift_img(img, self.shift_range)

        # Random Rotation
        if self.rotate_img is not None:
            img = self.rotate_img(img, self.rotate_angle)

        img = img.astype("float32") / 255.0  # float32
        img = np.clip(img, 0, 1)  # 0-1
        img = img.transpose(2, 0, 1)  # (ch, h, w)

        #return img, label, rank, img_path
        return img, label, img_path
