#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sift_registration(reference_img, floating_img):

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(reference_img, None)
    kp2, des2 = sift.detectAndCompute(floating_img, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Get the good matches
    good_matches = matches[:1000]

    # Get the source and destination points
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width, channels = reference_img.shape
    aligned_img = cv2.warpPerspective(floating_img, H, (width, height))

    return aligned_img


def rigid_registration(reference_img, floating_img, scale_factor):

    reference_gray_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    floating_gray_img = cv2.cvtColor(floating_img, cv2.COLOR_BGR2GRAY)

    reference_resized = cv2.resize(reference_gray_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    floating_resized = cv2.resize(floating_gray_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    shift, _ = cv2.phaseCorrelate(np.float32(reference_resized), np.float32(floating_resized))
    dx, dy = shift

    dx /= scale_factor
    dy /= scale_factor

    rows, cols = reference_gray_img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    aligned_img = cv2.warpAffine(floating_img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return aligned_img


def visualize_image_difference(img1, img2, output_path=None):

    print(img1.shape, img2.shape)
    if img1.shape != img2.shape:
        raise ValueError("The input images must have the same dimensions.")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.absdiff(img1, img2)
    # diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    plt.figure(dpi=500)
    sns.heatmap(diff_gray, cmap="viridis", cbar=True)
    plt.title("Image Difference Heatmap")

    if output_path:
        plt.savefig(output_path)


def calculate_rmse(image1, image2):

    if image1.shape != image2.shape:
        raise ValueError("画像の形状が一致していません。")

    error = image1.astype(np.float32) - image2.astype(np.float32)
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)

    return rmse


if __name__ == "__main__":
    reference_img = cv2.imread("/net/nfs3/export/dataset/morita/tlo/bestex-weld2/20240426/2024-04-26_15.57.20.349(DA2143910).png")
    # floating_img = cv2.imread("/net/nfs3/export/dataset/morita/tlo/bestex-weld2/20240521/2024-05-21_03.10.44.873(DA2143910).png")
    floating_img = cv2.imread("/net/nfs3/export/dataset/morita/tlo/bestex-weld2/20240906/2024-09-06_12.28.43.978(DA2143910).png")
    # floating_img = cv2.imread("/net/nfs3/export/dataset/morita/tlo/bestex-weld2/20240906/2024-09-06_12.24.19.582(DA2143910).png")

    cv2.imwrite("result/reference_img.png", reference_img)
    cv2.imwrite("result/floating_img.png", floating_img)

    # aligned_img = sift_registration(reference_img, floating_img)
    aligned_img = rigid_registration(reference_img, floating_img, 0.5)

    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    floating_img = cv2.cvtColor(floating_img, cv2.COLOR_BGR2GRAY)
    aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

    print("non-sift RMSE: ", calculate_rmse(reference_img, floating_img))
    print("sift RMSE: ", calculate_rmse(reference_img, aligned_img))
    cv2.imwrite("result/sift_aligned_img.png", aligned_img)
    exit(-1)

    visualize_image_difference(reference_img, aligned_img, output_path="result/diff_heatmap_aligned.png")
    visualize_image_difference(reference_img, floating_img, output_path="result/diff_heatmap_noaligned.png")
