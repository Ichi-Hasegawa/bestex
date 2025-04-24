#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # For headless environments (e.g., SSH server)


def heatmap(inputs, outputs, path, out_dir):
    # Tensor -> Numpy -> (H, W, C)
    inputs = inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    outputs = outputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)

    inputs = np.clip(inputs, 0, 1)
    outputs = np.clip(outputs, 0, 1)

    diff = np.abs(inputs - outputs)
    if diff.max() != diff.min():
        diff_img = (diff - diff.min()) / (diff.max() - diff.min())
    else:
        diff_img = np.zeros_like(diff[..., 0])  # 単色ヒートマップにしておく

    filename = os.path.splitext(os.path.basename(path))[0]  # "foo.png" -> "foo"

    # 入力画像保存
    inputs_dir = os.path.join(out_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    plt.axis("off")
    plt.imshow(inputs)
    plt.savefig(os.path.join(inputs_dir, f"{filename}.png"), bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()

    # 出力画像保存
    outputs_dir = os.path.join(out_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    plt.axis("off")
    plt.imshow(outputs)
    plt.savefig(os.path.join(outputs_dir, f"{filename}.png"), bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()

    # 差分ヒートマップ保存
    diff_dir = os.path.join(out_dir, "diff")
    os.makedirs(diff_dir, exist_ok=True)
    plt.axis("off")
    plt.imshow(diff_img, cmap="magma")
    plt.savefig(os.path.join(diff_dir, f"{filename}.png"), bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()
