#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # ssh server


def heatmap(inputs, outputs, path, out_dir):

    # (1, ch, h, w) -> (h, w, ch)
    inputs = inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    outputs = outputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)

    # 0-1(float32)
    inputs = np.clip(inputs, 0, 1)
    outputs = np.clip(outputs, 0, 1)

    # 0-1(float32)
    diff = np.abs(inputs - outputs)
    diff_img = (diff - diff.min()) / (diff.max() - diff.min())

    # path
    path = path.split("/")[-1].split(".png")[0]

    # inputs
    plt.axis("off")
    plt.imshow(inputs)

    if not os.path.exists(out_dir + "inputs/"):
        os.makedirs(out_dir + "inputs/")

    plt.savefig(out_dir + "inputs/{}.png".format(path), bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()

    # outputs
    plt.axis("off")
    plt.imshow(outputs)

    if not os.path.exists(out_dir + "outputs/"):
        os.makedirs(out_dir + "outputs/")

    plt.savefig(out_dir + "outputs/{}.png".format(path), bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()

    # diff
    plt.axis("off")
    plt.imshow(diff_img, cmap="magma")

    if not os.path.exists(out_dir + "diff/"):
        os.makedirs(out_dir + "diff/")

    plt.savefig(out_dir + "diff/{}.png".format(path), bbox_inches="tight", pad_inches=0)

    plt.clf()
    plt.close()
