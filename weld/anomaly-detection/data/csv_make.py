#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os


def find_png_files(directory: str, exclude_str: str) -> list[str]:
    png_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                file_path = os.path.join(root, file)
                if exclude_str not in file_path:
                    png_files.append(file_path)
    return png_files


def save_to_csv(file_paths: list[str], output_csv: str) -> None:

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label"])
        for path in file_paths:
            if "NG" in path:
                label = 1
            else:
                label = 0
            writer.writerow([path, label])


def main():

    # User settings
    directory = "/net/nfs3/export/dataset/morita/tlo/bestex-weld2/"
    output_csv = "data/data_0108.csv"

    # 取り除くファイルに含まれる文字列
    exclude_str = "溶接ズレ写真"

    png_files = find_png_files(directory, exclude_str)
    save_to_csv(png_files, output_csv)


if __name__ == "__main__":
    main()
