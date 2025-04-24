#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd

def make_csv(ok_dir, ng_dir, output_csv):
    data = []

    # OK画像
    ok_paths = sorted(glob.glob(os.path.join(ok_dir, "*.png")))
    for path in ok_paths:
        data.append([os.path.abspath(path), 0])

    # NG画像
    ng_paths = sorted(glob.glob(os.path.join(ng_dir, "*.png")))
    for path in ng_paths:
        data.append([os.path.abspath(path), 1])

    # CSV保存
    df = pd.DataFrame(data, columns=["path", "label"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV saved to: {output_csv}  ({len(df)} samples)")
    print(f"Found {len(ok_paths)} OK images.")
    print(f"Found {len(ng_paths)} NG images.")  


if __name__ == "__main__":
    # directory = "/net/nfs3/export/dataset/morita/tlo/bestex-splash/"
    ok_dir = "/net/nfs3/export/dataset/morita/tlo/bestex-splashguard/20250326/OK"
    ng_dir = "/net/nfs3/export/dataset/morita/tlo/bestex-splashguard/20250326/NG"
    output_csv = "data/data_20250326.csv"


    make_csv(ok_dir, ng_dir, output_csv)
