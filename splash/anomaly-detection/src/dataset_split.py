#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def dataset_split(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    train: OK 80
    val: OK 10 + NG 50
    test: OK 10 + NG 50
    すべて train_test_split を使ってランダム＆再現性ありで分割
    """
    os.makedirs(out_dir, exist_ok=True)

    # OKとNGを分ける
    ok_df = df[df["label"] == 0].reset_index(drop=True)
    ng_df = df[df["label"] == 1].reset_index(drop=True)

    # OK: 80 train, 20 val_test に分割
    ok_train, ok_val_test = train_test_split(ok_df, test_size=20, random_state=RANDOM_STATE, shuffle=True)
    # OK: 10 val, 10 test
    ok_val, ok_test = train_test_split(ok_val_test, test_size=0.5, random_state=RANDOM_STATE, shuffle=True)

    # NG: 50 val, 50 test
    ng_val, ng_test = train_test_split(ng_df, test_size=0.5, random_state=RANDOM_STATE, shuffle=True)

    # 結合して保存
    ok_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    pd.concat([ok_val, ng_val]).to_csv(os.path.join(out_dir, "val.csv"), index=False)
    pd.concat([ok_test, ng_test]).to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print("\u2705 データ分割完了:")
    print(f"  Train: {len(ok_train)} OK")
    print(f"  Val:   {len(ok_val)} OK + {len(ng_val)} NG")
    print(f"  Test:  {len(ok_test)} OK + {len(ng_test)} NG")

    return ng_val  # 閾値決定に使う異常サンプル（val用）


# if __name__ == "__main__":
#     input_csv = "data/data_20250326.csv"
#     output_dir = "data/split_20250326"

#     df = pd.read_csv(input_csv)
#     dataset_split(df, output_dir)

