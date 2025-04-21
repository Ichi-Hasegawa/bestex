#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas as pd
from sklearn.model_selection import train_test_split

# split settings
RANDOM_STATE = 42

VAL_DATA_SPLIT_RATIO = 0.2  # train:val = 8:2
TEST_DATA_NUM = 200


def dataset_split(df, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    anomalies = df[df["label"] == 1]
    normals = df[df["label"] == 0]

    # 閾値決定用のanomaly_val，その他のanomalies_test
    exclude_strings = ["20240906", "20240509", "20240520"]
    anomaly_val = anomalies[~anomalies["path"].str.contains("|".join(exclude_strings))].sample(n=2, random_state=RANDOM_STATE)

    anomalies_test = anomalies.drop(anomaly_val.index)

    normals_test = normals.sample(n=TEST_DATA_NUM - len(anomalies_test), random_state=RANDOM_STATE)

    test_data = pd.concat([normals_test, anomalies_test])
    test_data.to_csv(out_dir + "test.csv", index=False)

    # test_dataとanomalies_valに含まれないデータを取得
    remaining_data = df[~df["path"].isin(test_data["path"]) & ~df["path"].isin(anomaly_val["path"])]

    normal_train_data, normal_val_data = train_test_split(remaining_data, test_size=0.2, random_state=RANDOM_STATE)

    normal_train_data.to_csv(out_dir + "train.csv", index=False)
    normal_val_data.to_csv(out_dir + "val.csv", index=False)

    return anomaly_val
