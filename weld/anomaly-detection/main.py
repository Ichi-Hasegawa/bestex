#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import pandas as pd
import torch
from src.anomaly_score import get_threshold
from src.dataset import MyDataset
from src.dataset_split import dataset_split
from src.histogram import histogram, rank_histogram
from src.model import ConvAutoEncoder, ResNet50AutoEncoder
from src.test import test_model
from src.train import train_model
from torch.utils.data import DataLoader
from torchinfo import summary

# CPU
NUM_WORKERS = os.cpu_count() // 4

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experimental Settings(parser)
parser = argparse.ArgumentParser(description="Experimental Settings", epilog="End description !!")

parser.add_argument("--crop_flag", type=str, default="on", help="crop flag")
parser.add_argument("--mask_flag", type=str, default="on", help="mask flag")  # setting for img_crop2
parser.add_argument("--rotate_flag", type=str, default="on", help="rotate flag")
parser.add_argument("--rotate_angle", type=int, default=1, help="rotate angle")
parser.add_argument("--shift_flag", type=str, default="on", help="shift flag")
parser.add_argument("--shift_range", type=int, default=1, help="shift range")
parser.add_argument("--csv_path", type=str, default="data/data_250325.csv", help="csv path")

parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--epochs", type=int, default=1000, help="epochs")

args = parser.parse_args()


# Experimental Dir Settings
dl_model = "conv_autoencoder"
exp_name = f"crop-{args.crop_flag}_mask-{args.mask_flag}_rotate-{args.rotate_flag}({args.rotate_angle})_shift-{args.shift_flag}({args.shift_range})_{dl_model}/"
out_dir = "result/" + exp_name
log_dir = "log/" + exp_name


# Dataset Setting
def exp_setting():
    df = pd.read_csv(args.csv_path)
    anomaly_val_data = dataset_split(df, out_dir)
    return anomaly_val_data  # anomaly_val_data -> Determine threshold using anomaly data


# Train
def train(anomaly_val_data):

    train_dataset = MyDataset(
        out_dir + "train.csv",
        crop_flag=args.crop_flag,
        mask_flag=args.mask_flag,
        rotate_flag=args.rotate_flag,
        rotate_angle=args.rotate_angle,
        shift_flag=args.shift_flag,
        shift_range=args.shift_range,
    )
    val_dataset = MyDataset(
        out_dir + "val.csv",
        crop_flag=args.crop_flag,
        mask_flag=args.mask_flag,
        rotate_flag=args.rotate_flag,
        rotate_angle=args.rotate_angle,
        shift_flag=args.shift_flag,
        shift_range=args.shift_range,
    )

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS, shuffle=True, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS, shuffle=False, drop_last=False, pin_memory=True)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
    
    net = ConvAutoEncoder().to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.50)

    train_model(net, dataloaders_dict, scheduler, criterion, optimizer, num_epochs=args.epochs, expt_fold=out_dir, log_dir=log_dir, device=device)

    # Threshoold Setting Using anomaly_val_data
    threshold = get_threshold(anomaly_val_data, out_dir, criterion, device=device, crop_flag=args.crop_flag, mask_flag=args.mask_flag)
    df = pd.DataFrame([threshold], columns=["threshold"])
    df.to_csv(out_dir + "threshold.csv", index=False)

    return threshold


# Test
def test(threshold):

    test_dataset = MyDataset(
        out_dir + "test.csv",
        crop_flag=args.crop_flag,
        mask_flag=args.mask_flag,
        rotate_flag=args.rotate_flag,
        rotate_angle=args.rotate_angle,
        shift_flag=args.shift_flag,
        shift_range=args.shift_range,
    )

    # Dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS, shuffle=False, drop_last=False, pin_memory=True)

    criterion = torch.nn.MSELoss()

    anomaly_dict, normal_dict = test_model(test_dataloader, expt_fold=out_dir, criterion=criterion, device=device, threshold=threshold)

    # Test Result Visualization
    histogram(anomaly_dict["anomaly_scores"], normal_dict["normal_scores"], threshold, out_dir)
    rank_histogram(anomaly_dict, normal_dict, threshold, out_dir)


def main():

    # Experiment setting
    anomaly_val_data = exp_setting()

    # Training
    threshold = train(anomaly_val_data)

    # Testing
    test(threshold)


if __name__ == "__main__":
    main()
