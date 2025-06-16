#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Early stopping
EARLY_STOPPING_ITER_MAX = 50
EPOCH_MIN = 10


# train
def train_model(net, dataloaders_dict, scheduler, criterion, optimizer, num_epochs, expt_fold, log_dir, device):

    # Tensorboard
    log_writer = SummaryWriter(log_dir="{}".format(log_dir))

    # History
    best_loss = 1
    history = {"train_loss": [], "val_loss": []}

    # Move to GPU
    net = net.to(device)

    early_stopping_count = 0

    for epoch in tqdm(range(num_epochs)):

        if early_stopping_count > EARLY_STOPPING_ITER_MAX:
            print("Early Stopping")
            break

        for phase in ["train", "val"]:

            net.train() if phase == "train" else net.eval()

            epoch_loss = 0.0

            if (epoch == 0) and (phase == "train"):
                history["train_loss"].append(1.0)
                continue

            for inputs, _, _ in dataloaders_dict[phase]:
                # inputs.shape: (B, N, 3, H, W) → reshape to (B*N, 3, H, W)
                B, N, C, H, W = inputs.shape
                inputs = inputs.view(B * N, C, H, W).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(inputs, outputs)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * (B * N)

            epoch_loss = epoch_loss / (len(dataloaders_dict[phase].dataset) * N)
            print("Epoch {}/{} | {:^5} | Loss: {:.4f}".format(epoch + 1, num_epochs, phase, epoch_loss))

            log_writer.add_scalar("{}/loss".format(phase), epoch_loss, epoch)

            if phase == "train":
                log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
                scheduler.step()

            if phase == "val" and best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(net, "{}/best_model.pth".format(expt_fold))
                print("Updated Best model! : loss : {}→{}".format(best_loss, epoch_loss))

            if (phase == "val") and (EPOCH_MIN < epoch) and (best_loss < epoch_loss):
                early_stopping_count += 1

            history["{}_loss".format(phase)].append(epoch_loss)

    # Save history
    df = pd.DataFrame(columns=["train_loss", "val_loss"])
    for phase in ["train", "val"]:
        df["{}_loss".format(phase)] = history["{}_loss".format(phase)]
    df.to_csv("{}/history.csv".format(expt_fold), index=False, header=True)

    # Save loss graph
    df = pd.read_csv("{}/history.csv".format(expt_fold))
    plt.plot(df["train_loss"], label="train_loss")
    plt.plot(df["val_loss"], label="val_loss")
    plt.legend()
    plt.savefig("{}/loss.png".format(expt_fold))
    plt.close()
