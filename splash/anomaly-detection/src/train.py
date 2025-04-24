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

        # Early stopping
        if early_stopping_count > EARLY_STOPPING_ITER_MAX:
            print("Early Stopping")
            break

        for phase in ["train", "val"]:

            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0

            # If the training is epoch=0, skip the training to check the performance of the unlearned validation.
            if (epoch == 0) and (phase == "train"):
                history["train_loss"].append(1.0)
                continue

            #for inputs, _, _, _ in dataloaders_dict[phase]:
            for inputs, _, _ in dataloaders_dict[phase]:

                inputs = inputs.to(device)

                # Initialized optimizer
                optimizer.zero_grad()

                # Forward computation (when training, calculate the gradient)
                with torch.set_grad_enabled(phase == "train"):

                    outputs = net(inputs).to(device)
                    loss = criterion(inputs, outputs)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            print("Epoch {}/{} | {:^5} | Loss: {:.4f}".format(epoch + 1, num_epochs, phase, epoch_loss))

            # Tensorboard
            log_writer.add_scalar("{}/loss".format(phase), epoch_loss, epoch)

            # Update learning rate
            if phase == "train":
                # scheduler.step(epoch_loss)  # when using ReduceLROnPlateau
                log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
                scheduler.step()  # when using ExponentialLR

            # Save model when validation loss is lower than the previous best loss
            if phase == "val" and best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(net, "{}/best_model.pth".format(expt_fold))
                print("Updated Best model! : loss : {}→{}".format(best_loss, epoch_loss))

            # Early stopping
            if (phase == "val") and (EPOCH_MIN < epoch) and (best_loss < epoch_loss):
                early_stopping_count += 1

            # Save history
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
