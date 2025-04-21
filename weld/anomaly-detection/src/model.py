#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torchvision.models as models


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResNet50AutoEncoder(nn.Module):
    def __init__(self, latent_dim=256, height=1024, width=1024):  # height=1024, width=2304):
        super(ResNet50AutoEncoder, self).__init__()

        self.height = height
        self.width = width

        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, latent_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * (height // 32) * (width // 32)),
            nn.ReLU(True),
            nn.Unflatten(1, (512, height // 32, width // 32)),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # print("input", x.shape)
        x = self.encoder(x)
        # print("encoder", x.shape)
        x = self.decoder(x)
        # print("decoder", x.shape)
        return x
