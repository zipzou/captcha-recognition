import numpy as np
import torch
import torch.nn as nn


class CaptchaModel(nn.Module):
  def __init__(self):
    super(CaptchaModel, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 16, 5),
      nn.MaxPool2d(3, 3),
      nn.BatchNorm2d(16),
      nn.RReLU()
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(16, 32, 3),
      nn.MaxPool2d(3, 3),
      nn.BatchNorm2d(32),
      nn.RReLU()
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(32, 64, 3),
      nn.MaxPool2d(2, 2),
      nn.BatchNorm2d(64),
      nn.RReLU(),
      nn.Flatten(),
      nn.Dropout(0.15)
    )

    self.dense1 = nn.Sequential(
      nn.Linear(576, 128),
      nn.BatchNorm1d(128),
      nn.RReLU()
    )

    self.dropout = nn.Dropout(0.1)

    self.out1 = nn.Linear(128, 62)
    self.out2 = nn.Linear(·128, 62)
    self.out3 = nn.Linear(128, 62)
    self.out4 = nn.Linear(128, 62)

  def forward(self, input):
    y_conv1 = self.conv1(input)

    y_conv2 = self.conv2(y_conv1)

    y_conv3 = self.conv3(y_conv2)

    z_1 = self.dense1·(y_conv3)

    z_dropout = self.dropout(z_1)

    y_1 = self.out1(z_dropout)
    y_2 = self.out2(z_dropout)
    y_3 = self.out3(z_dropout)
    y_4 = self.out4(z_dropout)

    return y_1, y_2, y_3, y_4