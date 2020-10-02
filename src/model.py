import numpy as np
import torch
import torch.nn as nn

class CaptchaModel(nn.Module):
  def __init__(self):
    super(CaptchaModel, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 16, 5),
      nn.MaxPool2d(5, 5),
      nn.BatchNorm2d(16),
      nn.RReLU()
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(16, 32, 5),
      nn.MaxPool2d(4, 4),
      nn.BatchNorm2d(32),
      nn.RReLU(),
      nn.Flatten()
    )

    # self.conv3 = nn.Sequential(
    #   nn.Conv2d(32, 32, 3),
    #   nn.MaxPool2d(2, 2),
    #   nn.BatchNorm2d(32),
    #   nn.RReLU(),
    # )

    self.dense1 = nn.Sequential(
      nn.Linear(256, 64),
      nn.BatchNorm1d(64),
      nn.RReLU()
    )

    self.dropout = nn.Dropout(0.3)

    self.out1 = nn.Linear(256, 62)
    self.out2 = nn.Linear(256, 62)
    self.out3 = nn.Linear(256, 62)
    self.out4 = nn.Linear(256, 62)

  def forward(self, input):
    y_conv1 = self.conv1(input)

    y_conv2 = self.conv2(y_conv1)

    # y_conv3 = self.conv3(y_conv2)

    z_1 = self.dense1(y_conv2)

    z_dropout = self.dropout(y_conv2)

    y_1 = self.out1(z_dropout)
    y_2 = self.out2(z_dropout)
    y_3 = self.out3(z_dropout)
    y_4 = self.out4(z_dropout)

    return y_1, y_2, y_3, y_4