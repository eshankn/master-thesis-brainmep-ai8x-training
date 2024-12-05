# ------------------------------------------------------------------------------------------
# Author        : Eshank Jayant Nazare
# File          : epidenet_reference_test.py
# Project       : BrainMEP
# Modified      : 28.11.2024
# Description   : EpiDeNet model reference architecture for first three blocks
# ------------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import torch.nn as nn
import torch.nn.functional as F
import torch

# import your own module


class EpiDeNetReferenceTest(nn.Module):
    """
    NOTE: This model is based on EpiDeNet architecture and only implements the first three blocks. It uses
    1D convolutions for late-channel integration. This model is not compatible with the MAX78000. It serves as a
    reference for compatible implementations. This model can be trained using train.py.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=4, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(4)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)

        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=16, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=1, padding=4)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.pool5 = nn.AvgPool1d(kernel_size=6, stride=1)

        self.flatten = nn.Flatten()
        self.fcn = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fcn(x)
        return x


def epidenet_reference_test(pretrained=False, **kwargs):
    """
    Constructs a EpiDeNetReference model.
    """
    assert not pretrained
    return EpiDeNetReferenceTest(**kwargs)


models = [
    {
        'name': 'epidenet_reference_test',
        'min_input': 1,
        'dim': 2,
    },
]