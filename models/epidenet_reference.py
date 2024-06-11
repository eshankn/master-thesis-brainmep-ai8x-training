# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import torch.nn as nn
import torch.nn.functional as F

# import your own module


class EpiDeNetReference(nn.Module):
    """
    NOTE: This model is not compatible with the MAX78000. It serves as a
    reference for compatible implementations. This model can be trained using
    train.py.
    """
    def __init__(self, **kwargs):
        super().__init__()
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=4,
        #                       kernel_size=(1, 4), stride=(1, 1),
        #                       padding='same')
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4,
                               kernel_size=(1, 4), stride=(1, 1),
                               padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8))
        #self.conv2 = nn.Conv2d(in_channels=4, out_channels=16,
        #                       kernel_size=(1, 16), stride=(1, 1),
        #                       padding='same')
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=(1, 16), stride=(1, 1),
                               padding=(0, 8))
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        #self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
        #                       kernel_size=(1, 8), stride=(1, 1),
        #                       padding='same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=(1, 8), stride=(1, 1),
                               padding=(0, 4))
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        #self.conv4 = nn.Conv2d(in_channels=16, out_channels=16,
        #                       kernel_size=(16, 1), stride=(1, 1),
        #                       padding='same')
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=(16, 1), stride=(1, 1),
                               padding=(8, 0))
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        #self.conv5 = nn.Conv2d(in_channels=16, out_channels=16,
        #                       kernel_size=(8, 1), stride=(1, 1),
        #                       padding='same')
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=(8, 1), stride=(1, 1),
                               padding=(4, 0))
        self.bn5 = nn.BatchNorm2d(16)
        self.pool6 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        #self.fcn = nn.Linear(16, 2)
        self.fcn = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool6(x)
        x = self.flatten(x)
        x = self.fcn(x)
        return x


def epidenet_reference(pretrained=False, **kwargs):
    """
    Constructs a EpiDeNetReference model.
    """
    assert not pretrained
    return EpiDeNetReference(**kwargs)


models = [
    {
        'name': 'epidenet_reference',
        'min_input': 1,
        'dim': 2,
    },
]