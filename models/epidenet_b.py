# ------------------------------------------------------------------------------------------
# Author        : Eshank Jayant Nazare
# File          : epidenet_b.py
# Project       : BrainMEP
# Modified      : 28.11.2024
# Description   : EpiDeNet model architectures for implementation on MAX78000
# ------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

import ai8x


class EpiDeNetB(nn.Module):
    """
    NOTE: This model is the variant B of the EpiDeNet architecture. This variant satisfies the MAX78000 constraints.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # MAX78000 constraint: The maximum dimension (number of rows or columns) for input or output data to be 1023.
        # MAX78000 constraint for synthesis: Padding size should be smaller than kernel size.
        # Input: (B, 1, 768)

        # ----------------------------------------------------------------------------------------------------
        #                                           EpiDeNet_B Variant 1.1
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=3, stride=1,
        #                                                       padding=1)  # Output: (B, 4, 512)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 64)
        # self.block1_maxpool1d_2 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 4, 16)

        # # Block 5
        # self.block5_avgpool1d = ai8x.AvgPool1d(kernel_size=16, stride=1)  # Output: (B, 4, 1)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 4)
        # self.block6_dense = ai8x.Linear(in_features=4, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           EpiDeNet_B Variant 1.2
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=4, stride=1,
        #                                                       padding=2)  # Output: (B, 4, 513)
        # self.block1_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=4, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 512)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 64)
        # self.block1_maxpool1d_2 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 4, 16)

        # # Block 5
        # self.block5_avgpool1d = ai8x.AvgPool1d(kernel_size=16, stride=1)  # Output: (B, 4, 1)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 4)
        # self.block6_dense = ai8x.Linear(in_features=4, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           EpiDeNet_B Variant 2.1
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=4, stride=1,
        #                                                       padding=2)  # Output: (B, 4, 513)
        # self.block1_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=4, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 512)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 64)
        #
        # # Block 2
        # self.block2_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=16, kernel_size=9, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 60)
        # self.block2_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=8, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 57)
        # self.block2_conv1d_bn_relu_3 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=3, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 59)
        # self.block2_conv1d_bn_relu_4 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=3, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 61)
        # self.block2_conv1d_bn_relu_5 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=3, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 63)
        # self.block2_conv1d_bn_relu_6 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=2, stride=1,
        #                                                       padding=1)  # Output: (B, 16, 64)
        # self.block2_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 16, 16)

        # # Block 5
        # self.block5_avgpool1d = ai8x.AvgPool1d(kernel_size=16, stride=1)  # Output: (B, 16, 1)

        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 16)
        # self.block6_dense = ai8x.Linear(in_features=16, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           EpiDeNet_B Variant 2.2
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=4, stride=1,
        #                                                       padding=2)  # Output: (B, 4, 513)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 64)
        #
        # # Block 2
        # self.block2_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=16, kernel_size=9, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 60)
        # self.block2_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=8, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 57)
        # self.block2_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 16, 14)
        #
        # # Block 5
        # self.block5_avgpool1d = ai8x.AvgPool1d(kernel_size=14, stride=1)  # Output: (B, 16, 1)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 16)
        # self.block6_dense = ai8x.Linear(in_features=16, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           EpiDeNet_B Variant 2.3
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=4, stride=1,
        #                                                       padding=2)  # Output: (B, 4, 513)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 64)
        #
        # # Block 2
        # self.block2_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=16, kernel_size=9, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 60)
        # self.block2_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 16, 15)
        #
        # # Block 5
        # self.block5_avgpool1d = ai8x.AvgPool1d(kernel_size=15, stride=1)  # Output: (B, 16, 1)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 16)
        # self.block6_dense = ai8x.Linear(in_features=16, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           EpiDeNet_B Variant 3.1
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=4, stride=1,
        #                                                       padding=2)  # Output: (B, 4, 769)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 96)
        #
        # # Block 2
        # self.block2_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=16, kernel_size=9, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 92)
        # self.block2_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=8, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 89)
        # self.block2_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 16, 22)
        #
        # # Block 3
        # self.block3_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=8, stride=1,
        #                                                       padding=2)  # Output: (B, 16, 19)
        # self.block3_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 16, 4)
        #
        # # Block 5
        # self.block5_avgpool1d = ai8x.AvgPool1d(kernel_size=4, stride=1)  # Output: (B, 16, 1)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 16)
        # self.block6_dense = ai8x.Linear(in_features=16, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           EpiDeNet_B Variant 3.2
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=4, stride=1,
                                                              padding=2)  # Output: (B, 4, 769)
        self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 96)

        # Block 2
        self.block2_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=16, kernel_size=5, stride=1,
                                                              padding=2)  # Output: (B, 16, 96)
        self.block2_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=5, stride=1,
                                                              padding=2)  # Output: (B, 16, 96)
        self.block2_conv1d_bn_relu_3 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=5, stride=1,
                                                              padding=2)  # Output: (B, 16, 96)
        self.block2_conv1d_bn_relu_4 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=4, stride=1,
                                                              padding=2)  # Output: (B, 16, 97)
        self.block2_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 16, 24)

        # Block 3
        self.block3_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=16, out_channels=16, kernel_size=8, stride=1,
                                                              padding=2)  # Output: (B, 16, 21)
        self.block3_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 16, 5)

        # Block 5
        self.block5_avgpool1d = ai8x.AvgPool1d(kernel_size=5, stride=1)  # Output: (B, 16, 1)

        # Block 6
        self.block6_flatten = nn.Flatten()  # Output: (B, 16)
        self.block6_dense = ai8x.Linear(in_features=16, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))
        x = self.block1_conv1d_bn_relu_1(x)
        x = self.block1_maxpool1d_1(x)
        x = self.block2_conv1d_bn_relu_1(x)
        x = self.block2_conv1d_bn_relu_2(x)
        x = self.block2_conv1d_bn_relu_3(x)
        x = self.block2_conv1d_bn_relu_4(x)
        x = self.block2_maxpool1d_1(x)
        x = self.block3_conv1d_bn_relu_1(x)
        x = self.block3_maxpool1d_1(x)
        x = self.block5_avgpool1d(x)
        x = self.block6_flatten(x)
        x = self.block6_dense(x)

        return x


def epidenet_b(pretrained=False, **kwargs):
    """
    Constructs a EpiDeNet-B model.
    """
    assert not pretrained
    return EpiDeNetB(**kwargs)


models = [
    {
        'name': 'epidenet_b',
        'min_input': 1,
        'dim': 2,
    },
]
