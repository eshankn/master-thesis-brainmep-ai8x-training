import torch
import torch.nn as nn

import ai8x


class EnergyProfiling(nn.Module):
    """
    NOTE: This document features various model architectures to profile the MAX7800. The model variants satisfy
    the MAX78000 constraints.

    It uses inputs from the preprocessed CHB-MIT dataset for the EpiDeNet architecture
    """

    def __init__(self, **kwargs):
        super().__init__()

        # MAX78000 constraint: The maximum dimension (number of rows or columns) for input or output data to be 1023.
        # MAX78000 constraint for synthesis: Padding size should be smaller than kernel size.
        # Input: (B, 1, 768)

        # ----------------------------------------------------------------------------------------------------
        #                                           Variant 1
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=4, stride=1,
                                                              padding=2)  # Output: (B, 4, 1017)
        self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 127)

        # Block 2
        self.block2_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=4, kernel_size=4, stride=1,
                                                              padding=2)  # Output: (B, 4, 128)
        self.block2_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=8, stride=8)  # Output: (B, 4, 16)

        # Block 6
        self.block6_flatten = nn.Flatten()  # Output: (B, 64)
        self.block6_dense = ai8x.Linear(in_features=64, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           Variant 2.1
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 767)
        # self.block1_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 766)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=3, stride=3)  # Output: (B, 1, 255)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 255)
        # self.block6_dense = ai8x.Linear(in_features=255, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           Variant 3.1
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 767)
        # self.block1_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 766)
        # self.block1_conv1d_bn_relu_3 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 765)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=3, stride=3)  # Output: (B, 1, 255)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 4)
        # self.block6_dense = ai8x.Linear(in_features=255, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           Variant 4.1
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 767)
        # self.block1_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 766)
        # self.block1_conv1d_bn_relu_3 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 765)
        # self.block1_conv1d_bn_relu_4 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=1, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 1, 764)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=3, stride=3)  # Output: (B, 1, 255)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 4)
        # self.block6_dense = ai8x.Linear(in_features=254, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           Variant 4.2
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 767)
        # self.block1_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=2, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 766)
        # self.block1_conv1d_bn_relu_3 = ai8x.FusedConv1dBNReLU(in_channels=2, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 765)
        # self.block1_conv1d_bn_relu_4 = ai8x.FusedConv1dBNReLU(in_channels=2, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 764)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=12, stride=12)  # Output: (B, 4, 63)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 4)
        # self.block6_dense = ai8x.Linear(in_features=126, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           Variant 6.1
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 767)
        # self.block1_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=2, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 766)
        # self.block1_conv1d_bn_relu_3 = ai8x.FusedConv1dBNReLU(in_channels=2, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 765)
        # self.block1_conv1d_bn_relu_4 = ai8x.FusedConv1dBNReLU(in_channels=2, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 764)
        # self.block1_conv1d_bn_relu_5 = ai8x.FusedConv1dBNReLU(in_channels=2, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 763)
        # self.block1_conv1d_bn_relu_6 = ai8x.FusedConv1dBNReLU(in_channels=2, out_channels=2, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 762)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=12, stride=12)  # Output: (B, 2, 63)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 4)
        # self.block6_dense = ai8x.Linear(in_features=126, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------
        #                                           Variant 6.2
        # ----------------------------------------------------------------------------------------------------

        # Block 1
        # self.block1_conv1d_bn_relu_1 = ai8x.FusedConv1dBNReLU(in_channels=1, out_channels=4, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 767)
        # self.block1_maxpool1d_1 = ai8x.MaxPool1d(kernel_size=4, stride=4)  # Output: (B, 4, 191)
        #
        # self.block1_conv1d_bn_relu_2 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=4, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 190)
        # self.block1_conv1d_bn_relu_3 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=4, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 189)
        # self.block1_conv1d_bn_relu_4 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=4, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 188)
        # self.block1_conv1d_bn_relu_5 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=4, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 186)
        # self.block1_conv1d_bn_relu_6 = ai8x.FusedConv1dBNReLU(in_channels=4, out_channels=4, kernel_size=2, stride=1,
        #                                                       padding=0)  # Output: (B, 4, 185)
        # self.block1_maxpool1d_2 = ai8x.MaxPool1d(kernel_size=12, stride=12)  # Output: (B, 4, 15)
        #
        # # Block 6
        # self.block6_flatten = nn.Flatten()  # Output: (B, 4)
        # self.block6_dense = ai8x.Linear(in_features=60, out_features=2)  # Output: (B, 2)

        # ----------------------------------------------------------------------------------------------------

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))

        x = self.block1_conv1d_bn_relu_1(x)
        x = self.block1_maxpool1d_1(x)

        x = self.block2_conv1d_bn_relu_1(x)
        x = self.block2_maxpool1d_1(x)

        x = self.block6_flatten(x)
        x = self.block6_dense(x)

        return x


def energy_profiling(pretrained=False, **kwargs):
    """
    Constructs a EpiDeNet-B model.
    """
    assert not pretrained
    return EnergyProfiling(**kwargs)


models = [
    {
        'name': 'energy_profiling',
        'min_input': 1,
        'dim': 2,
    },
]
