# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# import your own module
import ai8x


class EpiDeNetA(nn.Module):
    """
    NOTE: This model is the variant A on the EpiDeNet architecture. This
    variant satisfies the MAX78000 constraints.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Input dimensions (B, C, H, W) = (B, 1, 4, 1024)

        # Block 1
        self.block1_conv2d_bn_relu = ai8x.FusedConv2dBNReLU(in_channels=1,
                                                          out_channels=4,
                                                          kernel_size=3,
                                                          stride=1,
                                                          padding=1)
        self.block1_conv2d_mask = torch.tensor([[0, 0, 0],
                                                [1, 1, 1],
                                                [0, 0, 0]])
        self.block1_maxpool1d = ai8x.MaxPool1d(kernel_size=8, stride=8)

        # Block 2
        self.block2_conv2d_bn_relu = ai8x.FusedConv2dBNReLU(in_channels=4,
                                                          out_channels=16,
                                                          kernel_size=3,
                                                          stride=1,
                                                          padding=1)
        self.block2_conv2d_mask = torch.tensor([[0, 0, 0],
                                                [1, 1, 1],
                                                [0, 0, 0]])
        self.block2_maxpool1d = ai8x.MaxPool1d(kernel_size=4, stride=4)

        # Block 3
        self.block3_conv2d_bn_relu = ai8x.FusedConv2dBNReLU(in_channels=16,
                                                            out_channels=16,
                                                            kernel_size=3,
                                                            stride=1,
                                                            padding=1)
        self.block3_conv2d_mask = torch.tensor([[0, 0, 0],
                                                [1, 1, 1],
                                                [0, 0, 0]])
        self.block3_maxpool1d = ai8x.MaxPool1d(kernel_size=4, stride=4)

        # Block 4
        self.block4_conv2d_bn_relu = ai8x.FusedConv2dBNReLU(in_channels=16,
                                                            out_channels=16,
                                                            kernel_size=3,
                                                            stride=1,
                                                            padding=1)
        self.block4_conv2d_mask = torch.tensor([[0, 1, 0],
                                                [0, 1, 0],
                                                [0, 1, 0]])
        self.block4_maxpool2d = ai8x.MaxPool2d(kernel_size=(4, 1),
                                               stride=(1, 1))

        # Block 5
        self.block5_conv1d_bn_relu = ai8x.FusedConv1dBNReLU(in_channels=16,
                                                            out_channels=16,
                                                            kernel_size=1,
                                                            stride=1,
                                                            padding=0)
        self.block5_avgpool1d = ai8x.AvgPool1d(kernel_size=8, stride=1)

        # Block 6
        self.block6_dense = ai8x.Linear(in_features=16, out_features=2)

    def forward(self, x):
        # (B, C, H, W)
        # x: (B, 1, 4, 1024)
        batch = x.shape[0]
        #print("Input")
        #print(x.shape)

        # Block 1
        #print("Block 1")
        self._mask_weights(self.block1_conv2d_bn_relu, self.block1_conv2d_mask)
        x = self.block1_conv2d_bn_relu(x)   # (B, 4, 4, 1024)
        #print(x.shape)

        # Apply MaxPool1d to each EEG channel (H dimension) individually and
        # concatenate all four outputs
        x1 = self.block1_maxpool1d(x[:, 0, :, :])  # (B, 4, 128)
        x2 = self.block1_maxpool1d(x[:, 1, :, :])  # (B, 4, 128)
        x3 = self.block1_maxpool1d(x[:, 2, :, :])  # (B, 4, 128)
        x4 = self.block1_maxpool1d(x[:, 3, :, :])  # (B, 4, 128)
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        #print(x4.shape)
        x = torch.cat((x1, x2, x3, x4), 2)  # (B, 4, 512)
        #print(x.shape)
        x = x.view(batch, 4, 4, 128)                    # (B, 4, 4, 128)
        #print(x.shape)

        # Block 2
        #print("Block 2")
        self._mask_weights(self.block2_conv2d_bn_relu, self.block2_conv2d_mask)
        x = self.block2_conv2d_bn_relu(x)   # (B, 16, 4, 128)
        #print(x.shape)
        x = x.view(batch, 16, 512)          # (B, 16, 512)
        #print(x.shape)
        x = self.block2_maxpool1d(x)        # (B, 16, 128)
        #print(x.shape)
        x = x.view(batch, 16, 4, 32)        # (B, 16, 4, 32)
        #print(x.shape)

        # Block 3
        #print("Block 3")
        self._mask_weights(self.block3_conv2d_bn_relu, self.block3_conv2d_mask)
        x = self.block3_conv2d_bn_relu(x)   # (B, 16, 4, 32)
        #print(x.shape)
        x = x.view(batch, 16, 128)          # (B, 16, 128)
        #print(x.shape)
        x = self.block3_maxpool1d(x)        # (B, 16, 32)
        #print(x.shape)
        x = x.view(batch, 16, 4, 8)         # (B, 16, 4, 8)
        #print(x.shape)

        # Block 4
        #print("Block 4")
        self._mask_weights(self.block4_conv2d_bn_relu, self.block4_conv2d_mask)
        x = self.block4_conv2d_bn_relu(x)   # (B, 16, 4, 8)
        #print(x.shape)
        x = self.block4_maxpool2d(x)        # (B, 16, 1, 8)
        #print(x.shape)

        # Block 5
        #print("Block 5")
        x = x.view(batch, 16, 8)            # (B, 16, 8)
        #print(x.shape)
        x = self.block5_conv1d_bn_relu(x)   # (B, 16, 8)
        #print(x.shape)
        x = self.block5_avgpool1d(x)        # (B, 16, 1)
        #print(x.shape)

        # Block 6
        #print("Block 6")
        x = x.view(batch, 16)                # (B, 16)
        #print(x.shape)
        x = self.block6_dense(x)            # (B, 2)
        #print(x.shape)
        return x

    @staticmethod
    def _mask_weights(layer: ai8x.QuantizationAwareModule,
                      mask: torch.Tensor):
        weights = layer.op.weight
        layer.op.weight = nn.Parameter(weights * mask)


def epidenet_a(pretrained=False, **kwargs):
    """
    Constructs a EpiDeNetA model.
    """
    assert not pretrained
    return EpiDeNetA(**kwargs)


models = [
    {
        'name': 'epidenet_a',
        'min_input': 1,
        'dim': 2,
    },
]