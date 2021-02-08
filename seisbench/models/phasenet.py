from .base import WaveformModel

import torch
import torch.nn as nn
import math


class Conv1dSame(nn.Module):
    """
    Add PyTorch compatible support for Tensorflow/Keras padding option: padding='same'.

    Discussions regarding feature implementation:
    https://discuss.pytorch.org/t/converting-tensorflow-model-to-pytorch-issue-with-padding/84224
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-598264120
    
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (
            kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1
        )
        self.padding = math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding + 1,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class PhaseNet(WaveformModel):
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.kernel_size = 7
        self.stride = 4
        self.activation = torch.relu

        self.inc = nn.Conv1d(self.in_channels, 8, 1)
        self.in_bn = nn.BatchNorm1d(8)

        self.conv1 = Conv1dSame(8, 11, self.kernel_size, self.stride)
        self.bnd1 = nn.BatchNorm1d(11)

        self.conv2 = Conv1dSame(11, 16, self.kernel_size, self.stride)
        self.bnd2 = nn.BatchNorm1d(16)

        self.conv3 = Conv1dSame(16, 22, self.kernel_size, self.stride)
        self.bnd3 = nn.BatchNorm1d(22)

        self.conv4 = Conv1dSame(22, 32, self.kernel_size, self.stride)
        self.bnd4 = nn.BatchNorm1d(32)

        self.up1 = nn.ConvTranspose1d(
            32, 22, self.kernel_size, self.stride, padding=self.conv4.padding
        )
        self.bnu1 = nn.BatchNorm1d(22)

        self.up2 = nn.ConvTranspose1d(
            44,
            16,
            self.kernel_size,
            self.stride,
            padding=self.conv3.padding,
            output_padding=1,
        )
        self.bnu2 = nn.BatchNorm1d(16)

        self.up3 = nn.ConvTranspose1d(
            32, 11, self.kernel_size, self.stride, padding=self.conv2.padding
        )
        self.bnu3 = nn.BatchNorm1d(11)

        self.up4 = nn.ConvTranspose1d(22, 8, self.kernel_size, self.stride, padding=3)
        self.bnu4 = nn.BatchNorm1d(8)

        self.out = nn.ConvTranspose1d(16, self.n_classes, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, summary=False):

        # Print architecture summary on forward-pass (to remove)
        if summary:

            print("PhaseNet\n\n[C, W]")
            print((x.shape[1], x.shape[2]))

            x_in = self.activation(self.in_bn(self.inc(x)))
            print((x_in.shape[1], x_in.shape[2]))
            x1 = self.activation(self.bnd1(self.conv1(x_in)))
            print("|\t", (x1.shape[1], x1.shape[2]))
            x2 = self.activation(self.bnd2(self.conv2(x1)))
            print("|\t|\t", (x2.shape[1], x2.shape[2]))
            x3 = self.activation(self.bnd3(self.conv3(x2)))
            print("|\t|\t|\t", (x3.shape[1], x3.shape[2]))
            x4 = self.activation(self.bnd4(self.conv4(x3)))

            print("|\t|\t|\t|\t", (x4.shape[1], x4.shape[2]))

            x = torch.cat([self.activation(self.bnu1(self.up1(x4))), x3], dim=1)
            print("|\t|\t|\t", (x.shape[1], x.shape[2]))
            x = torch.cat([self.activation(self.bnu2(self.up2(x))), x2], dim=1)
            print("|\t|\t", (x.shape[1], x.shape[2]))
            x = torch.cat([self.activation(self.bnu3(self.up3(x))), x1], dim=1)
            print("|\t", (x.shape[1], x.shape[2]))
            x = torch.cat([self.activation(self.bnu4(self.up4(x))), x_in], dim=1)
            print((x.shape[1], x.shape[2]))
            x = self.out(x)
            print((x.shape[1], x.shape[2]))

            x = self.softmax(x)

        else:

            x_in = self.activation(self.in_bn(self.inc(x)))

            x1 = self.activation(self.bnd1(self.conv1(x_in)))
            x2 = self.activation(self.bnd2(self.conv2(x1)))
            x3 = self.activation(self.bnd3(self.conv3(x2)))
            x4 = self.activation(self.bnd4(self.conv4(x3)))

            x = torch.cat([self.activation(self.bnu1(self.up1(x4))), x3], dim=1)
            x = torch.cat([self.activation(self.bnu2(self.up2(x))), x2], dim=1)
            x = torch.cat([self.activation(self.bnu3(self.up3(x))), x1], dim=1)
            x = torch.cat([self.activation(self.bnu4(self.up4(x))), x_in], dim=1)

            x = self.out(x)
            x = self.softmax(x)

        return x

    def annotate(self, stream, *args, **kwargs):
        raise NotImplementedError("Annotate is not yet implemented")

    def classify(self, stream, *args, **kwargs):
        raise NotImplementedError("Classify is not yet implemented")
