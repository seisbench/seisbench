from .base import WaveformModel, Conv1dSame

import torch
import torch.nn as nn


class BasicPhaseAE(WaveformModel):
    """
    Simple AutoEncoder network architecture to pick P-/S-phases,
    from Woollam et al., (2019).

    :param in_channels: Number of input channels, by default 3.
    :param in_samples: Number of input samples per channel, by default 600.
                       The model expects input shape (in_channels, in_samples)
    :param classes: Number of output classes, by default 3.
    :param phases: Phase hints for the classes, by default "NPS". Can be None.
    :param sampling_rate: Sampling rate of taces, by detault 100.
    :param kwargs: Keyword arguments passed to the constructor of :py:class:`WaveformModel`.

    """

    def __init__(
        self, in_channels=3, classes=3, phases="NPS", sampling_rate=100, **kwargs
    ):

        citation = (
            "Woollam, J., Rietbrock, A., Bueno, A. and De Angelis, S., 2019. "
            "Convolutional neural network for seismic phase classification, "
            "performance demonstration over a local seismic network. "
            "Seismological Research Letters, 90(2A), pp.491-502."
        )

        super().__init__(
            citation=citation,
            in_samples=601,
            output_type="array",
            default_args={"overlap": 100},
            pred_sample=(0, 601),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.kernel_size = 5
        self.stride = 4
        self.activation = torch.relu

        self.inc = Conv1dSame(3, 12, self.kernel_size, self.stride)

        self.conv1 = Conv1dSame(12, 24, self.kernel_size, self.stride)
        self.conv2 = Conv1dSame(24, 36, self.kernel_size, self.stride)
        self.drop1 = nn.Dropout(0.3)

        self.conv3 = Conv1dSame(36, 48, self.kernel_size, self.stride)
        self.maxpool1 = nn.MaxPool1d(2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv4 = Conv1dSame(48, 48, self.kernel_size, self.stride)
        self.upsample2 = nn.Upsample(scale_factor=5)

        self.conv5 = nn.ConvTranspose1d(48, 24, self.kernel_size)
        self.upsample3 = nn.Upsample(scale_factor=5)

        self.conv6 = nn.ConvTranspose1d(24, 12, self.kernel_size)
        self.upsample4 = nn.Upsample(scale_factor=3)

        self.out = nn.ConvTranspose1d(12, self.classes, self.kernel_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        x_in = self.activation(self.inc(x))

        x_l1 = self.activation(self.conv1(x_in))
        x_l2 = self.drop1(self.activation(self.conv2(x_l1)))
        x_l3 = self.upsample1(self.maxpool1(self.activation(self.conv3(x_l2))))

        x_r3 = self.upsample2((self.activation(self.conv4(x_l3))))
        x_r2 = self.upsample3((self.activation(self.conv5(x_r3))))
        x_r1 = self.upsample4((self.activation(self.conv6(x_r2))))

        x_out = self.softmax(self.out(x_r1))

        return x_out
