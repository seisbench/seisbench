from .base import WaveformModel

import torch
import torch.nn as nn
import numpy as np


class BasicPhaseAE(WaveformModel):
    """
    Simple AutoEncoder network architecture to pick P-/S-phases,
    from Woollam et al., (2019).

    :param in_channels: Number of input channels, by default 3.
    :type in_channels: int
    :param in_samples: Number of input samples per channel, by default 600.
                       The model expects input shape (in_channels, in_samples)
    :type in_samples: int
    :param classes: Number of output classes, by default 3.
    :type classes: int
    :param phases: Phase hints for the classes, by default "NPS". Can be None.
    :type phases: list, str
    :param sampling_rate: Sampling rate of traces, by default 100.
    :type sampling_rate: float
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
            in_samples=600,
            output_type="array",
            default_args={"overlap": 300},
            pred_sample=(0, 600),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.kernel_size = 5
        self.stride = 4
        self.activation = torch.relu

        # l1
        self.conv1 = nn.Conv1d(3, 12, self.kernel_size)
        self.conv2 = nn.Conv1d(12, 24, self.kernel_size, self.stride)
        # l2
        self.conv3 = nn.Conv1d(24, 36, self.kernel_size, self.stride)
        self.drop1 = nn.Dropout(0.3)
        # l3
        self.conv4 = nn.Conv1d(36, 48, self.kernel_size, self.stride)
        self.maxpool1 = nn.MaxPool1d(2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2)

        # r3
        self.conv5 = nn.ConvTranspose1d(48, 48, self.kernel_size, padding=2)
        self.upsample2 = nn.Upsample(scale_factor=5)
        # r2
        self.conv6 = nn.ConvTranspose1d(48, 24, self.kernel_size, padding=2)
        self.upsample3 = nn.Upsample(scale_factor=5)
        # r1
        self.conv7 = nn.ConvTranspose1d(24, 12, self.kernel_size, padding=2)
        self.upsample4 = nn.Upsample(scale_factor=3)

        self.out = nn.ConvTranspose1d(12, self.classes, self.kernel_size, padding=2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        x_l1 = self.activation(self.conv2(self.activation(self.conv1(x))))
        x_l2 = self.drop1(self.activation(self.conv3(x_l1)))
        x_l3 = self.upsample1(self.maxpool1(self.activation(self.conv4(x_l2))))

        x_r3 = self.upsample2(self.activation(self.conv5(x_l3)))
        x_r2 = self.upsample3(self.activation(self.conv6(x_r3)))
        x_r1 = self.upsample4(self.activation(self.conv7(x_r2)))

        x_out = self.softmax(self.out(x_r1))

        return x_out

    def annotate_window_pre(self, window, argdict):
        # Add a demean and normalize step to the preprocessing
        window = window - np.mean(window, axis=-1, keepdims=True)
        std = np.std(window, axis=-1, keepdims=True)
        std[std == 0] = 1  # Avoid NaN errors
        window = window / std
        return window

    def annotate_window_post(self, pred, piggyback=None, argdict=None):
        # Transpose predictions to correct shape
        pred[:, 130] = np.nan
        pred[:, -130:] = np.nan
        return pred.T

    def classify_aggregate(self, annotations, argdict):
        """
        Converts the annotations to discrete thresholds using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = []
        for phase in self.labels:
            if phase == "N":
                # Don't pick noise
                continue

            picks += self.picks_from_annotations(
                annotations.select(channel=f"BasicPhaseAE_{phase}"),
                argdict.get(f"{phase}_threshold", 0.3),
                phase,
            )

        return sorted(picks)

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
            "sampling_rate",
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate

        return model_args
