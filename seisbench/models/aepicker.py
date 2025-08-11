from typing import Any

import numpy as np
import torch
import torch.nn as nn

import seisbench.util as sbu

from .base import WaveformModel


class BasicPhaseAE(WaveformModel):
    """
    Simple AutoEncoder network architecture to pick P-/S-phases,
    from Woollam et al., (2019).

    .. document_args:: seisbench.models BasicPhaseAE

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

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 300)

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

    def forward(self, x, logits=False):
        x_l1 = self.activation(self.conv2(self.activation(self.conv1(x))))
        x_l2 = self.drop1(self.activation(self.conv3(x_l1)))
        x_l3 = self.upsample1(self.maxpool1(self.activation(self.conv4(x_l2))))

        x_r3 = self.upsample2(self.activation(self.conv5(x_l3)))
        x_r2 = self.upsample3(self.activation(self.conv6(x_r3)))
        x_r1 = self.upsample4(self.activation(self.conv7(x_r2)))

        x_out = self.out(x_r1)

        if logits:
            return x_out
        else:
            return self.softmax(x_out)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(axis=-1, keepdims=True)

        std = batch.std(axis=-1, keepdims=True)
        batch = batch / std.clip(1e-10)

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        # Transpose predictions to correct shape
        batch[..., 130] = np.nan
        batch[..., -130:] = np.nan
        return torch.transpose(batch, -1, -2)

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete thresholds using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = sbu.PickList()
        for phase in self.labels:
            if phase == "N":
                # Don't pick noise
                continue

            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )
        picks = sbu.PickList(sorted(picks))

        return sbu.ClassifyOutput(self.name, picks=picks)

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
