from .base import WaveformModel

import torch
import torch.nn as nn
import numpy as np


class GPD(WaveformModel):
    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases=None,
        eps=1e-10,
        sampling_rate=100,
        pred_sample=200,
        original_compatible=False,
        **kwargs,
    ):
        citation = (
            "Ross, Z. E., Meier, M.-A., Hauksson, E., & Heaton, T. H. (2018). "
            "Generalized Seismic Phase Detection with Deep Learning. "
            "ArXiv:1805.01075 [Physics]. https://arxiv.org/abs/1805.01075"
        )
        super().__init__(
            citation=citation,
            output_type="point",
            in_samples=400,
            pred_sample=pred_sample,
            labels=phases,
            sampling_rate=sampling_rate,
            default_args={"stride": 10},
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.eps = eps
        self._phases = phases
        self.original_compatible = original_compatible
        if phases is not None and len(phases) != classes:
            raise ValueError(
                f"Number of classes ({classes}) does not match number of phases ({len(phases)})."
            )

        self.conv1 = nn.Conv1d(in_channels, 32, 21, padding=10)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-3)
        self.conv2 = nn.Conv1d(32, 64, 15, padding=7)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-3)
        self.conv3 = nn.Conv1d(64, 128, 11, padding=5)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-3)
        self.conv4 = nn.Conv1d(128, 256, 9, padding=4)
        self.bn4 = nn.BatchNorm1d(256, eps=1e-3)

        self.fc1 = nn.Linear(6400, 200)
        self.bn5 = nn.BatchNorm1d(200, eps=1e-3)
        self.fc2 = nn.Linear(200, 200)
        self.bn6 = nn.BatchNorm1d(200, eps=1e-3)
        self.fc3 = nn.Linear(200, classes)

        self.activation = torch.relu

        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        # Max normalization
        x = x / (
            torch.max(
                torch.max(torch.abs(x), dim=-1, keepdims=True)[0], dim=-2, keepdims=True
            )[0]
            + self.eps
        )
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = self.pool(self.activation(self.bn4(self.conv4(x))))

        if self.original_compatible:
            # Permutation is required to be consistent with the following fully connected layer
            x = x.permute(0, 2, 1)
        x = torch.flatten(x, 1)

        x = self.activation(self.bn5(self.fc1(x)))
        x = self.activation(self.bn6(self.fc2(x)))
        x = self.fc3(x)

        if self.classes == 1:
            return torch.sigmoid(x)
        else:
            return torch.softmax(x, -1)

    @property
    def phases(self):
        if self._phases is not None:
            return self._phases
        else:
            return list(range(self.classes))

    def annotate_window_pre(self, window, argdict):
        # Add a demean step to the preprocessing
        return window - np.mean(window, axis=-1, keepdims=True)

    def classify_aggregate(self, annotations, argdict):
        """
        Converts the annotations to discrete picks using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = []
        for phase in self.phases:
            if phase == "N":
                # Don't pick noise
                continue

            picks += self.picks_from_annotations(
                annotations.select(channel=f"GPD_{phase}"),
                argdict.get(f"{phase}_threshold", 0.7),
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

        model_args["sampling_rate"] = self.sampling_rate
        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self._phases
        model_args["eps"] = self.eps
        model_args["sampling_rate"] = self.sampling_rate
        model_args["pred_sample"] = self.pred_sample
        model_args["original_compatible"] = self.original_compatible

        return model_args
