from .base import WaveformModel
import seisbench

import torch
import torch.nn as nn
import numpy as np
from obspy.signal.trigger import trigger_onset


class GPD(WaveformModel):
    def __init__(self, in_channels=3, classes=3, phases=None, eps=1e-10, **kwargs):
        citation = (
            "Ross, Z. E., Meier, M.-A., Hauksson, E., & Heaton, T. H. (2018). "
            "Generalized Seismic Phase Detection with Deep Learning. "
            "ArXiv:1805.01075 [Physics]. http://arxiv.org/abs/1805.01075"
        )
        # Define default value for sampling rate
        kwargs["sampling_rate"] = kwargs.get("sampling_rate", 100)

        super().__init__(
            citation=citation,
            output_type="point",
            in_samples=400,
            pred_sample=200,
            labels=phases,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.eps = eps
        self._phases = phases
        if phases is not None and len(phases) != classes:
            raise ValueError(
                f"Number of classes ({classes}) does not match number of phases ({len(phases)})."
            )

        # TODO: Verify Pooling
        # TODO: Verify order of activation, pooling and batch norm
        self.conv1 = nn.Conv1d(in_channels, 32, 21)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 21)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 21)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 21)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(1536, 200)
        self.bn5 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn6 = nn.BatchNorm1d(200)
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
        x = self.activation(self.pool(self.bn1(self.conv1(x))))
        x = self.activation(self.pool(self.bn2(self.conv2(x))))
        x = self.activation(self.pool(self.bn3(self.conv3(x))))
        x = self.activation(self.pool(self.bn4(self.conv4(x))))

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

    def _parse_metadata(self):
        super()._parse_metadata()
        self._phases = self._weights_metadata.get("phases", None)

    def annotate_window_pre(self, window, argdict):
        # Add a demean step to the preprocessing
        return window - np.mean(window, axis=-1, keepdims=True)

    def classify_aggregate(self, annotations, argdict):
        """
        Converts the annotations to discrete thresholds using a classical trigger on/off.
        Trigger onset thresholds are derived from the argdict at keys "[phase]_threshold".
        For all triggers the lower threshold is set to half the higher threshold.
        For each pick a triple is returned, consisting of the trace_id ("net.sta.loc"), the pick time and the phase.

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = []
        for phase in self.phases:
            if phase == "N":
                # Don't pick noise
                continue

            pick_threshold = argdict.get(f"{phase}_threshold", 0.3)
            for trace in annotations.select(channel=f"GPD_{phase}"):
                trace_id = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"
                triggers = trigger_onset(trace.data, pick_threshold, pick_threshold / 2)
                times = trace.times()
                for s0, _ in triggers:
                    t0 = trace.stats.starttime + times[s0]
                    picks.append((trace_id, t0, phase))

        return sorted(picks)
