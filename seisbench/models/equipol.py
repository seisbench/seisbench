import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import seisbench.util as sbu

from .base import WaveformModel


class EquiPol(WaveformModel):
    """
    EquiPol polarity picking model by Kartseva et al. (in revision).
    The model guarantees symmetry with respect to sign flips by jointly processing the waveform and it's inverse.
    Predictions from both paths are combined.

    .. document_args:: seisbench.models EquiPol
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["picks"] = (
        "[Required] Picks as either PickList or list of tuples with (station, time)",
        None,
    )
    _annotate_args["threshold"] = (
        "Only export polarities with at least this confidence",
        0.5,
    )

    def __init__(self, in_channels=3, component_order="ZNE", linear_size=128, **kwargs):
        citation = (
            "Kartseva, T., Münchmeyer, J., Gardonio, B., Helmstetter, A., Marsan, D., Socquet, A. (in revision) "
            "Generalization Across Tectonic Regions for Deep-Learning P-Polarity Pickers."
        )

        assert len(component_order) == in_channels, (
            "Number of channels needs to match component order."
        )
        self.in_channels = in_channels
        self.linear_size = linear_size

        super().__init__(
            output_type="triggered_point",
            samples_before_after=(200, 200),
            component_order=component_order,
            citation=citation,
            labels="DU",  # Down, Up
            pick_filter="P",
            grouping="instrument",
            **kwargs,
        )
        self._build_model(in_channels=in_channels, linear_size=linear_size)

    def _build_model(self, in_channels, linear_size):
        self.activation = torch.relu

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=21)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=9)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(in_features=480, out_features=linear_size)
        self.fc2 = nn.Linear(in_features=linear_size, out_features=linear_size)
        self.fc3 = nn.Linear(in_features=linear_size, out_features=2)

    def forward(self, x, logits=False):
        # Feeds waveform through the NN twice but with flipped sign
        # Combines logits afterwards by swapping output classes
        # This ensures equivariance to sign changes
        y_plus = self._forward_branch(x)
        y_minus = self._forward_branch(-x)
        y = torch.stack(
            [y_plus[:, 0] + y_minus[:, 1], y_plus[:, 1] + y_minus[:, 0]], dim=-1
        )

        if logits:
            return y
        else:
            return F.softmax(y, dim=-1)

    def _forward_branch(self, x):
        # x shape: (batch, channels, samples=400)
        x = self.activation(self.pool(self.bn1(self.conv1(x))))
        x = self.activation(self.pool(self.bn2(self.conv2(x))))
        x = self.activation(self.pool(self.bn3(self.conv3(x))))
        x = self.activation(self.pool(self.bn4(self.conv4(x))))

        x = x.reshape(-1, 480)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation for output layer, returns logits

        return x

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(axis=-1, keepdims=True)
        batch = sbu.torch_detrend(batch)

        std = batch.std(axis=(-1, -2), keepdims=True)
        batch = batch / (std + 1e-10)

        return batch

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to a ``PickList``. If the input is a ``PickList``, all information is retained.
        If not, creates a new ``PickList`` with only the parameters ``trace_id``, ``peak_time``, ``polarity``,
        and ``polarity_confidence`` filled.

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        """
        threshold = argdict.get("threshold", self._annotate_args.get("threshold")[1])
        picks_in = argdict["picks"]

        picks_out = sbu.PickList()
        for pick, polarity in zip(picks_in, annotations):
            if polarity is None or np.max(polarity) < threshold:
                polarity_out = None
                confidence = None

            else:
                polarity_out = self.labels[np.argmax(polarity)]
                confidence = np.max(polarity)

            if isinstance(picks_in, sbu.PickList):
                pick = copy.deepcopy(pick)
                pick.polarity = polarity_out
                pick.polarity_value = confidence
                picks_out.append(pick)

            else:
                picks_out.append(
                    sbu.Pick(
                        trace_id=pick[0],
                        peak_time=pick[1],
                        polarity=polarity_out,
                        polarity_value=confidence,
                    )
                )

        return sbu.ClassifyOutput(self.name, picks=picks_out)

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "samples_before_after",
            "output_type",
            "labels",
            "pick_filter",
            "grouping",
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["linear_size"] = self.linear_size

        return model_args
