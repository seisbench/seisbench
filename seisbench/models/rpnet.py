import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import seisbench.util as sbu

from .base import WaveformModel

# ===========================================================================
# RPNet — SeisBench WaveformModel
# ===========================================================================


class RPNet(WaveformModel):
    """
    RPNet polarity picking model by Han et al. (2025) wrapped as SeisBench WaveformModel.
    The model has been made equivariant using the approach from EquiPol (Kartseva et al., 2026).

    .. document_args:: seisbench.models RPNet
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

    def __init__(
        self, dropout_rate=0.2, lstm_hidden=480, attn_dim=32, attn_width=20, **kwargs
    ):
        citation = (
            "Han, J., Kim, S., & Sheen, D. H. (2025). "
            "RPNet: Robust p‐wave first‐motion polarity "
            "determination using deep learning. "
            "Seismological Research Letters, 96(4), 2405-2417."
        )
        super().__init__(
            output_type="triggered_point",
            samples_before_after=(200, 200),
            component_order="Z",
            citation=citation,
            labels="DU",  # Down, Up
            pick_filter="P",
            grouping="instrument",
            **kwargs,
        )
        self.dropout_rate = dropout_rate
        self.lstm_hidden = lstm_hidden
        self.attn_dim = attn_dim
        self.attn_width = attn_width
        self._build_model()

    def _build_model(self):
        # (B, 1, 400) → (B, 64, 200)
        self.conv1 = nn.Conv1d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=_same_padding(7, stride=2),
        )
        # → (B, 64, 100)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.incept1 = InceptionBlock1(64)  # → (B, 224, 100)
        self.incept2 = InceptionBlock2(224)  # → (B, 480, 100)

        # → (B, 480, 50)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # (B, 50, 480) → (B, 50, lstm_hidden)
        self.lstm = nn.LSTM(
            input_size=480,
            hidden_size=self.lstm_hidden,
            batch_first=True,
        )

        self.attn = AdditiveAttention(
            dim=self.lstm_hidden, attn_dim=self.attn_dim, width=self.attn_width
        )

        self.fc1 = nn.Linear(50 * self.lstm_hidden, 128)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(128, 2)

    def _forward_branch(self, x):
        x = F.relu(self.conv1(x))  # (B, 64, 200)
        x = self.pool1(x)  # (B, 64, 100)

        x = self.incept1(x)  # (B, 224, 100)
        x = self.incept2(x)  # (B, 480, 100)

        x = self.pool2(x)  # (B, 480, 50)

        x = x.permute(0, 2, 1)  # (B, 50, 480)
        x, _ = self.lstm(x)  # (B, 50, lstm_hidden)
        x, _ = self.attn(x)  # (B, 50, lstm_hidden)

        x = x.reshape(x.size(0), -1)  # (B, 50*lstm_hidden)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 2) — logits
        return x

    def forward(self, x, logits=False):
        y_plus = self._forward_branch(x)
        y_minus = self._forward_branch(-x)

        y = torch.stack(
            [y_plus[:, 0] + y_minus[:, 1], y_plus[:, 1] + y_minus[:, 0]], dim=-1
        )

        if logits:
            return y
        else:
            return F.softmax(y, dim=-1)

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
            "component_order",
            "citation",
            "samples_before_after",
            "output_type",
            "labels",
            "pick_filter",
            "grouping",
        ]:
            del model_args[key]

        model_args["dropout_rate"] = self.dropout_rate
        model_args["lstm_hidden"] = self.lstm_hidden
        model_args["attn_dim"] = self.attn_dim
        model_args["attn_width"] = self.attn_width

        return model_args


# ===========================================================================
# RPNet helpers
# ===========================================================================


def _same_padding(kernel_size, stride=1, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


class InceptionBlock1(nn.Module):
    """First inception block: 64 → 224 channels"""

    def __init__(self, in_ch):
        super().__init__()
        self.b1 = nn.Conv1d(in_ch, 64, kernel_size=1)

        self.b2_1 = nn.Conv1d(in_ch, 64, kernel_size=1)
        self.b2_2 = nn.Conv1d(64, 96, kernel_size=3, padding=_same_padding(3))

        self.b3_1 = nn.Conv1d(in_ch, 16, kernel_size=1)
        self.b3_2 = nn.Conv1d(16, 32, kernel_size=5, padding=_same_padding(5))

        self.b4_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.b4_1 = nn.Conv1d(in_ch, 32, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_1(self.b4_pool(x)))
        return torch.cat([b1, b2, b3, b4], dim=1)  # → 64+96+32+32 = 224


class InceptionBlock2(nn.Module):
    """Second inception block: 224 → 480 channels"""

    def __init__(self, in_ch):
        super().__init__()
        self.b1 = nn.Conv1d(in_ch, 128, kernel_size=1)

        self.b2_1 = nn.Conv1d(in_ch, 128, kernel_size=1)
        self.b2_2 = nn.Conv1d(128, 192, kernel_size=3, padding=_same_padding(3))

        self.b3_1 = nn.Conv1d(in_ch, 32, kernel_size=1)
        self.b3_2 = nn.Conv1d(32, 96, kernel_size=5, padding=_same_padding(5))

        self.b4_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.b4_1 = nn.Conv1d(in_ch, 64, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_1(self.b4_pool(x)))
        return torch.cat([b1, b2, b3, b4], dim=1)  # → 128+192+96+64 = 480


class AdditiveAttention(nn.Module):
    """
    Local additive (Bahdanau-style) attention.
    Matches SeqSelfAttention from the original Keras RPNet.
    """

    def __init__(self, dim, attn_dim=32, width=20):
        super().__init__()
        self.width = width
        self.W = nn.Linear(dim, attn_dim, bias=True)
        self.U = nn.Linear(dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=True)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        Q = self.W(x)  # (B, T, A)
        K = self.U(x)  # (B, T, A)
        scores = self.v(torch.relu(Q.unsqueeze(2) + K.unsqueeze(1))).squeeze(
            -1
        )  # (B, T, T)

        # local window mask
        idx = torch.arange(T, device=x.device)
        mask = ((idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.width).unsqueeze(0)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # (B, T, T)
        out = torch.matmul(attn, x)  # (B, T, C)
        return out, attn
