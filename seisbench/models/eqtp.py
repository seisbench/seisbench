from typing import Any

import numpy as np
import obspy
import torch
import torch.nn as nn

import seisbench.util as sbu
from .eqtransformer import (
    EQTransformer,
    SeqSelfAttention,
    Decoder,
    Encoder,
    ResCNNStack,
    Transformer,
    ActivationLSTMCell,
    CustomLSTM,
)


class EQTP(EQTransformer):
    """
    The EQTP from Peng et al. (2025)

    It is an extended version of the EQTransformer model, which builds upon its
    phase picking capabilities by adding P-wave polarity determination functionality.

    This model is designed for processing three-component seismic waveform data,
    and can simultaneously output picking results for phases such as P-waves/S-waves
    and polarity determination results for P-waves (Up U/Down D/Unknown N).

    Implementation is adapted from the EQTransformer with SeisBench GitHub repository (https://github.com/seisbench/seisbench).

    The EQTP model can be instantiated via the `from_pretrained("ncedc")` method.

    .. document_args:: seisbench.models EQTP

    """

    _annotate_args = EQTransformer._annotate_args.copy()
    _annotate_args["polarity_threshold"] = ("Polarity threshold", 0.3)

    def __init__(
        self,
        in_samples=12000,
        **kwargs,
    ):
        # Update citation for EQTP
        citation = (
            "Peng L, Li L, Zeng X. "
            "A Microseismic Phase Picking and Polarity Determination Model Based on the Earthquake Transformer[J ]. Applied Sciences, 2025, 15(7): 3424."
            "https://doi.org/10.3390/app15073424"
        )

        # Initialize parent class
        super().__init__(in_samples=in_samples, **kwargs)

        # Override citation and labels for EQTP
        self._citation = citation
        self.labels = ["Polarity_U", "Polarity_D"] + list(self.phases)

        # Override EQTP specific filter configurations
        self.filters = [8, 16, 16, 32, 64]
        self.kernel_sizes = [11, 9, 7, 7, 3]
        self.res_cnn_kernels = [3, 3, 3, 3, 2]

        # Rebuild encoder and res_cnn_stack with EQTP parameters
        self._rebuild_eqtp_components()

        # Remove detection branch (EQTP doesn't have detection)
        del self.decoder_d
        del self.conv_d

        # Override transformer input size to 64 for EQTP
        eps = 1e-7 if self.original_compatible else 1e-5
        self.transformer_d0 = self._create_transformer(64, self.drop_rate, eps)
        self.transformer_d = self._create_transformer(64, self.drop_rate, eps)

        # Add polarity branches
        self._add_polarity_branches(self.original_compatible, eps)

        # Rebuild picking branches with 64 input size
        self._rebuild_picking_branches(self.original_compatible, eps)

    def _rebuild_eqtp_components(self):
        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples,
        )

        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
        )

    def _create_transformer(self, input_size, drop_rate, eps):
        return Transformer(input_size=input_size, drop_rate=drop_rate, eps=eps)

    def _add_polarity_branches(self, original_compatible, eps):
        self.pol_lstms = []
        self.pol_attentions = []
        self.pol_decoders = []
        self.pol_convs = []

        for _ in range(2):  # Two polarity branches (U and D)
            if original_compatible == "conservative":
                lstm = CustomLSTM(ActivationLSTMCell, 64, 64, bidirectional=False)
            else:
                lstm = nn.LSTM(64, 64, bidirectional=False)
            self.pol_lstms.append(lstm)

            attention = SeqSelfAttention(input_size=64, attention_width=3, eps=eps)
            self.pol_attentions.append(attention)

            decoder = Decoder(
                input_channels=64,
                filters=self.filters[::-1],
                kernel_sizes=self.kernel_sizes[::-1],
                out_samples=self.in_samples,
                original_compatible=original_compatible,
            )
            self.pol_decoders.append(decoder)

            conv = nn.Conv1d(
                in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
            )
            self.pol_convs.append(conv)

        self.pol_lstms = nn.ModuleList(self.pol_lstms)
        self.pol_attentions = nn.ModuleList(self.pol_attentions)
        self.pol_decoders = nn.ModuleList(self.pol_decoders)
        self.pol_convs = nn.ModuleList(self.pol_convs)

    def _rebuild_picking_branches(self, original_compatible, eps):
        # Clear existing picking branches
        self.pick_lstms = nn.ModuleList()
        self.pick_attentions = nn.ModuleList()
        self.pick_decoders = nn.ModuleList()
        self.pick_convs = nn.ModuleList()

        for _ in range(self.classes):
            if original_compatible == "conservative":
                lstm = CustomLSTM(ActivationLSTMCell, 64, 64, bidirectional=False)
            else:
                lstm = nn.LSTM(64, 64, bidirectional=False)
            self.pick_lstms.append(lstm)

            attention = SeqSelfAttention(input_size=64, attention_width=3, eps=eps)
            self.pick_attentions.append(attention)

            decoder = Decoder(
                input_channels=64,
                filters=self.filters[::-1],
                kernel_sizes=self.kernel_sizes[::-1],
                out_samples=self.in_samples,
                original_compatible=original_compatible,
            )
            self.pick_decoders.append(decoder)

            conv = nn.Conv1d(
                in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
            )
            self.pick_convs.append(conv)

        self.pick_lstms = nn.ModuleList(self.pick_lstms)
        self.pick_attentions = nn.ModuleList(self.pick_attentions)
        self.pick_decoders = nn.ModuleList(self.pick_decoders)
        self.pick_convs = nn.ModuleList(self.pick_convs)

    def forward(self, x, logits=False):
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        # Shared encoder part
        x = self.encoder(x)
        x = self.res_cnn_stack(x)
        # Skip BiLSTM stack for EQTP
        x, _ = self.transformer_d0(x)
        x, _ = self.transformer_d(x)

        outputs = []

        # Polarity part
        for lstm, attention, decoder, conv in zip(
            self.pol_lstms, self.pol_attentions, self.pol_decoders, self.pol_convs
        ):
            polx = x.permute(2, 0, 1)
            polx = lstm(polx)[0]
            polx = self.dropout(polx)
            polx = polx.permute(1, 2, 0)
            polx, _ = attention(polx)
            polx = decoder(polx)
            if logits:
                predp = conv(polx)
            else:
                predp = torch.sigmoid(conv(polx))
            predp = torch.squeeze(predp, dim=1)
            outputs.append(predp)

        # Pick parts
        for lstm, attention, decoder, conv in zip(
            self.pick_lstms, self.pick_attentions, self.pick_decoders, self.pick_convs
        ):
            px = x.permute(2, 0, 1)
            px = lstm(px)[0]
            px = self.dropout(px)
            px = px.permute(1, 2, 0)
            px, _ = attention(px)
            px = decoder(px)
            if logits:
                pred = conv(px)
            else:
                pred = torch.sigmoid(conv(px))
            pred = torch.squeeze(pred, dim=1)
            outputs.append(pred)

        return tuple(outputs)

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete picks using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`
        and to discrete detections using :py:func:`~seisbench.models.base.WaveformModel.detections_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".
        Trigger onset thresholds for detections are derived from the argdict at key "detection_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks, list of detections
        """
        picks = sbu.PickList()
        for phase in self.phases:
            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )
        picks = sbu.PickList(sorted(picks))

        self._extract_polarities(annotations, picks, argdict)

        return sbu.ClassifyOutput(self.name, picks=picks)

    def _extract_polarities(
        self, annotations: obspy.Stream, picks: sbu.PickList, argdict: dict[str, Any]
    ):
        polarity_threshold = (
            argdict.get(
                "polarity_threshold", self._annotate_args.get("*_threshold")[1]
            ),
        )
        for pick in picks:
            if pick.phase == "P":
                t = pick.peak_time

                scores = {}
                for pol in "UD":
                    trace = annotations.select(
                        id=f"{pick.trace_id}.{self.__class__.__name__}_Polarity_{pol}"
                    ).slice(t - 5 / self.sampling_rate, t + 5 * self.sampling_rate)
                    if len(trace) != 1:
                        continue
                    trace = trace[0]
                    sample = int(
                        (t - trace.stats.starttime) * trace.stats.sampling_rate
                    )

                    segment = trace.data[
                        max(0, sample - 3) : sample + 3
                    ]  # Take a small tolerance around

                    scores[pol] = np.max(segment)

                if len(scores) != 2:
                    continue

                polarity = max(scores, key=scores.get)
                if scores[polarity] > polarity_threshold:
                    pick.polarity = polarity
                    pick.polarity_value = scores[polarity]
                else:
                    pick.polarity = "N"
                    pick.polarity_value = 1 - scores["U"] - scores["D"]

        return picks
