import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import seisbench.util as sbu
from .eqtransformer import EQTransformer, SeqSelfAttention, Decoder


class EQTP(EQTransformer):
    """
    The EQTP from Peng et al. (2025)

    It is an extended version of the EQTransformer model, which builds upon its
    phase picking capabilities by adding P-wave polarity determination functionality.

    This model is designed for processing three-component seismic waveform data,
    and can simultaneously output picking results for phases such as P-waves/S-waves
    and polarity determination results for P-waves (Up U/Down D/Unknown N).

    Implementation is adapted from the EQTransformer with SeisBench GitHub repository (https://github.com/seisbench/seisbench).

    the EQTP model can be instantiated via the `from_pretrained("NCEDC")` method.

    .. document_args:: seisbench.models EQTP

    """

    _annotate_args = EQTransformer._annotate_args.copy()
    _annotate_args["polarity_threshold"] = ("Polarity threshold", 0.3)
    _annotate_args["polarity_dict"] = ("The position of the polarity vector represents the polarity", {
        2: "N",
        0: "U",
        1: "D"
    })

    def __init__(
            self,
            in_channels=3,
            in_samples=12000,
            classes=2,
            phases="PS",
            cnn_blocks=5,
            res_cnn_blocks=5,
            lstm_blocks=3,
            drop_rate=0.3,
            original_compatible=False,
            sampling_rate=100,
            norm="std",
            **kwargs,
    ):

        self.cnn_blocks = cnn_blocks
        self.res_cnn_blocks = res_cnn_blocks

        # Update citation for EQTP
        citation = (
            "Peng L, Li L, Zeng X. "
            "A Microseismic Phase Picking and Polarity Determination Model Based on the Earthquake Transformer[J ]. Applied Sciences, 2025, 15(7): 3424."
            "https://doi.org/10.3390/app15073424"
        )

        # Initialize parent class
        super().__init__(
            in_channels=in_channels,
            in_samples=in_samples,
            classes=classes,
            phases=phases,
            lstm_blocks=lstm_blocks,
            drop_rate=drop_rate,
            original_compatible=original_compatible,
            sampling_rate=sampling_rate,
            norm=norm,
            **kwargs
        )

        # Override citation and labels for EQTP
        self._citation = citation
        self.labels = ["Polarity_U", "Polarity_D"] + list(phases)

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
        eps = 1e-7 if original_compatible else 1e-5
        self.transformer_d0 = self._create_transformer(64, drop_rate, eps)
        self.transformer_d = self._create_transformer(64, drop_rate, eps)

        # Add polarity branches
        self._add_polarity_branches(original_compatible, eps)

        # Rebuild picking branches with 64 input size
        self._rebuild_picking_branches(original_compatible, eps)

    def _rebuild_eqtp_components(self):
        from .eqtransformer import Encoder, ResCNNStack

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
        from .eqtransformer import Transformer
        return Transformer(input_size=input_size, drop_rate=drop_rate, eps=eps)

    def _add_polarity_branches(self, original_compatible, eps):
        from .base import ActivationLSTMCell, CustomLSTM

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
        """Rebuild picking branches with 64 input size for EQTP"""
        from .base import ActivationLSTMCell, CustomLSTM

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

        # Add polarity determination
        self.polarity_from_annotations(
            annotations.select(channel=f"{self.__class__.__name__}_Polarity_U"),
            annotations.select(channel=f"{self.__class__.__name__}_Polarity_D"),
            picks,
            argdict.get(
                f"polarity_threshold", self._annotate_args.get("polarity_threshold")[1]
            ),
        )

        return sbu.ClassifyOutput(self.name, picks=picks)

    def polarity_from_annotations(self, annotations_U, annotations_D, picks, polarity_threshold):
        polarity_dict = self._annotate_args.get("polarity_dict")[1]
        for trace_u, trace_d in zip(annotations_U, annotations_D):
            for pick in picks:
                if pick.phase == 'P':
                    pick_time = pick.peak_time - trace_u.stats.starttime
                    ind = int(pick_time / trace_u.stats.delta)
                    u_value = trace_u.data[ind]
                    d_value = trace_d.data[ind]
                    if u_value < polarity_threshold and d_value < polarity_threshold:
                        pick.add_polarity(polarity=polarity_dict[2], pol_value=1 - u_value - d_value)
                    elif u_value > d_value:
                        pick.add_polarity(polarity=polarity_dict[0], pol_value=u_value)
                    else:
                        pick.add_polarity(polarity=polarity_dict[1], pol_value=d_value)

    def get_model_args(self):
        model_args = super().get_model_args()
        model_args["cnn_blocks"] = self.cnn_blocks
        model_args["res_cnn_blocks"] = self.res_cnn_blocks
        return model_args