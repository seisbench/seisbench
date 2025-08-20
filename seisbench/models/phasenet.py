import json
from collections import defaultdict
from typing import Any, Optional

import obspy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from obspy import UTCDateTime
from packaging import version
from einops import rearrange
from einops.layers.torch import Rearrange

import seisbench.util as sbu

from .base import Conv1dSame, WaveformModel, _cache_migration_v0_v3


class PhaseNet(WaveformModel):
    """
    .. document_args:: seisbench.models PhaseNet

    :param filter_factor: Increase the number of filters used in each layer by this factor compared to the original
                          PhaseNet. Based on PhaseNetWC proposed by Naoi et al. (2024)
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (0, 0),
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)

    _weight_warnings = [
        (
            "ethz|geofon|instance|iquique|lendb|neic|scedc|stead",
            "1",
            "The normalization for this weight version is incorrect and will lead to degraded performance. "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/seisbench/seisbench/pull/188 .",
        ),
        (
            "diting",
            "1",
            "This version of the Diting picker uses an incorrect sampling rate (100 Hz instead of 50 Hz). "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/JUNZHU-SEIS/USTC-Pickers/issues/1 .",
        ),
    ]

    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases="NPS",
        sampling_rate=100,
        norm="std",
        filter_factor: int = 1,
        **kwargs,
    ):
        citation = (
            "Zhu, W., & Beroza, G. C. (2019). "
            "PhaseNet: a deep-neural-network-based seismic arrival-time picking method. "
            "Geophysical Journal International, 216(1), 261-273. "
            "https://doi.org/10.1093/gji/ggy423"
        )

        # PickBlue options
        for option in ("norm_amp_per_comp", "norm_detrend"):
            if option in kwargs:
                setattr(self, option, kwargs[option])
                del kwargs[option]
            else:
                setattr(self, option, False)

        super().__init__(
            citation=citation,
            in_samples=3001,
            output_type="array",
            pred_sample=(0, 3001),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.filter_factor = filter_factor
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        self.inc = nn.Conv1d(
            self.in_channels,
            self.filters_root * filter_factor,
            self.kernel_size,
            padding="same",
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root * filter_factor, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root * filter_factor
        for i in range(self.depth):
            filters = int(2**i * self.filters_root) * filter_factor
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        for i in range(self.depth - 1):
            filters = int(2 ** (3 - i) * self.filters_root) * filter_factor
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, logits=False):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(axis=-1, keepdims=True)
        if self.norm_detrend:
            batch = sbu.torch_detrend(batch)
        if self.norm_amp_per_comp:
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)
        else:
            if self.norm == "std":
                std = batch.std(axis=-1, keepdims=True)
                batch = batch / (std + 1e-10)
            elif self.norm == "peak":
                peak = batch.abs().max(axis=-1, keepdims=True)[0]
                batch = batch / (peak + 1e-10)

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        # Transpose predictions to correct shape
        batch = torch.transpose(batch, -1, -2)
        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            batch[:, :prenan] = np.nan
        if postnan > 0:
            batch[:, -postnan:] = np.nan
        return batch

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
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["norm_amp_per_comp"] = self.norm_amp_per_comp
        model_args["norm_detrend"] = self.norm_detrend

        return model_args

    @classmethod
    def from_pretrained_expand(
        cls, name, version_str="latest", update=False, force=False, wait_for_file=False
    ):
        """
        Load pretrained model with weights and copy the input channel weights that match the Z component to a new,
        4th dimension that is used to process the hydrophone component of the input trace.

        For further instructions, see :py:func:`~seisbench.models.base.SeisBenchModel.from_pretrained`. This method
        differs from :py:func:`~seisbench.models.base.SeisBenchModel.from_pretrained` in that it does not call helper
        functions to load the model weights. Instead it covers the same logic and, in addition, takes intermediate
        steps to insert a new `in_channels` dimension to the loaded model and copy weights.

        :param name: Model name prefix.
        :type name: str
        :param version_str: Version of the weights to load. Either a version string or "latest". The "latest" model is
                            the model with the highest version number.
        :type version_str: str
        :param force: Force execution of download callback, defaults to False
        :type force: bool, optional
        :param update: If true, downloads potential new weights file and config from the remote repository.
                       The old files are retained with their version suffix.
        :type update: bool
        :param wait_for_file: Whether to wait on partially downloaded files, defaults to False
        :type wait_for_file: bool, optional
        :return: Model instance
        :rtype: SeisBenchModel
        """
        cls._cleanup_local_repository()
        _cache_migration_v0_v3()

        if version_str == "latest":
            versions = cls.list_versions(name, remote=update)
            # Always query remote versions if cache is empty
            if len(versions) == 0:
                versions = cls.list_versions(name, remote=True)

            if len(versions) == 0:
                raise ValueError(f"No version for weight '{name}' available.")
            version_str = max(versions, key=version.parse)

        weight_path, metadata_path = cls._pretrained_path(name, version_str)

        cls._ensure_weight_files(
            name, version_str, weight_path, metadata_path, force, wait_for_file
        )

        if metadata_path.is_file():
            with open(metadata_path, "r") as f:
                weights_metadata = json.load(f)
        else:
            weights_metadata = {}
        model_args = weights_metadata.get("model_args", {})
        model_args["in_channels"] = 4
        cls._check_version_requirement(weights_metadata)
        model = cls(**model_args)

        model._weights_metadata = weights_metadata
        model._parse_metadata()

        state_dict = torch.load(weight_path)
        old_weight = state_dict["inc.weight"]
        state_dict["inc.weight"] = torch.zeros(
            old_weight.shape[0], old_weight.shape[1] + 1, old_weight.shape[2]
        ).type_as(old_weight)
        state_dict["inc.weight"][:, :3, ...] = old_weight
        state_dict["inc.weight"][:, 3, ...] = old_weight[:, 0, ...]
        model.load_state_dict(state_dict)
        return model


class PhaseNetLight(PhaseNet):
    """
    .. document_args:: seisbench.models PhaseNetLight

    PhaseNetLight is a slightly reduced version of PhaseNet. It is primarily included for compatibility reasons with
    an earlier, incomplete implementation of PhaseNet in SeisBench prior to v0.3.
    """

    _weight_warnings = [
        (
            "ethz|geofon|instance|iquique|lendb|neic|scedc|stead",
            "1",
            "The normalization for this weight version is incorrect and will lead to degraded performance. "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/seisbench/seisbench/pull/188 .",
        ),
    ]

    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases="NPS",
        sampling_rate=100,
        norm="std",
        **kwargs,
    ):
        citation = (
            "Zhu, W., & Beroza, G. C. (2019). "
            "PhaseNet: a deep-neural-network-based seismic arrival-time picking method. "
            "Geophysical Journal International, 216(1), 261-273. "
            "https://doi.org/10.1093/gji/ggy423"
        )

        # PickBlue options
        for option in ("norm_amp_per_comp", "norm_detrend"):
            if option in kwargs:
                setattr(self, option, kwargs[option])
                del kwargs[option]
            else:
                setattr(self, option, False)

        # Skip super call in favour of super-super class
        WaveformModel.__init__(
            self,
            citation=citation,
            in_samples=3001,
            output_type="array",
            pred_sample=(0, 3001),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
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

        self.out = nn.ConvTranspose1d(16, self.classes, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, logits=False):
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
        if logits:
            return x
        else:
            return self.softmax(x)


class VariableLengthPhaseNet(PhaseNet):
    """
    This version of PhaseNet has extended functionality:

    - The number of input samples can be changed.
      However, the number of layers in the model does not change, i.e., the receptive field stays unchanged.
      In addition, models will usually not perform well if applied to a different input length than trained on.
    - Output activation can be switched between softmax (all components sum to 1, i.e., no overlapping phases)
      and sigmoid (each component is normed individually between 0 and 1).
    - The axis for normalizing the waveforms before passing them to the model can be specified explicitly.

    .. document_args:: seisbench.models VariableLengthPhaseNet
    """

    _annotate_args = PhaseNet._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 0.5)

    def __init__(
        self,
        in_samples=600,
        in_channels=3,
        classes=3,
        phases="PSN",
        sampling_rate=100,
        norm="peak",
        norm_axis=(-1,),
        output_activation="softmax",
        empty=False,
        **kwargs,
    ):
        citation = (
            "Zhu, W., & Beroza, G. C. (2019). "
            "PhaseNet: a deep-neural-network-based seismic arrival-time picking method. "
            "Geophysical Journal International, 216(1), 261-273. "
            "https://doi.org/10.1093/gji/ggy423"
        )

        WaveformModel.__init__(
            self,
            citation=citation,
            in_samples=in_samples,
            output_type="array",
            pred_sample=(0, in_samples),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.norm_axis = tuple(norm_axis)
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        if output_activation == "softmax":
            self.output_activation = torch.nn.Softmax(dim=1)
        elif output_activation == "sigmoid":
            self.output_activation = torch.nn.Sigmoid()
        else:
            raise ValueError("Output activation needs to be softmax or sigmoid")

        # PhaseNet extra arguments
        self.norm_amp_per_comp = False
        self.norm_detrend = False

        if empty:
            self.inc = None
            self.in_bn = None
            self.down_branch = None
            self.up_branch = None
            self.out = None
        else:
            self.inc = nn.Conv1d(
                self.in_channels, self.filters_root, self.kernel_size, padding="same"
            )
            self.in_bn = nn.BatchNorm1d(8, eps=1e-3)

            self.down_branch = nn.ModuleList()
            self.up_branch = nn.ModuleList()

            last_filters = self.filters_root
            for i in range(self.depth):
                filters = int(2**i * self.filters_root)
                conv_same = nn.Conv1d(
                    last_filters, filters, self.kernel_size, padding="same", bias=False
                )
                last_filters = filters
                bn1 = nn.BatchNorm1d(filters, eps=1e-3)
                if i == self.depth - 1:
                    conv_down = None
                    bn2 = None
                else:
                    padding = self.kernel_size // 2
                    conv_down = nn.Conv1d(
                        filters,
                        filters,
                        self.kernel_size,
                        self.stride,
                        padding=padding,
                        bias=False,
                    )
                    bn2 = nn.BatchNorm1d(filters, eps=1e-3)

                self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

            for i in range(self.depth - 1):
                filters = int(2 ** (3 - i) * self.filters_root)
                conv_up = nn.ConvTranspose1d(
                    last_filters, filters, self.kernel_size, self.stride, bias=False
                )
                last_filters = filters
                bn1 = nn.BatchNorm1d(filters, eps=1e-3)
                conv_same = nn.Conv1d(
                    2 * filters, filters, self.kernel_size, padding="same", bias=False
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

                self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

            self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")

    def forward(self, x, logits=False):
        x = self._forward_single(x)

        if logits:
            return x
        else:
            return self.output_activation(x)

    def _forward_single(self, x):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        return self.out(x)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(axis=-1, keepdims=True)
        if self.norm_detrend:
            batch = sbu.torch_detrend(batch)

        if self.norm == "std":
            std = batch.std(axis=self.norm_axis, keepdims=True)
            batch = batch / (std + 1e-10)
        elif self.norm == "peak":
            peak = batch.abs().amax(axis=self.norm_axis, keepdims=True)
            batch = batch / (peak + 1e-10)

        return batch

    def get_model_args(self):
        model_args = super().get_model_args()

        model_args["in_samples"] = self.in_samples
        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["norm_axis"] = self.norm_axis
        model_args["output_activation"] = (
            self.output_activation.__class__.__name__.lower()
        )

        return model_args


class PhaseNetPlusEvent:
    """
    This class serves as container for storing event information with events detected on a single stations.
    """

    def __init__(
        self,
        trace_id: str,
        origin_time: UTCDateTime,
        center_time: UTCDateTime,
        confidence: float,
        p_pick: Optional[float] = None,
        s_pick: Optional[float] = None,
    ):
        self.trace_id = trace_id
        self.origin_time = origin_time
        self.center_time = center_time
        self.confidence = confidence
        self.p_pick = p_pick
        self.s_pick = s_pick

    def __lt__(self, other):
        """
        Compares start time and trace id in this order.
        """
        if self.origin_time == other.origin_time:
            return self.trace_id < other.trace_id
        return self.origin_time < other.origin_time

    def __str__(self):
        return (
            f"PhaseNetPlusEvent(trace_id={self.trace_id}, "
            f"origin_time={self.origin_time}, "
            f"center_time={self.center_time}, "
            f"confidence={self.confidence:.3f})"
        )


class PhaseNetPlus(WaveformModel):
    """
    This variant of PhaseNet extends the original phasenet architecture in several aspects:

    - Variable input length (using powers of 2)
    - Joint first-motion polarity estimation
    - Event detection framework (using the "event" mode in classify)

    .. document_args:: seisbench.models PhaseNetPlus

    :param log_scale: Log-scale the inputs
    :param add_polarity: Add first motion polarity output channel
    :param add_event: Add the event center and event time output channels
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["event_threshold"] = (
        "Detection threshold for event detections",
        0.3,
    )
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (100, 100),
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 0.5)
    _annotate_args["mode"] = ("The default mode for annotate", "pick")

    def __init__(
        self,
        log_scale: bool = True,
        add_polarity: bool = True,
        add_event: bool = True,
        sampling_rate: float = 100,
        **kwargs,
    ) -> None:
        citation = (
            "Zhu, W., Song, J., Wang, H., & Münchmeyer, J. (2025). "
            "Towards End-to-End Earthquake Monitoring Using a Multitask Deep Learning Model. "
            "arXiv preprint arXiv:2506.06939."
        )
        super().__init__(
            citation=citation,
            in_samples=1024,
            output_type="array",
            pred_sample=(0, 1024),
            labels=["N", "P", "S", "Polarity", "Event center", "Event time"],
            sampling_rate=sampling_rate,
            grouping="instrument",
            **kwargs,
        )

        self.add_polarity = add_polarity
        self.add_event = add_event
        self.log_scale = log_scale

        self.vpvs = 1.73  # vp/vs ratio for event extraction
        self.pick_tolerance = 3  # Tolerance for pick matching in event extraction

        self.backbone = UNet(
            channels=3,
            dim=16,
            out_dim=32,
            log_scale=log_scale,
            add_polarity=add_polarity,
            add_event=add_event,
        )
        self.phase_picker = UNetHead(32, 3, feature_name="phase")
        if self.add_polarity:
            self.polarity_picker = UNetHead(32, 1, feature_name="polarity")
        if self.add_event:
            self.event_detector = UNetHead(32, 1, feature_name="event")
            self.event_timer = EventHead(32, 1, feature_name="event")

    def get_model_args(self):
        model_args = super().get_model_args()

        for fixed_arg in [
            "citation",
            "in_samples",
            "output_type",
            "pred_sample",
            "labels",
            "grouping",
        ]:
            del model_args[fixed_arg]

        model_args["log_scale"] = self.log_scale
        model_args["add_polarity"] = self.add_polarity
        model_args["add_event"] = self.add_event
        model_args["sampling_rate"] = self.sampling_rate

        return model_args

    def forward(
        self, data: torch.Tensor, logits: bool = False
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(data)
        # features: (batch, channel, station, time)

        output_phase = self.phase_picker(features)
        if not logits:
            output_phase = torch.softmax(output_phase, dim=1)
        output = {"phase": output_phase}
        if self.add_event:
            output_event_center = self.event_detector(features)
            if not logits:
                output_event_center = torch.sigmoid(output_event_center)
            output["event_center"] = output_event_center
            output_event_time = self.event_timer(features)
            output["event_time"] = output_event_time
        if self.add_polarity:
            output_polarity = self.polarity_picker(features)
            if not logits:
                output_polarity = torch.sigmoid(output_polarity)
                output_polarity = (output_polarity - 0.5) * 2.0  # Convert to -1, 1
            output["polarity"] = output_polarity

        return output

    def _get_in_pred_samples(self, block: np.ndarray) -> tuple[int, tuple[int, int]]:
        in_samples = 2 ** int(
            np.log2(block.shape[-1])
        )  # The largest power of 2 below the block shape
        in_samples = min(
            max(in_samples, 2**10), 2**20
        )  # Enforce upper and lower bounds
        pred_sample = (0, in_samples)
        return in_samples, pred_sample

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        return batch.unsqueeze(-2)  # Add fake station dimension

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        y_phase = batch["phase"][..., 0, :]
        y_polarity = batch["polarity"][..., 0, :]
        y_center = torch.repeat_interleave(batch["event_center"][..., 0, :], 16, dim=-1)
        y_time = torch.repeat_interleave(batch["event_time"][..., 0, :], 16, dim=-1)

        y_full = torch.concat([y_phase, y_polarity, y_center, y_time], dim=1)

        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            y_full[..., :prenan] = np.nan
        if postnan > 0:
            y_full[..., -postnan:] = np.nan

        return torch.transpose(y_full, -1, -2)

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete picks (mode "pick") or events (mode "event").

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        mode = argdict.get("mode", self._annotate_args.get("mode")[1])
        if mode == "pick":
            return self.classify_aggregate_pick(annotations, argdict)
        elif mode == "event":
            return self.classify_aggregate_event(annotations, argdict)
        else:
            raise NotImplementedError(f"Mode '{mode}' unknown")

    def classify_aggregate_pick(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete thresholds using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = sbu.PickList()
        for phase in "PS":
            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )

        picks = sbu.PickList(sorted(picks))
        if self.add_polarity:
            picks = self._extract_polarities(annotations, picks, argdict)

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
                trace = annotations.select(
                    id=f"{pick.trace_id}.{self.__class__.__name__}_Polarity"
                ).slice(t - 0.1, t + 0.1)
                if len(trace) != 1:
                    continue
                trace = trace[0]
                sample = int((t - trace.stats.starttime) * trace.stats.sampling_rate)
                pick.polarity_value = trace.data[sample]
                if abs(pick.polarity_value) > polarity_threshold:
                    pick.polarity = "U" if pick.polarity_value > 0 else "D"

        return picks

    def classify_aggregate_event(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Extract events from the data. Uses the event_center attribute to identify the events, then determines picks
        based on this.
        """

        station_sets = defaultdict(obspy.Stream)
        for trace in annotations:
            trace_id = (
                f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"
            )
            station_sets[trace_id].append(trace)

        events = []
        p_picks = sbu.PickList()

        for trace_id, annotation_group in station_sets.items():
            centers = self.picks_from_annotations(
                annotation_group.select(
                    channel=f"{self.__class__.__name__}_Event center"
                ),
                argdict.get(
                    "event_threshold", self._annotate_args.get("event_threshold")[1]
                ),
                "center",
            )
            for center in centers:
                center_time = center.peak_time
                time_trace = annotation_group.select(
                    channel=f"{self.__class__.__name__}_Event time"
                ).slice(center_time - 0.1, center_time + 0.1)
                if len(time_trace) != 1:
                    continue
                time_trace = time_trace[0]
                time_sample = int(
                    (center_time - time_trace.stats.starttime)
                    * time_trace.stats.sampling_rate
                )
                dt = time_trace[time_sample] / time_trace.stats.sampling_rate
                origin_time = center_time - dt

                pick_time_est = {
                    "P": origin_time + (2 * dt / (1 + self.vpvs)),
                    "S": origin_time + (2 * self.vpvs * dt / (1 + self.vpvs)),
                }
                picks = {}
                for phase in "PS":
                    threshold = argdict.get(
                        f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                    )

                    time_est = pick_time_est[phase]
                    phase_trace = annotation_group.select(
                        channel=f"{self.__class__.__name__}_{phase}"
                    ).slice(
                        time_est - self.pick_tolerance, time_est + self.pick_tolerance
                    )
                    if len(phase_trace) != 1:
                        continue
                    phase_trace = phase_trace[0]

                    if np.max(phase_trace.data) > threshold:
                        sample_peak = np.argmax(phase_trace.data)
                        times = phase_trace.times()
                        pick = sbu.Pick(
                            trace_id=trace_id,
                            peak_value=np.max(phase_trace.data),
                            start_time=phase_trace.stats.starttime + times[sample_peak],
                            peak_time=phase_trace.stats.starttime + times[sample_peak],
                            phase=phase,
                        )
                        picks[f"{phase.lower()}_pick"] = pick

                        if phase == "P":
                            p_picks.append(pick)

                events.append(
                    PhaseNetPlusEvent(
                        trace_id=trace_id,
                        origin_time=origin_time,
                        center_time=center_time,
                        confidence=center.peak_value,
                        **picks,
                    )
                )

        if self.add_polarity:
            self._extract_polarities(annotations, p_picks, argdict)

        return sbu.ClassifyOutput(self.name, events=events)


def exists(x):
    return x is not None


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def upsample(dim, dim_out, stride=(1, 4)):
    return nn.ConvTranspose2d(dim, dim_out, stride, stride)


def downsample(dim, dim_out, stride=(1, 4)):
    return nn.Sequential(
        Rearrange("b c (h s1) (w s2) -> b (c s1 s2) h w", s1=stride[0], s2=stride[1]),
        nn.Conv2d(dim * stride[0] * stride[1], dim_out, 1),
    )


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 4),
        channels=3,
        kernel_size=(1, 7),
        scale_factor=None,
        moving_norm=(1024, 128),
        add_stft=False,
        log_scale=False,
        add_polarity=False,
        add_event=False,
        add_prompt=False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = tuple(k // 2 for k in kernel_size)
        self.scale_factor = (
            [tuple(map(lambda k: k // 2 + 1, kernel_size))] * 4
            if scale_factor is None
            else scale_factor
        )
        self.moving_norm = moving_norm
        self.log_scale = log_scale
        self.add_stft = add_stft
        self.add_polarity = add_polarity
        self.add_event = add_event
        self.add_prompt = add_prompt

        # determine dimensions
        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(
            input_channels, init_dim, kernel_size=self.kernel_size, padding=self.padding
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                        ),
                        nn.Identity(),
                        nn.Identity(),
                        (downsample(dim_in, dim_out, self.scale_factor[ind])),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, kernel_size=self.kernel_size, padding=self.padding
        )
        self.mid_upsample = upsample(mid_dim, dims[-2], stride=self.scale_factor[-1])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[:-1])):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out * 2,
                            dim_out,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                        ),
                        nn.Identity(),
                        nn.Identity(),
                        (
                            upsample(dim_out, dim_in, self.scale_factor[ind])
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 1)
                        ),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = ResnetBlock(
            init_dim * 2,
            self.out_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

        ## Polarity
        if self.add_polarity:
            self.polarity_init = nn.Conv2d(
                1, init_dim, kernel_size=self.kernel_size, padding=self.padding
            )
            dim_in, dim_out = dims[0], dims[1]
            self.polarity_encoder = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            ResnetBlock(
                                dim_in,
                                dim_in,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                            ),
                        ]
                    )
                ]
            )
            self.polarity_final = ResnetBlock(
                dim_in * 2,
                self.out_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
            )

        ## Event
        if self.add_event:
            self.event_feature_level = 3
            dim_in, dim_out = (
                dims[self.event_feature_level],
                dims[self.event_feature_level + 1],
            )
            self.event_final = nn.Sequential(
                ResnetBlock(
                    dim_in, dim_in, kernel_size=self.kernel_size, padding=self.padding
                ),
                upsample(
                    dim_in,
                    self.out_dim,
                    stride=self.scale_factor[self.event_feature_level],
                ),
            )

        ## STFT
        if self.add_stft:
            self.stft = STFT(n_fft=64 + 1, hop_length=self.scale_factor[0][-1])
            self.kernel_size_stft = [3, self.kernel_size[1]]
            self.padding_stft = [1, self.padding[1]]
            self.spec_init = nn.Sequential(
                nn.Conv2d(
                    channels,
                    init_dim,
                    kernel_size=self.kernel_size_stft,
                    padding=self.padding_stft,
                ),
            )
            self.spec_down = nn.ModuleList([])
            for ind, (dim_in, dim_out) in enumerate(in_out):
                if ind == 0:
                    continue
                self.spec_down.append(
                    nn.ModuleList(
                        [
                            ResnetBlock(
                                dim_in,
                                dim_in,
                                kernel_size=self.kernel_size_stft,
                                padding=self.padding_stft,
                            ),
                            downsample(dim_in, dim_out, self.scale_factor[ind]),
                            MergeFrequency(32),
                            MergeBranch(dim_in * 2, dim_in),
                        ]
                    )
                )
            self.mid_stft = nn.Sequential(
                ResnetBlock(
                    mid_dim,
                    mid_dim,
                    kernel_size=self.kernel_size_stft,
                    padding=self.padding_stft,
                ),
                MergeFrequency(32),
            )
            self.mid_merge = MergeBranch(mid_dim * 2, mid_dim)

    def log_transform(self, x):
        x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
        return x

    def moving_normalize(self, data, filter=1024, stride=128):
        nb, nch, nx, nt = data.shape

        padding = filter // 2

        with torch.no_grad():
            data_ = F.pad(data, (padding, padding, 0, 0), mode="reflect")
            mean = F.avg_pool2d(data_, kernel_size=(1, filter), stride=(1, stride))
            mean = F.interpolate(
                mean, scale_factor=(1, stride), mode="bilinear", align_corners=False
            )[:, :, :nx, :nt]
            data -= mean

            data_ = F.pad(data, (padding, padding, 0, 0), mode="reflect")
            std = F.avg_pool2d(
                torch.abs(data_), kernel_size=(1, filter), stride=(1, stride)
            )
            std = torch.mean(
                std, dim=(1,), keepdim=True
            )  ## keep relative amplitude between channels
            std = F.interpolate(
                std, scale_factor=(1, stride), mode="bilinear", align_corners=False
            )[:, :, :nx, :nt]
            std[std == 0.0] = 1.0
            data = data / std

        return data

    def forward(self, x):
        x = self.moving_normalize(
            x, filter=self.moving_norm[0], stride=self.moving_norm[1]
        )
        if self.log_scale:
            x = self.log_transform(x)

        # origin
        x_origin = x.clone()

        x = self.init_conv(x)
        if self.add_polarity:
            x_polarity = self.polarity_init(x_origin[:, -1:, :, :])

        r = x.clone()

        h = []

        for block1, _, _, downsample in self.downs:
            x = block1(x)
            h.append(x)
            x = downsample(x)

        if self.add_stft:
            nb, nc, nx, nt = x_origin.shape
            x_stft = x_origin.permute(0, 2, 1, 3).reshape(
                nb * nx, nc, nt
            )  # nb*nx, nc, nt
            x_stft = self.stft(x_stft)  # nb*nx, nc, nf, nt
            # if self.training:
            sgram = x_stft.clone()
            x_stft = self.spec_init(x_stft)
            for i, (
                block1,
                block2,
                attn,
                downsample,
                merge_freq,
                merge_branch,
            ) in enumerate(self.spec_down):
                x_stft = block1(x_stft)  # nb*nx, nc, nf, nt
                x_stft_m = merge_freq(x_stft)  # nb*nx, nc, nt
                x_stft_m = x_stft_m.view(nb, nx, *x_stft_m.shape[-2:]).permute(
                    0, 2, 1, 3
                )  # nb, nc, nx, nt
                h[i + 1] = merge_branch(h[i + 1], x_stft_m)

                x_stft = downsample(x_stft)

            x_stft = self.mid_stft(x_stft)
            x_stft = x_stft.view(nb, nx, *x_stft.shape[-2:]).permute(
                0, 2, 1, 3
            )  # nb, nc, nx, nt

        x = self.mid_block1(x)
        if self.add_stft:
            x = self.mid_merge(x, x_stft)
        x = self.mid_upsample(x)

        feature_level = 3
        for block1, _, _, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)

            if self.add_event and (feature_level == self.event_feature_level):
                x_event = x.clone()

            x = upsample(x)

            feature_level -= 1

        if self.add_polarity:
            x_polarity_aux = x.clone()

        x = torch.cat((x, r), dim=1)

        out_phase = self.final_res_block(x)

        # polarity
        if self.add_polarity:
            for block1 in self.polarity_encoder:
                x_polarity = block1[0](x_polarity)

            x_polarity = torch.cat((x_polarity, x_polarity_aux), dim=1)
            out_polarity = self.polarity_final(x_polarity)
        else:
            out_polarity = None

        # event
        if self.add_event:
            out_event = self.event_final(x_event)
        else:
            out_event = None

        out = {"phase": out_phase, "polarity": out_polarity, "event": out_event}

        if self.add_stft:
            out["spectrogram"] = sgram.squeeze(
                2
            )  ## nb, nc, nx, nf, nt -> nb, nc, nf, nt

        return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=(1, 7), padding=(0, 3)):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=padding)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        kernel_size=(1, 7),
        padding=(0, 3),
        time_emb_dim=None,
        classes_emb_dim=None,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2),
            )
            if exists(time_emb_dim) or exists(classes_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, kernel_size=kernel_size, padding=padding)
        self.block2 = Block(dim_out, dim_out, kernel_size=kernel_size, padding=padding)
        self.res_conv = (
            nn.Conv2d(dim, dim_out, (1, 1)) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class MergeFrequency(nn.Module):
    """
    Merge frequency dimension to 1 using a linear layer.
    """

    def __init__(self, dim_in):
        super().__init__()
        # self.linear = nn.Sequential(nn.Linear(dim_in, 1), nn.ReLU())
        self.linear = nn.Linear(dim_in, 1)

    def forward(self, x):
        # x: nb, nc, nf, nt
        x = x.permute(0, 1, 3, 2)  # nb, nc, nt, nf
        x = self.linear(x).squeeze(-1)  # nb, nc, nt
        return x


class MergeBranch(nn.Module):
    """
    Merge two branches of the same dimension.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # self.conv = nn.Sequential(nn.Conv2d(dim_in, dim_out, 1), nn.ReLU())
        self.conv = nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x1, x2):
        return self.conv(torch.cat((x1, x2), dim=1))


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=128 + 1,
        hop_length=4,
        window_fn=torch.hann_window,
        magnitude=True,
        normalize_freq=False,
        discard_zero_freq=True,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.magnitude = magnitude
        self.discard_zero_freq = discard_zero_freq
        self.normalize_freq = normalize_freq
        self.register_buffer("window", window_fn(n_fft))

    def forward(self, x):
        """
        x: bt, ch, nt
        """
        nb, nc, nt = x.shape
        x = x.view(-1, nt)  # nb*nc, nt
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            window=self.window,
            hop_length=self.hop_length,
            center=True,
            return_complex=True,
        )
        stft = torch.view_as_real(stft)
        if self.discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
        nf, nt, _ = stft.shape[-3:]
        if self.magnitude:
            stft = torch.norm(stft, dim=-1, keepdim=False).view(
                nb, nc, nf, nt
            )  # nb, nc, nf, nt
        else:
            stft = stft.view(nb, nc, nf, nt, 2)  # nb, nc, nf, nt, 2
            stft = stft.permute(0, 1, 4, 2, 3).view(
                nb, nc * 2, nf, nt
            )  # nb, nc*2, nf, nt

        if self.normalize_freq:
            vmax = torch.max(torch.abs(stft), dim=-2, keepdim=True)[0]
            vmax[vmax == 0.0] = 1.0
            stft = stft / vmax

        return stft


class UNetHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(1, 1),
        padding=(0, 0),
        feature_name: str = "phase",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_name = feature_name
        self.layers = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, features):
        x = features[self.feature_name]
        x = self.layers(x)
        return x


class EventHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(1, 1),
        padding=(0, 0),
        scaling=1000.0,
        feature_name: str = "event",
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_name = feature_name
        self.scaling = scaling

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.LeakyReLU(),
        )

    def forward(self, features):
        x = features[self.feature_name]
        x = self.layers(x) * self.scaling

        return x
