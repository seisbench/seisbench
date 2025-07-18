import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

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
