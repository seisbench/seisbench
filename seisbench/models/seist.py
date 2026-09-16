import math
from collections import OrderedDict
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import seisbench.util as sbu

from .base import WaveformModel

# Architecture presets for the three model sizes published with SeisT.
# The dropout rates correspond to the detection/picking configuration
# (seist_[s|m|l]_dpk) in the original repository.
_SEIST_PRESETS = {
    "s": dict(
        stem_channels=[16, 8, 16, 16],
        stem_kernel_sizes=[11, 5, 5, 7],
        stem_strides=[2, 1, 1, 2],
        layer_blocks=[2, 2, 3, 2],
        layer_channels=[16, 24, 32, 64],
        attn_blocks=[1, 1, 1, 1],
        stage_aggr_ratios=[2, 2, 2, 2],
        attn_aggr_ratios=[8, 4, 2, 1],
        head_dims=[8, 8, 8, 16],
        msmc_kernel_sizes=[5, 7],
        path_drop_rate=0.1,
        attn_drop_rate=0.1,
        key_drop_rate=0.1,
        mlp_drop_rate=0.1,
        other_drop_rate=0.1,
        attn_ratio=0.6,
        mlp_ratio=2,
    ),
    "m": dict(
        stem_channels=[16, 8, 16, 16],
        stem_kernel_sizes=[11, 5, 5, 7],
        stem_strides=[2, 1, 1, 2],
        layer_blocks=[2, 3, 6, 2],
        layer_channels=[24, 32, 64, 96],
        attn_blocks=[1, 1, 1, 1],
        stage_aggr_ratios=[2, 2, 2, 2],
        attn_aggr_ratios=[8, 4, 2, 1],
        head_dims=[8, 8, 16, 32],
        msmc_kernel_sizes=[5, 7],
        path_drop_rate=0.2,
        attn_drop_rate=0.2,
        key_drop_rate=0.2,
        mlp_drop_rate=0.2,
        other_drop_rate=0.2,
        attn_ratio=0.6,
        mlp_ratio=2,
    ),
    "l": dict(
        stem_channels=[16, 8, 16, 16],
        stem_kernel_sizes=[11, 5, 5, 7],
        stem_strides=[2, 1, 1, 2],
        layer_blocks=[2, 3, 6, 3],
        layer_channels=[32, 32, 64, 128],
        attn_blocks=[1, 1, 2, 1],
        stage_aggr_ratios=[2, 2, 2, 2],
        attn_aggr_ratios=[8, 4, 2, 1],
        head_dims=[8, 8, 16, 32],
        msmc_kernel_sizes=[3, 5, 7, 11],
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        attn_ratio=0.6,
        mlp_ratio=3,
    ),
}


class SeisT(WaveformModel):
    """
    SeisT (Seismogram Transformer) from Li et al. (2024) for earthquake detection and seismic phase picking.

    Implementation adapted from the Github repository https://github.com/senli1073/SeisT

    SeisT is a hierarchical encoder-decoder network. A multi-path depthwise-separable convolution stem is followed
    by four stages. Every stage starts with a local-aware aggregation layer that halves the temporal resolution.
    The remaining blocks of a stage are multi-scale mixed convolution blocks, i.e., grouped convolutions with
    different kernel sizes running in parallel, and multi-path transformer layers, which combine a self-attention
    branch with locally aggregated keys and values and a grouped convolution branch.
    A light-weight decoder upsamples the deepest feature map back to the input length.
    The model outputs one detection channel and one channel per phase, each with an independent sigmoid activation.

    The architecture is available in three sizes, SeisT-S, SeisT-M and SeisT-L, which can be selected with the
    ``variant`` argument. All architecture hyperparameters can be overwritten individually.
    As the model is fully convolutional/attentional, it accepts arbitrary input lengths.
    Nonetheless, models will usually perform best when applied to the input length they were trained on.

    .. document_args:: seisbench.models SeisT

    :param in_channels: Number of input channels, by default 3.
    :param in_samples: Number of input samples per channel, by default 6000.
                       The model expects input shape (in_channels, in_samples).
    :param classes: Number of output classes, by default 2. The detection channel is not counted.
    :param phases: Phase hints for the classes, by default "PS". Can be None.
    :param variant: Model size, one of "s", "m" or "l". Defines the default architecture hyperparameters.
    :param sampling_rate: Sampling rate of the model, by default 100 Hz.
    :param norm: Data normalization strategy, either "std" or "peak". Normalization is applied per component.
    :param stem_channels: Number of output channels of each stem layer.
    :param stem_kernel_sizes: Base kernel size of each stem layer.
    :param stem_strides: Stride of each stem layer.
    :param layer_blocks: Number of blocks in each stage.
    :param layer_channels: Number of channels in each stage.
    :param attn_blocks: Number of transformer blocks at the end of each stage.
                        The remaining blocks are multi-scale mixed convolution blocks.
    :param stage_aggr_ratios: Temporal downsampling factor of the aggregation layer at the start of each stage.
    :param attn_aggr_ratios: Aggregation factor for keys and values in the attention blocks of each stage.
    :param head_dims: Dimension of a single attention head in each stage.
                      Also defines the group size of the grouped convolutions.
    :param msmc_kernel_sizes: Kernel sizes of the parallel paths in the multi-scale mixed convolution blocks.
    :param path_drop_rate: Maximum stochastic depth rate. The rate increases linearly over the blocks.
    :param attn_drop_rate: Dropout rate applied to the attention weights.
    :param key_drop_rate: Dropout rate applied to the attention keys.
    :param mlp_drop_rate: Dropout rate in the feed-forward layers.
    :param other_drop_rate: Dropout rate applied to the output projection of the attention.
    :param attn_ratio: Fraction of the stage channels processed by the attention branch of a transformer block.
                       The remaining channels are processed by the grouped convolution branch.
    :param mlp_ratio: Expansion ratio of the feed-forward layers.
    :param qkv_bias: If true, use biases in the query, key, value and output projections of the attention.
    :param mlp_bias: If true, use biases in the feed-forward layers.
    :param kwargs: Keyword arguments passed to the constructor of :py:class:`WaveformModel`.
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["detection_threshold"] = ("Detection threshold", 0.5)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (500, 500),
    )
    _annotate_args["stacking"] = (
        "Stacking method for overlapping windows (only for window prediction models). "
        "Options are 'max' and 'avg'. ",
        "max",
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 3000)

    _arch_keys = tuple(_SEIST_PRESETS["m"].keys()) + ("qkv_bias", "mlp_bias")

    def __init__(
        self,
        in_channels: int = 3,
        in_samples: int = 6000,
        classes: int = 2,
        phases: Optional[str | Sequence[str]] = "PS",
        variant: str = "m",
        sampling_rate: float = 100,
        norm: str = "std",
        stem_channels: Optional[Sequence[int]] = None,
        stem_kernel_sizes: Optional[Sequence[int]] = None,
        stem_strides: Optional[Sequence[int]] = None,
        layer_blocks: Optional[Sequence[int]] = None,
        layer_channels: Optional[Sequence[int]] = None,
        attn_blocks: Optional[Sequence[int]] = None,
        stage_aggr_ratios: Optional[Sequence[int]] = None,
        attn_aggr_ratios: Optional[Sequence[int]] = None,
        head_dims: Optional[Sequence[int]] = None,
        msmc_kernel_sizes: Optional[Sequence[int]] = None,
        path_drop_rate: Optional[float] = None,
        attn_drop_rate: Optional[float] = None,
        key_drop_rate: Optional[float] = None,
        mlp_drop_rate: Optional[float] = None,
        other_drop_rate: Optional[float] = None,
        attn_ratio: Optional[float] = None,
        mlp_ratio: Optional[float] = None,
        qkv_bias: bool = True,
        mlp_bias: bool = True,
        **kwargs,
    ):
        citation = (
            "Li, S., Yang, X., Cao, A., Wang, C., Liu, Y., Liu, Y., & Niu, Q. (2024). "
            "SeisT: A Foundational Deep-Learning Model for Earthquake Monitoring Tasks. "
            "IEEE Transactions on Geoscience and Remote Sensing, 62, 1-15. "
            "https://doi.org/10.1109/TGRS.2024.3371503"
        )

        if phases is not None and len(phases) != classes:
            raise ValueError(
                f"Number of classes ({classes}) does not match number of phases ({len(phases)})."
            )
        if phases is None:
            labels = ["Detection"] + [str(i) for i in range(classes)]
        else:
            labels = ["Detection"] + list(phases)

        super().__init__(
            citation=citation,
            output_type="array",
            in_samples=in_samples,
            pred_sample=(0, in_samples),
            labels=labels,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        if variant not in _SEIST_PRESETS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: {list(_SEIST_PRESETS.keys())}"
            )
        if norm not in ("std", "peak"):
            raise ValueError(f"Unknown norm '{norm}'. Options are 'std' and 'peak'.")

        self.in_channels = in_channels
        self.classes = classes
        self._phases = phases
        self.variant = variant
        self.norm = norm

        # Resolve architecture: explicit arguments overwrite the preset
        arch = dict(_SEIST_PRESETS[variant])
        explicit = dict(
            stem_channels=stem_channels,
            stem_kernel_sizes=stem_kernel_sizes,
            stem_strides=stem_strides,
            layer_blocks=layer_blocks,
            layer_channels=layer_channels,
            attn_blocks=attn_blocks,
            stage_aggr_ratios=stage_aggr_ratios,
            attn_aggr_ratios=attn_aggr_ratios,
            head_dims=head_dims,
            msmc_kernel_sizes=msmc_kernel_sizes,
            path_drop_rate=path_drop_rate,
            attn_drop_rate=attn_drop_rate,
            key_drop_rate=key_drop_rate,
            mlp_drop_rate=mlp_drop_rate,
            other_drop_rate=other_drop_rate,
            attn_ratio=attn_ratio,
            mlp_ratio=mlp_ratio,
        )
        for key, value in explicit.items():
            if value is not None:
                arch[key] = list(value) if isinstance(value, (list, tuple)) else value
        arch["qkv_bias"] = qkv_bias
        arch["mlp_bias"] = mlp_bias
        self.arch = arch

        self.backbone = SeismogramTransformer(
            in_channels=in_channels, out_channels=1 + classes, **arch
        )

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch, in_channels, samples)
        :param logits: If true, return the logits instead of the sigmoid activations.
        :return: Tensor of shape (batch, 1 + classes, samples) with the detection channel first,
                 followed by one channel per phase.
        """
        x = self.backbone(x)
        if logits:
            return x
        return torch.sigmoid(x)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(dim=-1, keepdim=True)
        if self.norm == "std":
            std = batch.std(dim=-1, keepdim=True)
            batch = batch / (std + 1e-10)
        elif self.norm == "peak":
            peak = batch.abs().amax(dim=-1, keepdim=True)
            batch = batch / (peak + 1e-10)
        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        # Transpose predictions to (batch, samples, channels)
        batch = torch.transpose(batch, -1, -2)
        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            batch[:, :prenan] = np.nan
        if postnan > 0:
            batch[:, -postnan:] = np.nan
        return batch

    @property
    def phases(self):
        if self._phases is not None:
            return self._phases
        else:
            return [str(i) for i in range(self.classes)]

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

        detections = self.detections_from_annotations(
            annotations.select(channel=f"{self.__class__.__name__}_Detection"),
            argdict.get(
                "detection_threshold", self._annotate_args.get("detection_threshold")[1]
            ),
        )

        return sbu.ClassifyOutput(self.name, picks=picks, detections=detections)

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["in_samples"] = self.in_samples
        model_args["classes"] = self.classes
        model_args["phases"] = self._phases
        model_args["variant"] = self.variant
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        # Store the resolved architecture so that saved weights do not depend on the presets
        for key in self._arch_keys:
            model_args[key] = self.arch[key]

        return model_args


def _auto_pad_1d(
    x: torch.Tensor, kernel_size: int, stride: int = 1, padding_value: float = 0.0
) -> torch.Tensor:
    """
    Pads the last dimension such that a subsequent convolution returns ceil(len / stride) samples.
    Replaces padding="same" for strided convolutions, which is not supported by PyTorch.
    """
    if kernel_size < stride:
        raise ValueError(
            f"kernel_size ({kernel_size}) must be greater than or equal to stride ({stride})."
        )
    pds = (stride - (x.shape[-1] % stride)) % stride + kernel_size - stride
    return F.pad(x, (pds // 2, pds - pds // 2), "constant", padding_value)


def _make_divisible(v: int, divisor: int) -> int:
    """
    Returns the closest integer to v that is divisible by divisor, while not decreasing by more than 10%.
    Adapted from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DropPath(nn.Module):
    """
    Stochastic depth per sample (Huang et al., 2016). Adapted from the timm library.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            mask.div_(keep_prob)
        return x * mask


class LocalAwareAggregationBlock(nn.Module):
    """
    Downsamples the sequence by the sum of average and max pooling followed by a pointwise projection.
    """

    def __init__(self, in_dim, out_dim, kernel_size, norm_layer):
        super().__init__()

        if kernel_size > 1:
            self.avg_pool = nn.AvgPool1d(kernel_size, ceil_mode=True)
            self.max_pool = nn.MaxPool1d(kernel_size, ceil_mode=True)
        else:
            self.avg_pool = self.max_pool = None

        self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        if self.avg_pool is not None:
            x = self.avg_pool(x) + self.max_pool(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class MLP(nn.Module):
    """
    Feed-forward layer. Uses pointwise convolutions instead of linear layers to avoid transposes.
    """

    def __init__(self, in_dim, out_dim, mlp_ratio, bias, mlp_drop_rate, act_layer):
        super().__init__()

        ffwd_dim = int(in_dim * mlp_ratio)
        self.lin0 = nn.Conv1d(in_dim, ffwd_dim, kernel_size=1, bias=bias)
        self.act = act_layer()
        self.lin1 = nn.Conv1d(ffwd_dim, out_dim, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(mlp_drop_rate)

    def forward(self, x):
        x = self.lin0(x)
        x = self.act(x)
        x = self.lin1(x)
        x = self.dropout(x)
        return x


class DSConvNormAct(nn.Module):
    """
    Depthwise separable convolution with normalization and activation.
    """

    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()

        self.in_proj = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        self.dconv = nn.Conv1d(
            in_dim, in_dim, kernel_size, stride=stride, groups=in_dim, bias=False
        )
        self.pconv = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.in_proj(x)
        x = _auto_pad_1d(x, self.dconv.kernel_size[0], self.dconv.stride[0])
        x = self.dconv(x)
        x = self.pconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class StemBlock(nn.Module):
    """
    Stem layer with multiple depthwise separable convolution paths of increasing kernel size.
    """

    def __init__(
        self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer, npath=3
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                DSConvNormAct(
                    in_dim,
                    out_dim,
                    kernel_size + 4 * delta_k,
                    stride,
                    act_layer,
                    norm_layer,
                )
                for delta_k in range(npath)
            ]
        )
        self.out_proj = nn.Conv1d(npath * out_dim, out_dim, kernel_size=1, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.out_proj(x)
        x = self.norm(x)
        return x


class GroupConvBlock(nn.Module):
    """
    Residual block with a grouped convolution followed by a feed-forward layer.
    """

    def __init__(
        self,
        io_dim,
        groups,
        kernel_size,
        path_drop_rate,
        mlp_drop_rate,
        mlp_ratio,
        mlp_bias,
        act_layer,
        norm_layer,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            io_dim, io_dim, kernel_size, stride=1, groups=groups, bias=False
        )
        self.norm0 = norm_layer(io_dim)
        self.act = act_layer()
        self.proj = nn.Conv1d(io_dim, io_dim, kernel_size=1, bias=False)
        self.droppath0 = DropPath(path_drop_rate)

        self.norm1 = norm_layer(io_dim)
        self.mlp = MLP(
            in_dim=io_dim,
            out_dim=io_dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            mlp_drop_rate=mlp_drop_rate,
            act_layer=act_layer,
        )
        self.droppath1 = DropPath(path_drop_rate)

    def forward(self, x):
        x1 = _auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x1 = self.conv(x1)
        x1 = self.norm0(x1)
        x1 = self.act(x1)
        x1 = self.proj(x1)
        x = x + self.droppath0(x1)

        x1 = self.norm1(x)
        x1 = self.mlp(x1)
        x = x + self.droppath1(x1)

        return x


class MultiScaleMixedConv(nn.Module):
    """
    Splits the channels into parallel grouped convolution paths with different kernel sizes.
    """

    def __init__(
        self,
        io_dim,
        groups,
        kernel_sizes,
        path_drop_rate,
        mlp_drop_rate,
        mlp_ratio,
        mlp_bias,
        act_layer,
        norm_layer,
    ):
        super().__init__()

        group_size = io_dim // groups
        dims = []
        self.projs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            dim = _make_divisible(
                (io_dim - sum(dims)) // (len(kernel_sizes) - len(dims)), group_size
            )
            if dim <= 0:
                raise ValueError(
                    "Channel split for the multi-scale mixed convolution failed. "
                    "Increase the number of channels or reduce the number of kernel sizes."
                )
            dims.append(dim)

            self.projs.append(nn.Conv1d(io_dim, dim, kernel_size=1, bias=False))
            self.norms.append(norm_layer(dim))
            self.convs.append(
                GroupConvBlock(
                    io_dim=dim,
                    groups=dim // group_size,
                    kernel_size=kernel_size,
                    path_drop_rate=path_drop_rate,
                    mlp_drop_rate=mlp_drop_rate,
                    mlp_ratio=mlp_ratio,
                    mlp_bias=mlp_bias,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            )

        if sum(dims) != io_dim:
            raise ValueError(
                f"Channel split {dims} of the multi-scale mixed convolution does not sum up to {io_dim}."
            )

        self.out_norm = norm_layer(io_dim)

    def forward(self, x):
        outs = []
        for proj, norm, conv in zip(self.projs, self.norms, self.convs):
            xi = norm(proj(x))
            xi = xi + conv(xi)
            outs.append(xi)

        x = torch.cat(outs, dim=1)
        x = self.out_norm(x)

        return x


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention. Keys and values are computed on a locally aggregated (downsampled) sequence.
    """

    def __init__(
        self,
        io_dim,
        head_dim,
        qkv_bias,
        attn_drop_rate,
        key_drop_rate,
        proj_drop_rate,
        attn_aggr_ratio,
        norm_layer,
    ):
        super().__init__()

        self.num_heads = io_dim // head_dim

        if attn_aggr_ratio > 1:
            self.aggr = LocalAwareAggregationBlock(
                in_dim=io_dim,
                out_dim=io_dim,
                kernel_size=attn_aggr_ratio,
                norm_layer=norm_layer,
            )
            self.norm = norm_layer(io_dim)
        else:
            self.aggr = nn.Identity()
            self.norm = nn.Identity()

        self.q_proj = nn.Conv1d(io_dim, io_dim, kernel_size=1, bias=qkv_bias)
        self.k_proj = nn.Conv1d(io_dim, io_dim, kernel_size=1, bias=qkv_bias)
        self.v_proj = nn.Conv1d(io_dim, io_dim, kernel_size=1, bias=qkv_bias)
        self.k_dropout = nn.Dropout(key_drop_rate)
        self.attn_dropout = nn.Dropout(attn_drop_rate)

        self.out_proj = nn.Conv1d(io_dim, io_dim, kernel_size=1, bias=qkv_bias)
        self.proj_dropout = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        n, c, length = x.shape
        head_dim = c // self.num_heads

        q = self.q_proj(x).view(n, self.num_heads, head_dim, length)

        x = self.aggr(x)
        x = self.norm(x)

        k = self.k_proj(x).view(n, self.num_heads, head_dim, -1)
        v = self.v_proj(x).view(n, self.num_heads, head_dim, -1)
        k = self.k_dropout(k)

        q = q / math.sqrt(head_dim)
        attn = (q.transpose(-1, -2) @ k).softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(n, c, length)

        x = self.out_proj(x)
        x = self.proj_dropout(x)

        return x


class MultiPathTransformerLayer(nn.Module):
    """
    Transformer layer with an attention path and a grouped convolution path processing disjoint channel subsets.
    """

    def __init__(
        self,
        io_dim,
        path_drop_rate,
        attn_aggr_ratio,
        attn_ratio,
        head_dim,
        qkv_bias,
        mlp_ratio,
        mlp_bias,
        attn_drop_rate,
        key_drop_rate,
        attn_out_drop_rate,
        mlp_drop_rate,
        act_layer,
        norm_layer,
    ):
        super().__init__()

        if not 0 <= attn_ratio <= 1:
            raise ValueError(f"attn_ratio must be between 0 and 1, got {attn_ratio}.")

        self.attn_out_dim = (
            _make_divisible(int(io_dim * attn_ratio), head_dim) if attn_ratio > 0 else 0
        )
        self.conv_out_dim = max(io_dim - self.attn_out_dim, 0)

        self.has_attn = self.attn_out_dim > 0
        self.has_conv = self.conv_out_dim > 0

        if self.has_attn:
            self.attn_proj = nn.Conv1d(
                io_dim, self.attn_out_dim, kernel_size=1, bias=False
            )
            self.norm0 = norm_layer(self.attn_out_dim)
            self.attention = AttentionBlock(
                io_dim=self.attn_out_dim,
                head_dim=head_dim,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                key_drop_rate=key_drop_rate,
                proj_drop_rate=attn_out_drop_rate,
                attn_aggr_ratio=attn_aggr_ratio,
                norm_layer=norm_layer,
            )
            self.attn_droppath = DropPath(path_drop_rate * attn_ratio)
        else:
            self.attn_proj = self.norm0 = self.attention = self.attn_droppath = None

        if self.has_conv:
            self.conv_proj = nn.Conv1d(
                io_dim, self.conv_out_dim, kernel_size=1, bias=False
            )
            self.norm1 = norm_layer(self.conv_out_dim)
            self.gconv = GroupConvBlock(
                io_dim=self.conv_out_dim,
                groups=self.conv_out_dim // head_dim,
                kernel_size=3,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            self.gconv_droppath = DropPath(path_drop_rate * (1 - attn_ratio))
        else:
            self.conv_proj = self.norm1 = self.gconv = self.gconv_droppath = None

        self.norm2 = norm_layer(io_dim)
        self.mlp = MLP(
            in_dim=io_dim,
            out_dim=io_dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            mlp_drop_rate=mlp_drop_rate,
            act_layer=act_layer,
        )
        self.mlp_droppath = DropPath(path_drop_rate)

    def forward(self, x):
        outs = []
        if self.has_attn:
            x1 = self.norm0(self.attn_proj(x))
            x1 = x1 + self.attn_droppath(self.attention(x1))
            outs.append(x1)

        if self.has_conv:
            x2 = self.norm1(self.conv_proj(x))
            x2 = x2 + self.gconv_droppath(self.gconv(x2))
            outs.append(x2)

        x = torch.cat(outs, dim=1)
        x = self.norm2(x)
        x = x + self.mlp_droppath(self.mlp(x))

        return x


class HeadDetectionPicking(nn.Module):
    """
    Decoder for detection and phase picking.
    Upsamples the feature map in several steps with linear interpolation followed by convolutions.
    """

    def __init__(
        self,
        feature_channels,
        layer_channels,
        layer_kernel_sizes,
        act_layer,
        norm_layer,
        out_channels=1,
    ):
        super().__init__()

        if len(layer_channels) != len(layer_kernel_sizes):
            raise ValueError(
                "layer_channels and layer_kernel_sizes must have equal length."
            )

        self.depth = len(layer_channels)
        self.up_layers = nn.ModuleList()

        for inc, outc, kers in zip(
            [feature_channels] + list(layer_channels[:-1]),
            list(layer_channels[:-1]) + [out_channels * 2],
            layer_kernel_sizes,
        ):
            self.up_layers.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv1d(inc, outc, kernel_size=kers)),
                            ("norm", norm_layer(outc)),
                            ("act", act_layer()),
                        ]
                    )
                )
            )

        self.out_conv = nn.Conv1d(
            out_channels * 2, out_channels, kernel_size=7, padding=3
        )

    def _upsampling_sizes(self, in_size: int, out_size: int) -> list[int]:
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, out_size):
        up_sizes = self._upsampling_sizes(in_size=x.shape[-1], out_size=out_size)
        for layer, upsize in zip(self.up_layers, up_sizes):
            x = F.interpolate(x, size=upsize, mode="linear")
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        return x


class SeismogramTransformer(nn.Module):
    """
    SeisT backbone with a detection/picking head. Returns logits.

    For a description of the parameters see :py:class:`SeisT`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stem_channels,
        stem_kernel_sizes,
        stem_strides,
        layer_blocks,
        layer_channels,
        attn_blocks,
        stage_aggr_ratios,
        attn_aggr_ratios,
        head_dims,
        msmc_kernel_sizes,
        path_drop_rate,
        attn_drop_rate,
        key_drop_rate,
        mlp_drop_rate,
        other_drop_rate,
        attn_ratio,
        mlp_ratio,
        qkv_bias=True,
        mlp_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
    ):
        super().__init__()

        if not len(stem_channels) == len(stem_kernel_sizes) == len(stem_strides):
            raise ValueError("Stem configuration lists must have equal length.")
        if not (
            len(layer_blocks)
            == len(layer_channels)
            == len(stage_aggr_ratios)
            == len(attn_aggr_ratios)
            == len(attn_blocks)
            == len(head_dims)
        ):
            raise ValueError("Stage configuration lists must have equal length.")

        self.stem = nn.Sequential(
            *[
                StemBlock(
                    in_dim=inc,
                    out_dim=outc,
                    kernel_size=kers,
                    stride=strd,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for inc, outc, kers, strd in zip(
                    [in_channels] + list(stem_channels[:-1]),
                    stem_channels,
                    stem_kernel_sizes,
                    stem_strides,
                )
            ]
        )

        # Stochastic depth rate increases linearly with depth
        pdprs = [x.item() for x in torch.linspace(0, path_drop_rate, sum(layer_blocks))]

        self.encoder_layers = nn.ModuleList()
        for i, (
            num_blocks,
            inc,
            lc,
            num_attns,
            aggr_ratio,
            attn_aggr_ratio,
            head_dim,
        ) in enumerate(
            zip(
                layer_blocks,
                list(stem_channels[-1:]) + list(layer_channels),
                layer_channels,
                attn_blocks,
                stage_aggr_ratios,
                attn_aggr_ratios,
                head_dims,
            )
        ):
            layer_modules = [
                LocalAwareAggregationBlock(
                    in_dim=inc,
                    out_dim=lc,
                    kernel_size=aggr_ratio,
                    norm_layer=norm_layer,
                )
            ]

            for j in range(num_blocks):
                pdpr = pdprs[sum(layer_blocks[:i]) + j]
                if j >= num_blocks - num_attns:
                    block = MultiPathTransformerLayer(
                        io_dim=lc,
                        path_drop_rate=pdpr,
                        attn_aggr_ratio=attn_aggr_ratio,
                        attn_ratio=attn_ratio,
                        head_dim=head_dim,
                        qkv_bias=qkv_bias,
                        mlp_ratio=mlp_ratio,
                        mlp_bias=mlp_bias,
                        attn_drop_rate=attn_drop_rate,
                        key_drop_rate=key_drop_rate,
                        attn_out_drop_rate=other_drop_rate,
                        mlp_drop_rate=mlp_drop_rate,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                    )
                else:
                    block = MultiScaleMixedConv(
                        io_dim=lc,
                        groups=lc // head_dim,
                        kernel_sizes=msmc_kernel_sizes,
                        path_drop_rate=pdpr,
                        mlp_drop_rate=mlp_drop_rate,
                        mlp_ratio=mlp_ratio,
                        mlp_bias=mlp_bias,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                    )
                layer_modules.append(block)

            self.encoder_layers.append(nn.Sequential(*layer_modules))

        # The decoder mirrors the downsampling steps of stem and stages
        head_layer_channels = []
        head_layer_kernel_sizes = []
        for channel, kernel, stride in zip(
            [in_channels] + list(stem_channels) + list(layer_channels[:-1]),
            list(stem_kernel_sizes) + [max(msmc_kernel_sizes)] * len(layer_channels),
            list(stem_strides) + list(stage_aggr_ratios),
        ):
            if stride > 1:
                head_layer_channels.insert(0, channel)
                head_layer_kernel_sizes.insert(0, kernel)

        self.out_head = HeadDetectionPicking(
            feature_channels=layer_channels[-1],
            layer_channels=head_layer_channels,
            layer_kernel_sizes=head_layer_kernel_sizes,
            act_layer=act_layer,
            norm_layer=norm_layer,
            out_channels=out_channels,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(
            m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d)
        ):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out_size = x.shape[-1]

        x = self.stem(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.out_head(x, out_size)

        return x
