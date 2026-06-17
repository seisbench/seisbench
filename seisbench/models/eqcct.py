"""
EQCCT P- and S-wave phase pickers as SeisBench :class:`~seisbench.models.base.WaveformModel` classes.

EQCCT uses separate models for P and S phase picking. Load the P-branch with
:py:class:`EQCCTP` and the S-branch with :py:class:`EQCCTS`, each via
:py:meth:`~seisbench.models.base.SeisBenchModel.from_pretrained`.
Pretrained weights are stored under ``<cache_model_root>/eqcct/`` and
``<cache_model_root>/eqccts/`` respectively.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import seisbench.util as sbu
from .base import WaveformModel

_EQCCT_CITATION = (
    "Saad, O. M., Chen, Y., Siervo, D., Zhang, F., Savvaidis, A., Huang, G.-c., "
    "Igonin, N., Fomel, S., & Chen, Y. (2023). "
    "EQCCT: A Production-Ready Earthquake Detection and Phase-Picking Method Using "
    "the Compact Convolutional Transformer. "
    "IEEE Transactions on Geoscience and Remote Sensing, 61, 1-15. "
    "https://doi.org/10.1109/TGRS.2023.3319440"
)

# Model parameters (matching TensorFlow / EQCCT reference)
stochastic_depth_rate = 0.1
image_size = 6000
patch_size = 40
num_patches = image_size // patch_size  # 150
projection_dim = 40
num_heads = 4
patch_dim = 40 * 1 * patch_size
transformer_layers = 4


class ConvF1Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=11, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, in_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(in_channels, eps=0.001)

        self.conv2 = nn.Conv1d(
            in_channels, in_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(in_channels, eps=0.001)

        self.conv3 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn3 = nn.BatchNorm1d(out_channels, eps=0.001)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        out = out + x

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.gelu(out)
        self.dropout(out)

        return out


class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        B = images.size(0)
        P = self.patch_size

        patches = images.unfold(1, P, P)

        patches = patches.permute(0, 1, 4, 2, 3).contiguous()

        patches = patches.view(B, patches.size(1), -1)
        return patches


class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim, patch_dim):
        super().__init__()
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, x):
        positions = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
        x = self.projection(x) + self.position_embedding(positions)
        return x


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return (x / keep_prob) * binary_tensor


class TransformerMLP(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_prob=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = KerasMHA(embed_dim=dim, num_heads=num_heads, key_dim=40)
        self.drop_path1 = StochasticDepth(drop_prob)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = TransformerMLP(dim, dropout_rate=0.1)
        self.drop_path2 = StochasticDepth(drop_prob)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        attn_out = self.attn(x)
        x = identity + self.drop_path1(attn_out)

        identity = x
        x = self.norm2(x)
        x = identity + self.drop_path2(self.mlp(x))
        return x


class OutputHead(nn.Module):
    def __init__(self, in_channels=1, kernel_size=15):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.activation(x).transpose(1, 2)


class KerasMHA(nn.Module):
    """
    Faithful re-implementation of tf.keras.layers.MultiHeadAttention
    with   key_dim = 40,  num_heads = 4,  embed_dim = 40.
    Internal hidden size = key_dim * num_heads = 160.
    """

    def __init__(self, embed_dim=40, num_heads=4, key_dim=40):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.inner_dim = num_heads * key_dim  # 160
        self.scale = 1.0 / np.sqrt(key_dim)

        self.q = nn.Linear(embed_dim, self.inner_dim, bias=True)
        self.k = nn.Linear(embed_dim, self.inner_dim, bias=True)
        self.v = nn.Linear(embed_dim, self.inner_dim, bias=True)
        self.o = nn.Linear(self.inner_dim, embed_dim, bias=True)

    # helper
    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.key_dim).transpose(1, 2)  # (B,H,T,D)

    def _merge(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).reshape(B, T, H * D)  # (B,T,160)

    def forward(self, x):
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        ctx = torch.matmul(weights, v)  # (B,H,T,D)
        ctx = self._merge(ctx)  # (B,T,160)
        return self.o(ctx)


class EQCCTModelP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvF1Block(3, 10)
        self.conv2 = ConvF1Block(10, 20)
        self.conv3 = ConvF1Block(20, 40)

        self.patch = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim, patch_dim)

        self.transformer = nn.Sequential(
            *[
                TransformerBlock(
                    projection_dim,
                    num_heads,
                    drop_prob=stochastic_depth_rate * (i / transformer_layers),
                )
                for i in range(transformer_layers)
            ]
        )
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head = OutputHead(in_channels=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = self.patch(x)
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.norm(x)

        x = x.reshape(x.size(0), 6000, 1)
        return self.head(x)


class EQCCTModelS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvF1Block(3, 10)
        self.conv2 = ConvF1Block(10, 20)
        self.conv3 = ConvF1Block(20, 40)

        self.patch = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, projection_dim, patch_dim)

        self.extra_pre = nn.ModuleList(
            [ConvF1Block(40, 40) for _ in range(transformer_layers)]
        )
        self.extra_post = nn.ModuleList(
            [ConvF1Block(40, 40) for _ in range(transformer_layers)]
        )

        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    projection_dim,
                    num_heads,
                    drop_prob=stochastic_depth_rate * (i / transformer_layers),
                )
                for i in range(transformer_layers)
            ]
        )

        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head = OutputHead(in_channels=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = self.patch(x)
        x = self.encoder(x)

        for i in range(transformer_layers):
            x_pre_conv = x.transpose(1, 2)
            x_pre_conv = self.extra_pre[i](x_pre_conv).transpose(1, 2)
            x = x_pre_conv

            identity = x
            x_norm1 = self.transformers[i].norm1(x)
            attention_output = self.transformers[i].attn(x_norm1)
            attention_output_post_conv = attention_output.transpose(1, 2)
            attention_output_post_conv = self.extra_post[i](
                attention_output_post_conv
            ).transpose(1, 2)
            x = identity + self.transformers[i].drop_path1(attention_output_post_conv)

            identity2 = x
            x_norm2 = self.transformers[i].norm2(x)
            x_mlp = self.transformers[i].mlp(x_norm2)
            x = identity2 + self.transformers[i].drop_path2(x_mlp)

        x = self.norm(x)
        x = x.reshape(x.size(0), 6000, 1)
        return self.head(x)


class _EQCCTBranchWaveform(WaveformModel):
    """
    Shared windowing, preprocessing, and I/O for EQCCT P/S branches.

    Expects 6000-sample (60 s at 100 Hz) three-component windows. Subclasses supply
    the PyTorch backbone and phase label list (:class:`EQCCTP` for P, :class:`EQCCTS`
    for S).
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (500, 500),
    )
    _annotate_args["S_threshold"] = (
        "Detection threshold for S-phase probability curve",
        0.5,
    )
    _annotate_args["stacking"] = (
        "Stacking method for overlapping windows (only for window prediction models). "
        "Options are 'max' and 'avg'. ",
        "max",
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 0.5)

    def __init__(
        self,
        backbone: nn.Module,
        labels: list[str],
        citation: str,
        sampling_rate: float = 100,
        norm: str = "std",
        norm_amp_per_comp: bool = False,
        norm_detrend: bool = False,
        **kwargs,
    ):
        super().__init__(
            citation=citation,
            output_type="array",
            in_samples=6000,
            pred_sample=(0, 6000),
            labels=labels,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.norm = norm
        self.norm_amp_per_comp = norm_amp_per_comp
        self.norm_detrend = norm_detrend
        self._eqcct_backbone = backbone

    def forward(self, x):
        """
        Run the EQCCT backbone on a SeisBench-formatted batch.

        :param x: Tensor of shape ``(batch, 3, in_samples)``.
        :return: Phase probability curves of shape ``(batch, n_labels, in_samples)``.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, 3, T), got shape {tuple(x.shape)}")
        if x.shape[2] != self.in_samples:
            raise ValueError(
                f"Expected last dim in_samples={self.in_samples}, got {x.shape[2]}"
            )
        if x.shape[1] == 3:
            x = x.transpose(1, 2)
        y = self._eqcct_backbone(x)
        return y.transpose(1, 2)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        EQCCT preprocessing applied to each window before inference.

        Removes the per-window mean, optionally detrends and peak/std-normalizes the
        waveforms, then applies a short cosine taper to the first and last six samples
        of each trace.
        """
        batch = batch - batch.mean(axis=-1, keepdims=True)
        if self.norm_detrend:
            batch = sbu.torch_detrend(batch)
        if self.norm_amp_per_comp:
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)
        else:
            if self.norm == "std":
                std = batch.std(axis=(-1, -2), keepdims=True)
                batch = batch / (std + 1e-10)
            elif self.norm == "peak":
                peak = batch.abs().max(axis=-1, keepdims=True)[0]
                batch = batch / (peak + 1e-10)

        tap = 0.5 * (
            1
            + torch.cos(
                torch.linspace(
                    np.pi,
                    2 * np.pi,
                    6,
                    device=batch.device,
                    dtype=batch.dtype,
                )
            )
        )
        batch[:, :, :6] *= tap
        batch[:, :, -6:] *= tap.flip(dims=(0,))

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Transpose model outputs to SeisBench channel layout and blind edge samples.

        Samples at the beginning and end of each prediction (see ``blinding`` in
        :py:func:`annotate` / :py:func:`classify`) are set to NaN before stacking
        overlapping windows.
        """
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
        Convert stacked probability annotations into discrete phase picks.

        Uses the phase-specific threshold from ``argdict`` (for example
        ``P_threshold`` or ``S_threshold``).
        """
        picks = sbu.PickList()
        for phase in self.labels:
            if phase == "N":
                continue
            threshold_key = f"{phase}_threshold"
            default_threshold = self._annotate_args.get(threshold_key, (None, 0.3))[1]
            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(threshold_key, default_threshold),
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
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["norm_amp_per_comp"] = self.norm_amp_per_comp
        model_args["norm_detrend"] = self.norm_detrend
        return model_args


class EQCCTP(_EQCCTBranchWaveform):
    """
    The EQCCT P-wave phase picker from Saad et al. (2023).

    EQCCT uses separate compact convolutional-transformer models for P and S picking.
    This class wraps the P-branch architecture for SeisBench :py:func:`annotate` and
    :py:func:`classify` on 6000-sample (60 s) three-component windows at 100 Hz.

    By instantiating the model with ``from_pretrained("original")``, the PyTorch weights
    converted from the institutional TensorFlow EQCCT P-branch checkpoint can be loaded.

    .. document_args:: seisbench.models EQCCTP

    :param sampling_rate: Target sampling rate in Hz, by default 100.
                          Incoming traces are resampled automatically when this differs.
    :param norm: Data normalization strategy, either ``"peak"`` or ``"std"``, by default
                 ``"std"``.
    :param norm_amp_per_comp: If True, normalize each component independently by its peak
                              amplitude. Defaults to False.
    :param norm_detrend: If True, apply linear detrending before normalization.
                         Defaults to False.
    :param kwargs: Keyword arguments passed to the constructor of
                   :py:class:`~seisbench.models.base.WaveformModel`.
    """

    def __init__(
        self,
        sampling_rate=100,
        norm="std",
        norm_amp_per_comp=False,
        norm_detrend=False,
        **kwargs,
    ):
        super().__init__(
            EQCCTModelP(),
            ["P"],
            _EQCCT_CITATION,
            sampling_rate=sampling_rate,
            norm=norm,
            norm_amp_per_comp=norm_amp_per_comp,
            norm_detrend=norm_detrend,
            **kwargs,
        )


class EQCCTS(_EQCCTBranchWaveform):
    """
    The EQCCT S-wave phase picker from Saad et al. (2023).

    EQCCT uses separate compact convolutional-transformer models for P and S picking.
    This class wraps the deeper S-branch architecture (with additional convolutional
    stems around each transformer block) for SeisBench :py:func:`annotate` and
    :py:func:`classify` on 6000-sample (60 s) three-component windows at 100 Hz.

    Use a **separate** S-branch checkpoint; do not load P-branch weights into this model.

    By instantiating the model with ``from_pretrained("original")``, the PyTorch weights
    converted from the institutional TensorFlow EQCCT S-branch checkpoint can be loaded.

    .. document_args:: seisbench.models EQCCTS

    :param sampling_rate: Target sampling rate in Hz, by default 100.
                          Incoming traces are resampled automatically when this differs.
    :param norm: Data normalization strategy, either ``"peak"`` or ``"std"``, by default
                 ``"std"``.
    :param norm_amp_per_comp: If True, normalize each component independently by its peak
                              amplitude. Defaults to False.
    :param norm_detrend: If True, apply linear detrending before normalization.
                         Defaults to False.
    :param kwargs: Keyword arguments passed to the constructor of
                   :py:class:`~seisbench.models.base.WaveformModel`.
    """

    def __init__(
        self,
        sampling_rate=100,
        norm="std",
        norm_amp_per_comp=False,
        norm_detrend=False,
        **kwargs,
    ):
        super().__init__(
            EQCCTModelS(),
            ["S"],
            _EQCCT_CITATION,
            sampling_rate=sampling_rate,
            norm=norm,
            norm_amp_per_comp=norm_amp_per_comp,
            norm_detrend=norm_detrend,
            **kwargs,
        )
