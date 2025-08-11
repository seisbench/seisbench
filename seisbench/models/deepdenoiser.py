from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import WaveformModel


class DeepDenoiser(WaveformModel):
    """
    .. document_args:: seisbench.models DeepDenoiser
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)

    def __init__(self, sampling_rate=100, **kwargs):
        citation = (
            "Zhu, W., Mousavi, S. M., & Beroza, G. C. (2019). "
            "Seismic signal denoising and decomposition using deep neural networks. "
            "IEEE Transactions on Geoscience and Remote Sensing, 57.11(2019), 9476 - 9488. "
            "https://doi.org/10.1109/TGRS.2019.2926772"
        )

        super().__init__(
            citation=citation,
            in_samples=3000,
            output_type="array",
            pred_sample=(0, 3000),
            labels=self.generate_label,
            sampling_rate=sampling_rate,
            grouping="channel",
            **kwargs,
        )

        self.inc = nn.Conv2d(2, 8, (3, 3), padding=(1, 1), bias=False)
        self.in_bn = nn.BatchNorm2d(8, eps=1e-3)

        self.down_conv_blocks = nn.ModuleList(
            [DownConvBlock(8 * 2 ** max(0, i - 1), 8 * 2**i) for i in range(5)]
        )

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-3)

        self.up_conv_blocks = nn.ModuleList(
            [UpConvBlock(8 * 2 ** (5 - i), 8 * 2 ** (4 - i)) for i in range(5)]
        )

        self.outc = nn.Conv2d(8, 2, (1, 1), bias=True)

    def forward(self, x):
        x = torch.relu(self.in_bn(self.inc(x)))

        mids = []
        for layer in self.down_conv_blocks:
            x, mid = layer(x)
            mids.append(mid)

        x = torch.relu(self.bn5(self.conv5(x)))

        for layer, mid in zip(self.up_conv_blocks, mids[::-1]):
            x = layer(x, mid)

        logits = self.outc(x)
        preds = torch.softmax(logits, dim=1)

        return preds

    @staticmethod
    def generate_label(stations):
        # Simply use channel as label
        return stations[0].split(".")[-1]

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        # Reproduces scipy.signal.stft(window, fs=self.sampling_rate, nperseg=30, nfft=60, boundary="zeros")
        # Note that the outputs need to be rotated by a factor of i ** dim to match numpy
        tmp_signal = (
            2
            / 30
            * torch.stft(
                batch,
                n_fft=60,
                return_complex=True,
                win_length=30,
                window=torch.hann_window(30).to(batch.device),
                pad_mode="constant",
                normalized=False,
            )
            * torch.pow(1j, torch.arange(31, device=batch.device)).unsqueeze(-1)
        )

        noisy_signal = torch.stack([tmp_signal.real, tmp_signal.imag], dim=1)

        noisy_signal[torch.isnan(noisy_signal)] = 0
        noisy_signal[torch.isinf(noisy_signal)] = 0

        return self._normalize_batch(noisy_signal), noisy_signal

    @staticmethod
    def _normalize_batch(data: torch.Tensor, window: int = 200) -> torch.Tensor:
        """
        Adapted from original DeepDenoiser implementation available at
        https://github.com/wayneweiqiang/DeepDenoiser/blob/7bd9284ece73e25c99db2ad101aacda2a215a41a/deepdenoiser/app.py#L72

        data shape: 2, nf, nt
        data: nbn, nf, nt, 2
        """
        data = data.permute(0, 2, 3, 1)  # nbn, nf, nt, 2
        assert len(data.shape) == 4
        shift = window // 2
        nbt, nf, nt, nimg = data.shape

        # std in slide windows
        data_pad = torch.nn.functional.pad(
            data, (0, 0, window // 2, window // 2), mode="reflect"
        )
        t = torch.arange(
            0, nt + shift - 1, shift, device=data.device
        )  # 201 => 0, 100, 200

        std = torch.zeros([nbt, len(t)], dtype=data.dtype, device=data.device)
        mean = torch.zeros([nbt, len(t)], dtype=data.dtype, device=data.device)
        for i in range(std.shape[1]):
            std[:, i] = torch.std(
                data_pad[:, :, i * shift : i * shift + window, :], axis=(1, 2, 3)
            )
            mean[:, i] = torch.mean(
                data_pad[:, :, i * shift : i * shift + window, :], axis=(1, 2, 3)
            )

        std[:, -1], mean[:, -1] = std[:, -2], mean[:, -2]
        std[:, 0], mean[:, 0] = std[:, 1], mean[:, 1]

        # normalize data with interpolated std
        interp_matrix = torch.zeros(3, 201, dtype=std.dtype, device=std.device)
        interp_matrix[0, :101] = 1 - torch.linspace(
            0, 1, 101, dtype=std.dtype, device=std.device
        )
        interp_matrix[1, :101] = torch.linspace(
            0, 1, 101, dtype=std.dtype, device=std.device
        )
        interp_matrix[1, 100:] = 1 - torch.linspace(
            0, 1, 101, dtype=std.dtype, device=std.device
        )
        interp_matrix[2, 100:] = torch.linspace(
            0, 1, 101, dtype=std.dtype, device=std.device
        )
        std_interp = std @ interp_matrix
        std_interp[std_interp == 0] = 1.0
        mean_interp = mean @ interp_matrix

        data = (data - mean_interp[:, np.newaxis, :, np.newaxis]) / std_interp[
            :, np.newaxis, :, np.newaxis
        ]

        if len(t) > 3:  # need to address this normalization issue in training
            data /= 2.0

        data = data.permute(0, 3, 1, 2)  # 1 (nbt), 2, nf, nt

        return data

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        signal = (piggyback[:, 0] + piggyback[:, 1] * 1j) * batch[:, 0]
        signal = signal / torch.pow(
            1j, torch.arange(signal.shape[-2], device=signal.device).unsqueeze(-1)
        )

        denoised_signal = 15 * torch.istft(
            signal,
            n_fft=60,
            win_length=30,
            window=torch.hann_window(30).to(batch.device),
            normalized=False,
        )

        return denoised_signal

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
            "grouping",
        ]:
            del model_args[key]

        model_args["sampling_rate"] = self.sampling_rate

        return model_args


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, (3, 3), padding=(1, 1), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            (3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        mid = x

        # Required for compatibility with tensorflow version - stride is treated differently otherwise
        remove2 = False
        remove3 = False
        if x.shape[2] % 2 == 0:
            x = F.pad(x, (0, 0, 1, 0), "constant", 0)
            remove2 = True
        if x.shape[3] % 2 == 0:
            x = F.pad(x, (1, 0), "constant", 0)
            remove3 = True

        x = self.conv2(x)

        if remove2:
            x = x[:, :, 1:]  # Remove padding
        if remove3:
            x = x[:, :, :, 1:]  # Remove padding

        x = torch.relu(self.bn2(x))

        return x, mid


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            in_channels, out_channels, (3, 3), stride=(2, 2), padding=(0, 0), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3)

        # Again in_channels to account for the added residual connections
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, (3, 3), padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x, mid):
        x = self.conv1(x)

        x = torch.relu(self.bn1(x))

        # Truncation is necessary to get correct shapes and be compatible with tensorflow implementation
        if mid.shape[2] % 2 == 0:
            x = x[:, :, :-1]
        else:
            x = x[:, :, :-2]

        if mid.shape[3] % 2 == 0:
            x = x[:, :, :, :-1]
        else:
            x = x[:, :, :, :-2]

        x = torch.cat([mid, x], dim=1)

        x = torch.relu(self.bn2(self.conv2(x)))

        return x
