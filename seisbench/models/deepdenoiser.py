from .base import WaveformModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d


class DeepDenoiser(WaveformModel):
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
            default_args={"overlap": 1000},
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
    def generate_label(trace_stats):
        # Simply use channel as label
        return trace_stats.channel

    def annotate_window_pre(self, window, argdict):
        f, t, tmp_signal = scipy.signal.stft(
            window, fs=self.sampling_rate, nperseg=30, nfft=60, boundary="zeros"
        )
        noisy_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=0)

        noisy_signal[np.isnan(noisy_signal)] = 0
        noisy_signal[np.isinf(noisy_signal)] = 0

        return self._normalize_batch(noisy_signal), noisy_signal

    @staticmethod
    def _normalize_batch(data, window=200):
        """
        Adapted from original DeepDenoiser implementation available at
        https://github.com/wayneweiqiang/DeepDenoiser/blob/7bd9284ece73e25c99db2ad101aacda2a215a41a/deepdenoiser/app.py#L72

        data shape: 2, nf, nt
        data: nbn, nf, nt, 2
        """
        data = np.expand_dims(data, 0)  # 1 (nbt), 2, nf, nt
        data = data.transpose(0, 2, 3, 1)  # nbn, nf, nt, 2
        assert len(data.shape) == 4
        shift = window // 2
        nbt, nf, nt, nimg = data.shape

        # std in slide windows
        data_pad = np.pad(
            data, ((0, 0), (0, 0), (window // 2, window // 2), (0, 0)), mode="reflect"
        )
        t = np.arange(0, nt + shift - 1, shift, dtype="int")  # 201 => 0, 100, 200

        std = np.zeros([nbt, len(t)])
        mean = np.zeros([nbt, len(t)])
        for i in range(std.shape[1]):
            std[:, i] = np.std(
                data_pad[:, :, i * shift : i * shift + window, :], axis=(1, 2, 3)
            )
            mean[:, i] = np.mean(
                data_pad[:, :, i * shift : i * shift + window, :], axis=(1, 2, 3)
            )

        std[:, -1], mean[:, -1] = std[:, -2], mean[:, -2]
        std[:, 0], mean[:, 0] = std[:, 1], mean[:, 1]

        # normalize data with interplated std
        t_interp = np.arange(nt, dtype="int")
        std_interp = interp1d(t, std, kind="slinear")(t_interp)
        std_interp[std_interp == 0] = 1.0
        mean_interp = interp1d(t, mean, kind="slinear")(t_interp)

        data = (data - mean_interp[:, np.newaxis, :, np.newaxis]) / std_interp[
            :, np.newaxis, :, np.newaxis
        ]

        if len(t) > 3:  # need to address this normalization issue in training
            data /= 2.0

        data = data.transpose(0, 3, 1, 2)  # 1 (nbt), 2, nf, nt
        data = data[0]  # Remove batch dim

        return data

    def annotate_window_post(self, pred, piggyback=None, argdict=None):
        noisy_signal = piggyback
        _, denoised_signal = scipy.signal.istft(
            (noisy_signal[0] + noisy_signal[1] * 1j) * pred[0],
            fs=self.sampling_rate,
            nperseg=30,
            nfft=60,
            boundary="zeros",
        )
        return denoised_signal.reshape(-1, 1)

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
