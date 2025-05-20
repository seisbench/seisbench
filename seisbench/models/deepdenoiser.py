from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import istft, stft

from ..util.torch_helpers import (
    min_max_normalization,
    output_shape_conv2d_layers,
    padding_transpose_conv2d_layers,
)
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


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_x = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.W_g = nn.Conv2d(
            in_channels=gating_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.psi = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        psi = self.relu(x1 + g1)
        psi = self.sigmoid(self.psi(psi))

        return x * psi


class SeisDAE(DeepDenoiser):
    """ """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (0, 0),
    )

    def __init__(
        self,
        in_samples: int = 3000,
        in_channels: int = 2,
        sampling_rate: float = 100,
        filters_root: int = 8,
        depth: int = 6,
        kernel_size: tuple[int, int] = (3, 3),
        strides: tuple[int, int] = (2, 2),
        skip_connections: bool = True,
        activation=torch.relu,
        output_activation=torch.nn.Softmax(dim=1),
        norm: str = "peak",
        scale: tuple[float, float] = (0, 1),
        nfft: int = 60,
        nperseg: int = 30,
        attention: bool = False,
        **kwargs,
    ):

        citation = (
            "Zhu, W., Mousavi, S. M., & Beroza, G. C. (2019). "
            "Seismic signal denoising and decomposition using deep neural networks. "
            "IEEE Transactions on Geoscience and Remote Sensing, 57.11(2019), 9476 - 9488. "
            "https://doi.org/10.1109/TGRS.2019.2926772"
            " "
            "Heuel, J., & Friederich, W. (2022). "
            "Suppression of wind turbine noise from seismological data using nonlinear thresholding "
            "and denoising autoencoder. "
            "Journal of Seismology, 26(5), 913-934. "
            "https://doi.org/10.1007/s10950-022-10097-6"
        )

        WaveformModel.__init__(
            self,
            citation=citation,
            in_samples=in_samples,
            output_type="array",
            default_args={"overlap": in_samples // 2},
            labels=self.generate_label,
            pred_sample=(0, in_samples),
            sampling_rate=sampling_rate,
            grouping="channel",
            **kwargs,
        )

        self.in_samples = in_samples  # XXX not necessary
        self.sampling_rate = sampling_rate  # XXX not necessary
        self.in_channels = in_channels
        self.filters_root = filters_root
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = strides
        self.skip_connections = skip_connections
        self.activation = activation
        self.output_activation = output_activation
        self.norm = norm
        self.scale = scale
        self.norm_factors = None
        self.attention = attention

        # Determine input shape from STFT and check if STFT and ISTFT work
        self.nfft = nfft
        self.nperseg = nperseg
        # Performing STFT
        _, _, dummystft = stft(
            x=np.random.rand(self.in_samples),
            fs=self.sampling_rate,
            nfft=self.nfft,
            nperseg=self.nperseg,
        )
        # Performing ISTFT
        t, dummy_x = istft(
            Zxx=dummystft,
            fs=self.sampling_rate,
            nfft=self.nfft,
            nperseg=self.nperseg,
        )

        if len(dummy_x) != self.in_samples:
            msg = (
                f"If data with length {self.in_samples} are transformed with STFT and back transformed with "
                f"ISTFT, the output lenght of ISTFT ({len(dummy_x)}) does not match. Choose different values "
                f"for nfft={self.nfft} and nperseg={self.nperseg}."
            )
            raise ValueError(msg)
        self.input_shape = dummystft.shape

        # Write STFT values to default args dictionary
        self.default_args["nfft"] = self.nfft
        self.default_args["npserg"] = self.nperseg

        # Initial layer
        self.inc = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.filters_root,
            kernel_size=self.kernel_size,
            padding="same",
        )
        self.in_bn = nn.BatchNorm2d(num_features=self.filters_root, eps=1e-3)

        # Set up ModuleLists for encoder, decoder and attention gates
        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        # Down branch (Contracting path / Encoder)
        last_filters = self.filters_root
        down_shapes = {-1: self.input_shape}
        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv2d(
                in_channels=last_filters,
                out_channels=filters,
                kernel_size=self.kernel_size,
                padding="same",
                bias=False,
            )
            last_filters = filters
            bn1 = nn.BatchNorm2d(num_features=filters, eps=1e-3)

            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                down_shapes.update(
                    {
                        i: output_shape_conv2d_layers(
                            input_shape=down_shapes[i - 1],
                            padding=(1, 1),
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                        )
                    }
                )
                conv_down = nn.Conv2d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=(1, 1),
                    bias=False,
                )
                bn2 = nn.BatchNorm2d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        # Up branch (Expansive path / Decoder)
        for i in range(self.depth - 2, -1, -1):
            filters = int(2**i * self.filters_root)

            padding = padding_transpose_conv2d_layers(
                input_shape=down_shapes[i],
                output_shape=down_shapes[i - 1],
                kernel_size=self.kernel_size,
                stride=self.stride,
            )

            conv_up = nn.ConvTranspose2d(
                in_channels=last_filters,
                out_channels=filters,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=padding,
                bias=False,
            )
            last_filters = filters
            bn1 = nn.BatchNorm2d(filters, eps=1e-3)

            if self.skip_connections:
                in_channels_conv_same = int(2 ** (i + 1) * self.filters_root)
            else:
                in_channels_conv_same = filters

            conv_same = nn.Conv2d(
                in_channels=in_channels_conv_same,
                out_channels=filters,
                kernel_size=self.kernel_size,
                padding="same",
                bias=False,
            )
            bn2 = nn.BatchNorm2d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

            # Setting up attention layers (also if self.attention is False)
            # During the forward pass the attention layer is only applied if self.attention is True
            self.attention_gates.append(
                AttentionGate(
                    in_channels=filters,
                    gating_channels=last_filters,
                    inter_channels=filters // 2,
                )
            )

        # Final layer
        self.out = nn.Conv2d(
            in_channels=last_filters,
            out_channels=self.in_channels,
            kernel_size=(1, 1),
            padding="same",
        )

    def forward(self, x):
        x = self.activation(self.in_bn(self.inc(x)))

        skip_connections = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))
            if conv_down is not None:
                skip_connections.append(x)
                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), att_gate, skip) in enumerate(
            zip(self.up_branch, self.attention_gates, skip_connections[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))

            # Crop x to correct size as given from down branch
            x = self.crop(x, skip)

            # Add skip connections and attention gates, if self.attention is True
            if self.skip_connections:
                if self.attention:
                    skip = att_gate(skip, x)
                x = torch.cat([skip, x], dim=1)  # Skip connections (concatenation)

            x = self.activation(bn2(conv_same(x)))

        # Apply out activation function
        if self.output_activation:
            x = self.output_activation(self.out(x))
        else:
            x = self.out(x)

        return x

    @staticmethod
    def crop(tensor, target_tensor):
        """Crop the tensor to match the target tensor size."""
        _, _, h, w = target_tensor.size()
        return tensor[:, :, :h, :w]

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Does preprocessing for prediction
        """
        # STFT of each batch and component
        self.norm_factors = np.empty(shape=batch.shape[0])
        _, _, noisy_stft = stft(
            x=self._normalize_trace(batch=batch),
            fs=self.sampling_rate,
            nfft=self.nfft,
            nperseg=self.nperseg,
        )

        # Min-max normalize STFT
        noisy_stft_real = min_max_normalization(x=noisy_stft.real)
        noisy_stft_imag = min_max_normalization(x=noisy_stft.imag)

        noisy_input = torch.stack(
            tensors=[torch.Tensor(noisy_stft_real), torch.Tensor(noisy_stft_imag)],
            dim=1,
        )

        # Replace nans and infs
        noisy_input[torch.isnan(noisy_input)] = 0
        noisy_input[torch.isinf(noisy_input)] = 0

        return noisy_input, noisy_stft

    def _normalize_trace(self, batch: torch.Tensor):
        for trace_id in range(batch.shape[0]):
            batch[trace_id, :] -= torch.mean(input=batch[trace_id, :])  # Demean trace
            if self.norm == "peak":
                norm = torch.max(torch.abs(batch[trace_id, :]))
            elif self.norm == "std":
                norm = torch.std(batch[trace_id, :])
            batch[trace_id, :] /= norm

            # Save normalization factors for conservation of amplitude
            self.norm_factors[trace_id] = norm

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Does postprocessing when predicted datasets
        """
        # Multiply piggyback (noisy signal) with batch (predicted mask)
        signal_stft = piggyback * batch.numpy()[:, 0, :]

        _, denoised_signal = istft(
            Zxx=signal_stft,
            fs=self.sampling_rate,
            nfft=self.nfft,
            nperseg=self.nperseg,
        )

        # Apply blinding
        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            denoised_signal[:, :prenan] = np.nan
        if postnan > 0:
            denoised_signal[:, -postnan:] = np.nan

        return (
            denoised_signal * self.norm_factors[:, None]
        )  # Convert denoised to original amplitude

    def get_model_args(self):
        model_args = super().get_model_args()
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["in_samples"] = self.in_samples
        model_args["nfft"] = self.nfft
        model_args["nperseg"] = self.nperseg

        return model_args
