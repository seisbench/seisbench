from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import istft, stft

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
    def __init__(
        self,
        in_channels_encoder: int,
        in_channels_decoder: int,
        inter_channels: int,
        bias: bool = False,
    ):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(
                in_channels_decoder,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(
                in_channels_encoder,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_encoder, g_decoder):
        g1 = self.W_g(g_decoder)
        x1 = self.W_x(x_encoder)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x_encoder * psi  # element-wise gating


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]] = 3,
        stride: Union[int, tuple[int, int]] = 1,
        drop_rate: float = 0.3,
        use_bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=use_bias,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
        )

    def forward(self, x):
        return self.conv(x)


class TransposeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]] = 3,
        stride: Union[int, tuple[int, int]] = 1,
        drop_rate: float = 0.3,
        use_bias: bool = False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                output_padding=1,
                bias=use_bias,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.block(x)


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
        output_activation=torch.nn.Softmax(dim=1),
        drop_rate: float = 0.0,
        use_bias: bool = False,
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

        self.in_samples = in_samples
        self.sampling_rate = sampling_rate
        self.in_channels = in_channels
        self.filters_root = filters_root
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = strides
        self.output_activation = output_activation
        self.drop_rate = drop_rate
        self.use_bias = use_bias
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

        # Allocate empty lists for all branches of Autoencoder
        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        # Encoder
        for i in range(depth):
            out_channels = 2**i * self.filters_root
            self.encoder_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    drop_rate=self.drop_rate,
                    use_bias=self.use_bias,
                )
            )
            if i < depth - 1:
                self.down_blocks.append(
                    ConvBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        drop_rate=self.drop_rate,
                        use_bias=self.use_bias,
                    )
                )
            in_channels = out_channels

        # Decoder
        for i in range(depth - 2, -1, -1):
            in_channels = 2 ** (i + 1) * self.filters_root
            out_channels = 2**i * self.filters_root
            self.up_blocks.append(
                TransposeConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    drop_rate=self.drop_rate,
                    use_bias=self.use_bias,
                )
            )
            self.attention_gates.append(
                AttentionGate(
                    in_channels_encoder=out_channels,
                    in_channels_decoder=out_channels,
                    inter_channels=out_channels // 2,
                )
            )
            self.decoder_blocks.append(
                ConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    drop_rate=self.drop_rate,
                    use_bias=self.use_bias,
                )
            )

        self.output_conv = nn.Conv2d(
            in_channels=self.filters_root, out_channels=self.in_channels, kernel_size=1
        )

    def forward(self, x):
        enc_features = []  # List to store encoder feature for skip connections

        # Encoder
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            enc_features.append(x)
            if i < self.depth - 1:
                x = self.down_blocks[i](x)

        # Decoder
        for i in range(self.depth - 2, -1, -1):
            x = self.up_blocks[self.depth - 2 - i](x)

            # Pad if needed to match encoder feature size
            diff_y = enc_features[i].size(2) - x.size(2)
            diff_x = enc_features[i].size(3) - x.size(3)
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

            if self.attention:  # Apply attention gate to encoder feature
                skip = self.attention_gates[self.depth - 2 - i](enc_features[i], x)
                x = x + skip  # gated skip connection
            else:  # No attention gate
                x = x + enc_features[i]  # Skip connection
            x = self.decoder_blocks[self.depth - 2 - i](x)

        x = self.output_conv(x)

        # Apply out activation function
        if self.output_activation:
            x = self.output_activation(x)

        return x

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

        # Normalize real and imaginary input to range [-1, 1]
        noisy_stft_real = noisy_stft.real / np.max(np.abs(noisy_stft.real))
        noisy_stft_imag = noisy_stft.imag / np.max(np.abs(noisy_stft.imag))

        noisy_input = torch.stack(
            tensors=[torch.Tensor(noisy_stft_real), torch.Tensor(noisy_stft_imag)],
            dim=1,
        )

        # Replace nans and infs
        noisy_input[torch.isnan(noisy_input)] = 0
        noisy_input[torch.isinf(noisy_input)] = 0

        return noisy_input, noisy_stft

    def _normalize_trace(self, batch: torch.Tensor):
        """
        Normalize each trace and save norm factors to scale back denoised traces.
        """
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
        Does postprocessing when predicting datasets
        """
        # Multiply piggyback (STFT of noisy signal) with batch (predicted mask) for signal
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
