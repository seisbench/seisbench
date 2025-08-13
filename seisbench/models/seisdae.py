from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import istft, stft

from .base import WaveformModel


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net architectures.

    This module implements an attention mechanism that selectively emphasizes relevant features
    in encoder outputs before concatenation with decoder features. It is based on the additive
    attention gating mechanism from *Attention U-Net: Learning Where to Look for the Pancreas*
    (Oktay et al., 2018).

        .. admonition:: Citation
          Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa,
          Kensaku Morim, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, Ben Glocker, Daniel Rueckert (2018)
          Attention U-Net: Learning Where to Look for the Pancreas

          https://arxiv.org/abs/1804.03999

    :param in_channels_encoder: Number of input channels from the encoder (skip connection).
    :param in_channels_decoder: Number of input channels from the decoder (gating signal).
    :param inter_channels: Number of intermediate channels used in attention computations.
    :param bias: If True, adds a learnable bias to the convolution layers. Default is False.
    """

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


class SeisDAE(WaveformModel):
    """
    Seismic Denoising Autoencoder using U-Net Architecture with additional attention gates.
    A configurable denoising autoencoder for seismic waveform data that operates in the
    time-frequency domain using the Short-Time Fourier Transform (STFT). The model is based
    on a U-Net structure with optional attention gates and skip connections.

    :param in_samples: Length of the input waveform in samples. Default is 3000 samples.
    :param in_channels: Number of input channels (e.g., 2 for real and imaginary STFT components).
                        Default are 2 channels.
    :param sampling_rate: Sampling rate of the waveform data in Hz.
                          Default sampling rate is 100 Hz.
    :param filters_root: Number of filters in the first convolutional layer (doubles with depth).
                         Default is 8.
    :param depth: Number of encoding/decoding levels in the U-Net.
                  Default is 6 for STFT
    :param kernel_size: Kernel size for convolutional layers.
                        Default is (3, 3).
    :param strides: Stride size used for down/upsampling using Conv2D and transpose Conv2D layers.
                    Default is (2, 2).
    :param output_activation: Activation function applied to final output
                              Default is Softmax.
    :param drop_rate: Dropout rate used throughout the network.
                      Default drop_rate is 0
    :param use_bias: Whether to use bias in convolutional layers.
                     Default is False.
    :param norm: Type of normalization applied to traces ("peak" or "std").
                 Default is "peak"
    :param eps: Factor to avoid division by zero. Default value is 1e-13.
    :param nfft: Length of the FFT used, if a zero padded FFT is desired for scipy STFT.
                 If None, the FFT length is nperseg.
    :param nperseg: Length of each segment for scipy STFT. Default is 60
    :param attention: Whether to use attention gates in skip connections or not.
                      Default is False
    :param kwargs: Additional arguments passed to the base `WaveformModel`.
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 0.5)
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
        eps: float = 1e-13,
        nfft: int = 60,
        nperseg: int = 30,
        attention: bool = False,
        **kwargs,
    ):
        citation = (
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
            pred_sample=(0, in_samples),
            sampling_rate=sampling_rate,
            labels=self.generate_label,
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
        self.eps = eps
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
        self.default_args["nperseg"] = self.nperseg

        # Allocate empty lists for all branches of Autoencoder
        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        # Encoder
        cur_channels = in_channels
        for i in range(depth):
            out_channels = 2**i * self.filters_root
            self.encoder_blocks.append(
                ConvBlock(
                    in_channels=cur_channels,
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
            cur_channels = out_channels

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
            if self.attention:
                self.attention_gates.append(
                    AttentionGate(
                        in_channels_encoder=out_channels,
                        in_channels_decoder=out_channels,
                        inter_channels=out_channels // 2,
                    )
                )
            else:
                self.attention_gates.append(None)

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

    @staticmethod
    def generate_label(stations):
        # Simply use channel as label
        return stations[0].split(".")[-1]

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Does preprocessing for prediction
        """
        # STFT of each batch and component
        batch, norm_factors = self._normalize_trace(batch=batch)
        noisy_stft = torch.stft(
            input=batch,
            n_fft=self.nfft,
            win_length=self.nperseg,
            window=torch.hann_window(self.nperseg).to(batch.device),
            hop_length=self.nperseg // 2,  # for 50% overlap like SciPy default
            pad_mode="constant",
            return_complex=True,
            normalized=False,
        )

        # Normalize real and imaginary input to range [-1, 1]
        # Normalizing real and imaginary part might distort amplitude and phase, however, from my experiences the
        # denoising result is more accurate. If you train a SeisDAE model without normalization, don't forget to
        # remove the normalization in the training (labeling.STFTDenoiserLabeller)
        noisy_stft_real = noisy_stft.real / torch.max(torch.abs(noisy_stft.real))
        noisy_stft_imag = noisy_stft.imag / torch.max(torch.abs(noisy_stft.imag))

        noisy_input = torch.stack(
            tensors=[torch.Tensor(noisy_stft_real), torch.Tensor(noisy_stft_imag)],
            dim=1,
        )

        # Replace nans and infs
        noisy_input[torch.isnan(noisy_input)] = 0
        noisy_input[torch.isinf(noisy_input)] = 0

        return noisy_input, (noisy_stft, norm_factors)

    def _normalize_trace(self, batch: torch.Tensor):
        """
        Normalize each trace and save norm factors to scale back denoised traces.
        """
        # Demean each trace
        batch = batch - batch.mean(dim=1, keepdim=True)

        # Compute norm factor
        if self.norm == "peak":
            norm = batch.abs().max(dim=1, keepdim=True).values
        elif self.norm == "std":
            norm = batch.std(dim=1, keepdim=True)

        # Normalize (and avoid division by zero)
        batch = batch / (norm + self.eps)

        # Save normalization factors
        norm_factors = norm.squeeze(1)

        return batch, norm_factors

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Does postprocessing when predicting datasets
        """
        # Multiply piggyback[0] (STFT of noisy signal) with batch (predicted mask) for signal
        signal_stft = piggyback[0] * batch[:, 0, :]

        denoised_signal = torch.istft(
            input=signal_stft,
            n_fft=self.nfft,
            win_length=self.nperseg,
            hop_length=self.nperseg // 2,  # match overlap used in stft
            window=torch.hann_window(self.nperseg).to(batch.device),
            normalized=False,
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
            denoised_signal * piggyback[1][:, None]
        )  # Convert denoised to original amplitude

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
            "grouping",
        ]:
            del model_args[key]

        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["in_samples"] = self.in_samples
        model_args["nfft"] = self.nfft
        model_args["nperseg"] = self.nperseg

        return model_args
