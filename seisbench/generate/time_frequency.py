"""
Function gets signal and noise datasets, adds noise and signal as done in
sbg.RealNoise and computes stft of noisy, signal and noise and
returns all to create masks/labels for denoiser
"""

import copy
import random

import numpy as np
from scipy.signal import stft

import seisbench
from seisbench.data.base import WaveformDataset


class DenoiserPreprocessing:
    def __init__(
        self,
        # earthquake_dataset: WaveformDataset,
        noise_dataset: WaveformDataset,
        scale: tuple[float, float] = (0, 1),
        scaling_type: str = "peak",
        component: str = "ZNE",
        key: tuple[str, str] = ("X", "y"),
        nfft: int = 60,
        nperseg: int = 30,
        **kwargs,
    ):
        # TODO: noise should be optional for data sets with noise and signal in one data set

        # self.earthquake_dataset = earthquake_dataset
        self.noise_dataset = noise_dataset
        self.scale = scale
        self.scaling_type = scaling_type.lower()
        self.key = key
        self.component = component
        self.noise_generator = seisbench.generate.GenericGenerator(noise_dataset)
        self.noise_samples = len(noise_dataset)
        self.nfft = nfft
        self.nperseg = nperseg
        self.kwargs = kwargs

        if scaling_type.lower() not in ["std", "peak"]:
            msg = "Argument scaling_type must be either 'std' or 'peak'."
            raise ValueError(msg)

    def __call__(self, state_dict):
        """ """
        x, metadata = state_dict[self.key[0]]

        if self.key[0] != self.key[1]:
            # Ensure data and metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)
            x = x.copy()

        # Select component if only one is given. If more than one component is given, choose random component
        # Note, if more than one component is given, components of signal and noise are not mixed
        if len(self.component) == 1:
            component_idx = metadata["trace_component_order"].index(self.component)
        else:
            component_idx = random.randint(0, len(self.component) - 1)

        # Select component from x and modify metadata
        x = x[component_idx, :]
        metadata["trace_component_order"] = self.component[component_idx]

        # Defining scale for noise amplitude
        scale = np.random.uniform(*self.scale)

        if self.scaling_type == "peak":
            scale = scale * np.max(np.abs(x - np.mean(x, axis=-1, keepdims=True)))
        elif self.scaling_type == "std":
            scale = scale * np.std(x)

        # Draw random noise sample from the dataset and cut to same length as x
        n = self.noise_generator[np.random.randint(low=0, high=self.noise_samples)]["X"]

        # Select noise component
        n = n[component_idx, :]

        # Removing mean from noise samples
        n -= np.mean(n, axis=-1, keepdims=True)

        # Normalize noise samples
        if self.scaling_type == "peak":
            n = n / np.max(np.abs(n))
        elif self.scaling_type == "std":
            n = n / np.std(n)
        n = n * scale

        # Cutting noise to same length as x
        if len(n) - len(x) < 0:
            msg = (
                f"The length of the data ({len(x)}) and the noise ({len(n)}) must either be the same or the "
                f"shape of the noise must be larger than the shape of the data."
            )
            raise ValueError(msg)

        if len(n) - len(x) > 0:
            spoint = np.random.randint(low=0, high=len(n) - len(x))
        else:
            spoint = 0
        n = n[spoint : spoint + len(x)]

        # Add noise and signal to create noisy signal
        noisy = x + n

        # Normalize noisy, x and n by normalization factor of noisy
        if self.scaling_type == "peak":
            norm_noisy = np.max(np.abs(noisy))
        elif self.scaling_type == "std":
            norm_noisy = np.std(noisy)

        x = x / norm_noisy
        n = n / norm_noisy
        noisy = noisy / norm_noisy

        # DO STFT of x, n, noisy
        _, _, stft_x = stft(
            x=x,
            fs=metadata["trace_sampling_rate_hz"],
            nperseg=self.nperseg,
            nfft=self.nfft,
            **self.kwargs,
        )
        _, _, stft_n = stft(
            x=n,
            fs=metadata["trace_sampling_rate_hz"],
            nperseg=self.nperseg,
            nfft=self.nfft,
            **self.kwargs,
        )
        _, _, stft_noisy = stft(
            x=noisy,
            fs=metadata["trace_sampling_rate_hz"],
            nperseg=self.nperseg,
            nfft=self.nfft,
            **self.kwargs,
        )

        # Determine output masks
        # TODO: remove ignoring of runtime warnings
        np.seterr(
            divide="ignore", invalid="ignore"
        )  # Ignoring Runtime Warnings for division by zero
        X = np.empty(
            shape=(2, *stft_n.shape)
        )  # Input, i.e. real and imag part of noisy
        y = np.empty(
            shape=(2, *stft_n.shape)
        )  # Target, i.e. masks for signal and noise
        X[0, :, :] = stft_noisy.real / np.max(np.abs(stft_noisy.real))
        X[1, :, :] = stft_noisy.imag / np.max(np.abs(stft_noisy.imag))
        y[0, :, :] = 1 / (1 + np.abs(stft_n) / np.abs(stft_x))
        y[1, :, :] = 1 - y[0, :, :]

        # Replace nan values in X and y
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

        # Update state_dicts for input and target
        # state_dict[self.key[1]] = (x, metadata)
        state_dict[self.key[0]] = (X, metadata)
        state_dict[self.key[1]] = (y, metadata)

    def __str__(self):
        return "STFT for Denoising labeller"
