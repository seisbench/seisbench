from .base import WaveformModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import stft
import numpy as np


class CRED(WaveformModel):
    """
    Note: There are subtle differences between the model presented in the paper (as in Figure 1) and the code on Github.
          Here we follow the implementation from Github to allow for compatibility with the pretrained weights.
    """

    def __init__(
        self,
        in_samples=3000,
        in_channels=3,
        sampling_rate=100,
        original_compatible=False,
        **kwargs,
    ):
        citation = (
            "Mousavi, S. M., Zhu, W., Sheng, Y., & Beroza, G. C. (2019). "
            "CRED: A deep residual network of convolutional and recurrent units for earthquake signal detection. "
            "Scientific reports, 9(1), 1-14. "
            "https://doi.org/10.1038/s41598-019-45748-1"
        )
        super().__init__(
            citation=citation,
            labels=["Detection"],
            sampling_rate=sampling_rate,
            in_samples=in_samples,
            output_type="array",
            pred_sample=(0, in_samples),
            default_args={"overlap": 500},
            **kwargs,
        )

        self.in_channels = in_channels
        self.original_compatible = original_compatible

        self.conv1 = nn.Conv2d(in_channels, 8, 9, padding=4, stride=2)
        self.cnn_block1 = BlockCNN(8, 8, 7)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2, stride=2)
        self.cnn_block2 = BlockCNN(16, 16, 3)
        self.lstm_block = BlockBiLSTM(176, 64)
        self.lstm = nn.LSTM(128, 64, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.norm1 = nn.BatchNorm1d(64, eps=1e-3)
        self.fc1 = nn.Linear(64, 64)
        self.norm2 = nn.BatchNorm1d(64, eps=1e-3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.cnn_block1(x) + x

        if self.original_compatible:
            # Required for compatibility with tensorflow version
            x = F.pad(x, (0, 0, 1, 0), "constant", 0)
            x = torch.relu(self.conv2(x))
            x = x[:, :, 1:]  # Remove padding
        else:
            x = torch.relu(self.conv2(x))

        x = self.cnn_block2(x) + x

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[:2] + (-1,))

        x = self.lstm_block(x)
        x = self.lstm(x)[0]
        x = self.dropout(x)

        # BatchNorm needs permutation
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = x.permute(0, 2, 1)

        shape_save = x.shape
        x = x.reshape((-1,) + x.shape[2:])
        x = torch.relu(self.fc1(x))
        x = x.reshape(shape_save)

        # BatchNorm needs permutation
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = x.permute(0, 2, 1)

        shape_save = x.shape
        x = x.reshape((-1,) + x.shape[2:])
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        x = x.reshape(shape_save[:2] + (1,))

        return x

    @staticmethod
    def waveforms_to_spectrogram(wv):
        """
        Transforms waveforms into spectrogram using short term fourier transform
        :param wv: Waveforms with shape (channels, samples)
        :return: Spectrogram with shape (channels, times, frequencies)
        """
        _, _, z = stft(wv, fs=100, nperseg=80)
        return np.transpose(np.abs(z), (0, 2, 1))

    def annotate_window_pre(self, window, argdict):
        # Add a demean and an amplitude normalization step to the preprocessing
        window = window - np.mean(window, axis=-1, keepdims=True)
        window = window / (np.std(window) + 1e-10)
        return self.waveforms_to_spectrogram(window)

    def classify_aggregate(self, annotations, argdict):
        """
        Converts the annotations to discrete detections using
        :py:func:`~seisbench.models.base.WaveformModel.detections_from_annotations`.
        Trigger onset thresholds are derived from the argdict at key "detection_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of detections
        """
        detections = self.detections_from_annotations(
            annotations.select(channel="CRED_Detection"),
            argdict.get("detection_threshold", 0.5),
        )

        return detections

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
        ]:
            del model_args[key]

        model_args["in_samples"] = self.in_samples
        model_args["in_channels"] = self.in_channels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["original_compatible"] = self.original_compatible

        return model_args


class BlockCNN(nn.Module):
    """
    CNN Block
    """

    def __init__(self, input_channels, filters, kernel_size):
        super().__init__()

        norms = [
            nn.BatchNorm2d(input_channels, eps=1e-3),
            nn.BatchNorm2d(filters, eps=1e-3),
        ]
        convs = [
            nn.Conv2d(input_channels, filters, kernel_size, padding=kernel_size // 2),
            nn.Conv2d(filters, filters, kernel_size, padding=kernel_size // 2),
        ]

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x):
        x = self.norms[0](x)
        x = torch.relu(x)
        x = self.convs[0](x)
        x = self.norms[1](x)
        x = torch.relu(x)
        x = self.convs[1](x)
        return x


class BlockBiLSTM(nn.Module):
    """
    BiLSTM Block
    """

    def __init__(self, input_channels, filters):
        super().__init__()

        lstms = [
            nn.LSTM(input_channels, filters, bidirectional=True, batch_first=True),
            nn.LSTM(2 * filters, filters, bidirectional=True, batch_first=True),
        ]
        self.dropout = nn.Dropout(0.3)

        self.lstms = nn.ModuleList(lstms)

    def forward(self, x):
        x = self.lstms[0](x)[0]
        x = self.dropout(x)
        y = self.lstms[1](x)[0]
        x = self.dropout(y) + x

        return x
