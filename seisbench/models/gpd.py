from .base import SeisBenchModel, WaveformModel
import seisbench

import torch
import torch.nn as nn
import obspy
import numpy as np


class GPD(SeisBenchModel, WaveformModel):
    # TODO: How to handle filtering (e.g. highpass/lowpass)?
    def __init__(self, in_channels=3, classes=3, eps=1e-10):
        citation = (
            "Ross, Z. E., Meier, M.-A., Hauksson, E., & Heaton, T. H. (2018). "
            "Generalized Seismic Phase Detection with Deep Learning. "
            "ArXiv:1805.01075 [Physics]. http://arxiv.org/abs/1805.01075"
        )
        super().__init__(name="gpd", citation=citation)

        self.in_channels = in_channels
        self.classes = classes
        self.eps = eps

        # TODO: Verify Pooling
        # TODO: Verify order of activation, pooling and batch norm
        self.conv1 = nn.Conv1d(in_channels, 32, 21)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 21)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 21)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 21)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(1536, 200)
        self.bn5 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn6 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, classes)

        self.activation = torch.relu

        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = x / (
            torch.max(torch.max(x, dim=-1, keepdims=True)[0], dim=-2, keepdims=True)[0]
            + self.eps
        )
        x = self.activation(self.pool(self.bn1(self.conv1(x))))
        x = self.activation(self.pool(self.bn2(self.conv2(x))))
        x = self.activation(self.pool(self.bn3(self.conv3(x))))
        x = self.activation(self.pool(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)

        x = self.activation(self.bn5(self.fc1(x)))
        x = self.activation(self.bn6(self.fc2(x)))
        x = self.fc3(x)

        if self.classes == 1:
            return torch.sigmoid(x)
        else:
            return torch.softmax(x, -1)

    @property
    def device(self):
        return next(self.parameters()).device

    def annotate(self, stream, strict=True, prediction_rate=100, **kwargs):
        """
        Annotates the stream using a sliding window approach.
        :param stream:
        :param strict:
        :param kwargs:
        :return:
        """
        if prediction_rate > 100:
            raise ValueError("Prediction rate needs to be below sampling rate.")

        stream = stream.copy()
        stream.merge(-1)

        output = obspy.Stream()
        if len(stream) == 0:
            return output

        if self.has_mismatching_records(stream):
            raise ValueError(
                "Detected multiple records for the same time and component that did not agree."
            )

        self.resample(stream, 100)

        groups = self.groups_stream_by_instrument(stream)

        for group in groups:
            times, data = self.stream_to_arrays(
                group, self.component_order, strict=strict
            )
            if 100 % prediction_rate != 0:
                seisbench.logger.warning(
                    "Prediction rate is no real divisor of sampling rate. "
                    "Adjusting to next possible prediction rate."
                )

            d = 100 // prediction_rate
            true_prediction_rate = 100 / d
            for t0, block in zip(times, data):
                mids = np.arange(200, block.shape[1] - 200, d)

                fragments = np.stack(
                    [block[:, m - 200 : m + 200] for m in mids], axis=0
                )
                fragments = torch.tensor(
                    fragments, device=self.device, dtype=torch.float32
                )

                with torch.no_grad():
                    train_mode = self.training
                    try:
                        self.eval()
                        # TODO: Add batching
                        pred = self(fragments)
                    finally:
                        if train_mode:
                            self.train()
                    pred = pred.cpu().numpy()

                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)

                for i in range(pred.shape[1]):
                    trace = group[0]
                    output.append(
                        obspy.Trace(
                            pred[:, i],
                            {
                                "starttime": t0 + 2,
                                "sampling_rate": true_prediction_rate,
                                "network": trace.stats.network,
                                "station": trace.stats.station,
                                "location": trace.stats.location,
                                "channel": f"GPD{i}",
                            },
                        )
                    )

        return output

    def classify(self, stream, *args, **kwargs):
        raise NotImplementedError("Classify is not yet implemented")
