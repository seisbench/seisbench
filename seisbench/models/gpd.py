from .base import SeisBenchModel, WaveformModel

import torch
import torch.nn as nn


class GPD(SeisBenchModel, WaveformModel):
    def __init__(self, in_channels=3, classes=3):
        citation = (
            "Ross, Z. E., Meier, M.-A., Hauksson, E., & Heaton, T. H. (2018). "
            "Generalized Seismic Phase Detection with Deep Learning. "
            "ArXiv:1805.01075 [Physics]. http://arxiv.org/abs/1805.01075"
        )
        super().__init__(name="gpd", citation=citation)

        self.in_channels = in_channels
        self.classes = classes

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

    def annotate(self, stream, *args, **kwargs):
        raise NotImplementedError("Annotate is not yet implemented")

    def classify(self, stream, *args, **kwargs):
        raise NotImplementedError("Classify is not yet implemented")
