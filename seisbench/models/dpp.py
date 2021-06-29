from .base import WaveformPipeline, WaveformModel

import torch
import torch.nn as nn
import numpy as np
import math
from obspy.signal.trigger import trigger_onset


class DeepPhasePick(WaveformPipeline):
    """
    Note: ...

    """

    #TODO: update
    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases=None,
        eps=1e-10,
        sampling_rate=100,
        pred_sample=200, # should be different for P and S phases based on optimized hyperparameter
        original_compatible=False,
        **kwargs,
    ):
        #TODO: update
        citation = (
            "Soto, H. & Schurr, B. (2020). "
            "DeepPhasePick: A method for Detecting and Picking SeismicPhases from Local "
            "Earthquakes based on highly optimized Convolutional and Recurrent Deep Neural Networks. "
            "EarthArXiv. https://doi.org/10.31223/X5BC8B"
        )
        #TODO: update, give optimized hyperparameters and optional user-defined params here  ??
        super().__init__(
            citation=citation,
            output_type="point",
            in_samples=480,
            pred_sample=pred_sample,
            labels=phases,
            sampling_rate=sampling_rate,
            default_args={"stride": 10},
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        # self.eps = eps
        self._phases = phases
        self.original_compatible = original_compatible
        if phases is not None and len(phases) != classes:
            raise ValueError(
                f"Number of classes ({classes}) does not match number of phases ({len(phases)})."
            )


        # TODO: add functionality to link Detector and Picker
        # -> Detector and Picker are not part of the same model.
        # -> If Detector detects a P phase, then Picker(P) is called. If a S phase is detected, Picker(S) is called
        # -> Picker(P/S) picks the detected phase in a "picking" window defined
        # by optimized hyperparameters controlling its relative position with respect to the highest detection probability
        #
        # Detector(self.in_channels, self.classes)
        #
        # Picker()


class DPPDetector(WaveformModel):

    def __init__(self, input_channels, nclasses):
        super().__init__()

        # TODO: check if momentum is the same than in keras
        #
        # in_channels = 3
        # classes = 3
        self.in_channels = input_channels
        self.classes = nclasses
        self.stride = 1

        # # self.original_compatible = True
        # self.original_compatible = False

        # groups == in_channels in Conv1d is for “depthwise convolution”, equivalent to keras SeparableConv1D layer.

        # self.conv1 = Conv1dSame(12, 11, self.kernel_size, self.stride) # padding="same" from Conv1dSame as in phasenet ??
        # self.conv1 = nn.Conv1d(self.in_channels, 12, 17, self.stride, padding="same", groups=self.in_channels) # padding="same" as in keras (torch>1.9.0)
        self.conv1 = nn.Conv1d(self.in_channels, 12, 17, self.stride, padding=17//2, groups=self.in_channels)
        self.bn1 = nn.BatchNorm1d(12, eps=1e-3, momentum=0.99)
        self.dropout1 = nn.Dropout(0.25)
        # self.conv2 = nn.Conv1d(12, 24, 11, self.stride, padding="same", groups=12)
        self.conv2 = nn.Conv1d(12, 24, 11, self.stride, padding=11//2, groups=12)
        self.bn2 = nn.BatchNorm1d(24, eps=1e-3, momentum=0.99)
        self.dropout2 = nn.Dropout(0.25)
        # self.conv3 = nn.Conv1d(24, 48, 5, self.stride, padding="same", groups=24)
        self.conv3 = nn.Conv1d(24, 48, 5, self.stride, padding=5//2, groups=24)
        self.bn3 = nn.BatchNorm1d(48, eps=1e-3, momentum=0.99)
        self.dropout3 = nn.Dropout(0.3)
        # self.conv4 = nn.Conv1d(48, 96, 9, self.stride, padding="same", groups=48)
        self.conv4 = nn.Conv1d(48, 96, 9, self.stride, padding=9//2, groups=48)
        self.bn4 = nn.BatchNorm1d(96, eps=1e-3, momentum=0.99)
        self.dropout4 = nn.Dropout(0.4)
        # self.conv5 = nn.Conv1d(96, 192, 17, self.stride, padding="same", groups=96)
        self.conv5 = nn.Conv1d(96, 192, 17, self.stride, padding=17//2, groups=96)
        self.bn5 = nn.BatchNorm1d(192, eps=1e-3, momentum=0.99)
        self.dropout5 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(2880, 50)
        self.bn6 = nn.BatchNorm1d(50, eps=1e-3, momentum=0.99)
        self.dropout6 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, self.classes)

        self.pool = nn.MaxPool1d(2, 2)

        self.activation1 = torch.relu
        self.activation2 = torch.sigmoid
        self.activation3 = torch.nn.Softmax(dim=1)

    def forward(self, x):

        x = self.bn1(self.activation1((self.conv1(x))))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.bn2(self.activation1((self.conv2(x))))
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.bn3(self.activation1((self.conv3(x))))
        x = self.pool(x)
        x = self.dropout3(x)

        x = self.bn4(self.activation1((self.conv4(x))))
        x = self.pool(x)
        x = self.dropout4(x)

        x = self.bn5(self.activation2((self.conv5(x))))
        x = self.pool(x)
        x = self.dropout5(x)

        # if self.original_compatible:
        #     # Permutation is required to be consistent with the following fully connected layer
        #     x = x.permute(0, 2, 1)
        x = torch.flatten(x, 1)

        x = self.bn6(self.activation1(self.fc1(x)))
        x = self.dropout6(x)
        x = self.fc2(x)
        x = self.activation3(x)

        return x


class DPPPicker(WaveformModel):

    def __init__(self, mode):
        super().__init__()

        # TODO: implement LSTM recurrent dropout

        self.mode = mode

        # # self.original_compatible = True
        # self.original_compatible = False

        if self.mode == 'P':
            self.lstm1 = nn.LSTM(1, 100, bidirectional=True, batch_first=True)
            self.lstm2 = nn.LSTM(200, 160, bidirectional=True, batch_first=True)
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.35)
            self.fc1 = nn.Linear(320, 1)
        elif self.mode == 'S':
            self.lstm1 = nn.LSTM(2, 20, bidirectional=True, batch_first=True)
            self.lstm2 = nn.LSTM(40, 30, bidirectional=True, batch_first=True)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.45)
            self.fc1 = nn.Linear(60, 1)

        self.activation = torch.sigmoid

    def forward(self, x):

        x = self.lstm1(x)[0]
        x = self.dropout1(x)
        x = self.lstm2(x)[0]
        x = self.dropout2(x)

        # keras TimeDistributed layer is applied by:
        # -> reshaping from (batch, sequence, *) to (batch * sequence, *)
        # -> then applying the layer,
        # -> then reshaping back to (batch, sequence, *)
        #
        shape_save = x.shape
        x = x.reshape((-1,) + x.shape[2:])

        x = self.activation(self.fc1(x))
        x = x.reshape(shape_save[:2] + (1,))

        return x
