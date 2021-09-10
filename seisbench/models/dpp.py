from .base import WaveformPipeline, WaveformModel, CustomLSTM, ActivationLSTMCell

import torch
import torch.nn as nn


class DeepPhasePick(WaveformPipeline):
    """
    Note: ...

    """

    # TODO: update
    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases=None,
        sampling_rate=100,
        pred_sample=200,  # should be different for P and S phases based on optimized hyperparameter
        original_compatible=False,
        **kwargs,
    ):
        # TODO: update
        citation = (
            "Soto, H. & Schurr, B. (2020). "
            "DeepPhasePick: A method for Detecting and Picking SeismicPhases from Local "
            "Earthquakes based on highly optimized Convolutional and Recurrent Deep Neural Networks. "
            "EarthArXiv. https://doi.org/10.31223/X5BC8B"
        )
        # TODO: update, give optimized hyperparameters and optional user-defined params here  ??
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


class SeparableConv1d(nn.Module):
    """
    A depthwise separable convolution combine from two convolutions:
    1. A grouped convolution
    2. A pointwise convolution

    Assumes padding="same" and odd kernel_size
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv1d, self).__init__()
        assert kernel_size % 2 == 1  # Required for padding
        self.depthwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=kernel_size // 2,
        )
        self.pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DPPDetector(WaveformModel):
    def __init__(self, input_channels=3, nclasses=3):
        super().__init__()

        # TODO: check if momentum is the same than in keras

        self.in_channels = input_channels
        self.classes = nclasses
        self.stride = 1

        self.conv1 = SeparableConv1d(self.in_channels, 12, 17)
        self.bn1 = nn.BatchNorm1d(12, eps=1e-3, momentum=0.99)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = SeparableConv1d(12, 24, 11)
        self.bn2 = nn.BatchNorm1d(24, eps=1e-3, momentum=0.99)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = SeparableConv1d(24, 48, 5)
        self.bn3 = nn.BatchNorm1d(48, eps=1e-3, momentum=0.99)
        self.dropout3 = nn.Dropout(0.3)
        self.conv4 = SeparableConv1d(48, 96, 9)
        self.bn4 = nn.BatchNorm1d(96, eps=1e-3, momentum=0.99)
        self.dropout4 = nn.Dropout(0.4)
        self.conv5 = SeparableConv1d(96, 192, 17)
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

        x = torch.flatten(x, 1)

        x = self.bn6(self.activation1(self.fc1(x)))
        x = self.dropout6(x)
        x = self.fc2(x)
        x = self.activation3(x)

        return x


class DPPPicker(WaveformModel):
    def __init__(self, mode):
        super().__init__()

        self.mode = mode

        if self.mode == "P":
            """
            Custom LSTM removed for performance reasons.
            Could probably be fixed with JIT CustomLSTM.

            self.lstm1 = CustomLSTM(
                ActivationLSTMCell,
                1,
                100,
                bidirectional=True,
                recurrent_dropout=0.25,
                gate_activation=torch.sigmoid,
            )
            self.lstm2 = CustomLSTM(
                ActivationLSTMCell,
                200,
                160,
                bidirectional=True,
                recurrent_dropout=0.25,
                gate_activation=torch.sigmoid,
            )
            """
            self.lstm1 = nn.LSTM(1, 100, bidirectional=True)
            self.lstm2 = nn.LSTM(200, 160, bidirectional=True)
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.35)
            self.fc1 = nn.Linear(320, 1)
        elif self.mode == "S":
            """
            See remark for P mode above

            self.lstm1 = CustomLSTM(
                ActivationLSTMCell,
                2,
                20,
                bidirectional=True,
                recurrent_dropout=0.25,
                gate_activation=torch.sigmoid,
            )
            self.lstm2 = CustomLSTM(
                ActivationLSTMCell,
                40,
                30,
                bidirectional=True,
                recurrent_dropout=0.25,
                gate_activation=torch.sigmoid,
            )
            """
            self.lstm1 = nn.LSTM(2, 20, bidirectional=True)
            self.lstm2 = nn.LSTM(40, 30, bidirectional=True)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.45)
            self.fc1 = nn.Linear(60, 1)

        self.activation = torch.sigmoid

    def forward(self, x):
        # Permute shapes to match LSTM  --> (seq, batch, channels)

        x = x.permute(2, 0, 1)  # (batch, channels, seq) --> (seq, batch, channels)
        x = self.lstm1(x)[0]
        x = self.dropout1(x)
        x = self.lstm2(x)[0]
        x = self.dropout2(x)
        x = x.permute(1, 0, 2)  # (seq, batch, channels) --> (batch, seq, channels)

        # keras TimeDistributed layer is applied by:
        # -> reshaping from (batch, sequence, *) to (batch * sequence, *)
        # -> then applying the layer,
        # -> then reshaping back to (batch, sequence, *)
        #
        shape_save = x.shape
        x = x.reshape((-1,) + x.shape[2:])

        x = self.activation(self.fc1(x))
        x = x.reshape(shape_save[:2] + (1,))
        x = x.squeeze(-1)

        return x
