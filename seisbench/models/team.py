import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import WaveformModel


class PhaseTEAM(WaveformModel):
    """
    A multi-station phase picking model, based on a combination of the PhaseNet and the TEAM architecture.

    .. document_args:: seisbench.models PhaseTEAM
    """

    def __init__(
        self,
        in_channels=3,
        classes=2,
        phases=("P", "P"),
        sampling_rate=20,
        transformer_layers=2,
        transformer_compression=32,
        transformer_dim=256,
        norm="std",
        **kwargs,
    ):
        citation = "TBD"

        super().__init__(
            citation=citation,
            in_samples=3001,
            output_type="array",
            default_args={"overlap": 250},
            pred_sample=(0, 3001),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.transformer_dim = transformer_dim
        self.transformer_layers = transformer_layers
        self.transformer_compression = transformer_compression
        self.norm = norm

        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(classes=classes)

        self.conv_pre_transformer = nn.Conv1d(
            128, transformer_compression, 1, padding="same", bias=True
        )
        self.conv_post_transformer = nn.Conv1d(
            transformer_compression, 128, 1, padding="same", bias=True
        )

        # Shape for transformer
        self.mlp1 = MLP(
            dims=[transformer_compression * 12, self.transformer_dim],
            activation=torch.relu,
        )
        self.mlp2 = MLP(
            dims=[self.transformer_dim, transformer_compression * 12],
            activation=torch.relu,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=4,
            dim_feedforward=2 * self.transformer_dim,
            dropout=0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.transformer_layers,
            norm=nn.LayerNorm(self.transformer_dim),
            enable_nested_tensor=False,
        )

    def forward(self, x, logits=False):
        """
        :param x: Windows
        :return:
        """
        input_shape = x.shape  # (batch, stations, channels, samples)
        padding_mask = (x == 0).all(axis=-1).all(axis=-1)
        x = x.reshape((-1,) + input_shape[2:])  # (batch * stations, channels, samples)
        x, skips = self.encoder(x)

        # Apply transformer
        x_comp = torch.relu(self.conv_pre_transformer(x))
        transformer_input = x_comp.reshape(
            -1, self.transformer_compression * 12
        )  # (batch * stations, filters * samples)
        transformer_input = self.mlp1(transformer_input)
        transformer_input = transformer_input.reshape(
            input_shape[:2] + (self.transformer_dim,)
        )  # (batch, stations, features)

        transformer_output = self.transformer(
            transformer_input, src_key_padding_mask=padding_mask
        )

        transformer_output = transformer_output.reshape(
            -1, self.transformer_dim
        )  # (batch * stations, features)
        transformer_output = self.mlp2(transformer_output)
        transformer_output = transformer_output.reshape(
            -1, self.transformer_compression, 12
        )
        x_uncomp = torch.relu(self.conv_post_transformer(transformer_output))

        x = x + x_uncomp

        y = self.decoder(x, skips)

        output_shape = list(input_shape)
        output_shape[-2] = self.classes
        output_shape = tuple(output_shape)

        y = y.reshape(output_shape)  # back to input shape

        if logits:
            return y
        else:
            return torch.sigmoid(y)


class Encoder(nn.Module):
    """
    The encoder branch from PhaseNet
    """

    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        self.inc = nn.Conv1d(
            self.in_channels, self.filters_root, self.kernel_size, padding="same"
        )
        self.in_bn = nn.BatchNorm1d(8, eps=1e-3)

        self.down_branch = nn.ModuleList()

        last_filters = self.filters_root
        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

    def forward(self, x):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        return x, skips


class Decoder(nn.Module):
    """
    The decoder branch from PhaseNet
    """

    def __init__(self, classes):
        super().__init__()

        self.classes = classes
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root
        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            last_filters = filters

        for i in range(self.depth - 1):
            filters = int(2 ** (3 - i) * self.filters_root)
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")

    def forward(self, x, skips):
        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)

        return x

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)


class MLP(nn.Module):
    """
    A simple multi-layer perceptron
    """

    def __init__(self, dims, activation=lambda x: x, bias=True):
        super().__init__()
        assert len(dims) >= 2

        self.fcs = nn.ModuleList(
            [
                nn.Linear(dim_in, dim_out, bias=bias)
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ]
        )

        self.activation = activation

    def forward(self, x):
        for fc in self.fcs:
            x = self.activation(fc(x))

        return x

    def parameters(self):
        return itertools.chain.from_iterable(fc.parameters() for fc in self.fcs)
