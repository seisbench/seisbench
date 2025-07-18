from typing import Any

import numpy as np
import torch
import torch.nn as nn

import seisbench.util as sbu

from .base import WaveformModel


class Skynet(WaveformModel):
    """
    Skynet is a phase picker with a U-Net architecture with a focus on regional phases and a long receptive field.
    The model offers first arrival pickers and multiphase pickers for Pn, Pg, Sn, and Sg phases.

    .. document_args:: seisbench.models Skynet
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.5)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (0, 0),
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 15000)
    _annotate_args["stacking"] = (_annotate_args["stacking"][0], "max")

    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases="PSN",
        sampling_rate=100,
        norm="peak",
        **kwargs,
    ):
        citation = (
            "Aguilar Suarez, A. L., & Beroza, G. (2025). "
            "Picking Regional Seismic Phase Arrival Times with Deep Learning. "
            "Seismica, 4(1). "
            "https://doi.org/10.26443/seismica.v4i1.1431"
        )

        super().__init__(
            citation=citation,
            in_samples=30000,
            output_type="array",
            pred_sample=(0, 30000),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        kernel_size = 7
        stride = 4

        pad_size = int(kernel_size / 2)

        self.conv1 = nn.Conv1d(
            in_channels, 8, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.conv2 = nn.Conv1d(8, 8, kernel_size=kernel_size, stride=1, padding="same")
        self.bn2 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.conv3 = nn.Conv1d(
            8, 8, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn3 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.conv4 = nn.Conv1d(8, 11, kernel_size=kernel_size, stride=1, padding="same")
        self.bn4 = nn.BatchNorm1d(num_features=11, eps=1e-3)
        self.conv5 = nn.Conv1d(
            11, 11, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn5 = nn.BatchNorm1d(num_features=11, eps=1e-3)
        self.conv6 = nn.Conv1d(
            11, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn6 = nn.BatchNorm1d(num_features=16, eps=1e-3)
        self.conv7 = nn.Conv1d(
            16, 16, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn7 = nn.BatchNorm1d(num_features=16, eps=1e-3)
        self.conv8 = nn.Conv1d(
            16, 22, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn8 = nn.BatchNorm1d(num_features=22, eps=1e-3)
        self.conv9 = nn.Conv1d(
            22, 22, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn9 = nn.BatchNorm1d(num_features=22, eps=1e-3)
        self.conv10 = nn.Conv1d(
            22, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn10 = nn.BatchNorm1d(num_features=32, eps=1e-3)
        # extra from original UNet
        self.conv11 = nn.Conv1d(
            32, 32, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn11 = nn.BatchNorm1d(num_features=32, eps=1e-3)
        self.conv12 = nn.Conv1d(
            32, 40, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn12 = nn.BatchNorm1d(num_features=40, eps=1e-3)
        self.dconv0 = nn.ConvTranspose1d(
            40, 32, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bnd0 = nn.BatchNorm1d(num_features=32, eps=1e-3)
        self.dconv01 = nn.Conv1d(
            64, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd01 = nn.BatchNorm1d(num_features=32, eps=1e-3)
        #
        self.dconv1 = nn.ConvTranspose1d(
            32, 22, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bnd1 = nn.BatchNorm1d(num_features=22, eps=1e-3)
        self.dconv2 = nn.Conv1d(
            44, 22, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd2 = nn.BatchNorm1d(num_features=22, eps=1e-3)
        self.dconv3 = nn.ConvTranspose1d(
            22, 16, kernel_size=kernel_size, stride=stride, padding=pad_size - 1
        )
        self.bnd3 = nn.BatchNorm1d(num_features=16, eps=1e-3)
        self.dconv4 = nn.Conv1d(
            32, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd4 = nn.BatchNorm1d(num_features=16, eps=1e-3)
        self.dconv5 = nn.ConvTranspose1d(
            16, 11, kernel_size=kernel_size, stride=stride, padding=pad_size - 1
        )
        self.bnd5 = nn.BatchNorm1d(num_features=11, eps=1e-3)
        self.dconv6 = nn.Conv1d(
            22, 11, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd6 = nn.BatchNorm1d(num_features=11, eps=1e-3)
        self.dconv7 = nn.ConvTranspose1d(
            11, 8, kernel_size=kernel_size, stride=stride, padding=pad_size - 1
        )
        self.bnd7 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.dconv8 = nn.Conv1d(
            16, 8, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd8 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.dconv9 = nn.Conv1d(
            8, classes, kernel_size=kernel_size, stride=1, padding="same"
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, logits=False):
        X1 = torch.relu(self.bn1(self.conv1(X)))
        X2 = torch.relu(self.bn2(self.conv2(X1)))
        X3 = torch.relu(self.bn3(self.conv3(X2)))
        X4 = torch.relu(self.bn4(self.conv4(X3)))
        X5 = torch.relu(self.bn5(self.conv5(X4)))
        X6 = torch.relu(self.bn6(self.conv6(X5)))
        X7 = torch.relu(self.bn7(self.conv7(X6)))
        X8 = torch.relu(self.bn8(self.conv8(X7)))
        X9 = torch.relu(self.bn9(self.conv9(X8)))
        X10 = torch.relu(self.bn10(self.conv10(X9)))
        # extra from original UNet
        X10_a = torch.relu(self.bn11(self.conv11(X10)))
        X10_b = torch.relu(self.bn12(self.conv12(X10_a)))
        X10_c = torch.relu(self.bnd0(self.dconv0(X10_b)))
        X10_c = torch.cat(
            (
                X10_c,
                torch.zeros((X10_c.shape[0], X10_c.shape[1], 1), device=X10_c.device),
            ),
            dim=-1,
        )
        X10_c = torch.cat((X10, X10_c), dim=1)
        X10_d = torch.relu(self.bnd01(self.dconv01(X10_c)))
        X11 = torch.relu(self.bnd1(self.dconv1(X10_d)))
        X12 = torch.cat((X11, X8), dim=1)
        X12 = torch.relu(self.bnd2(self.dconv2(X12)))
        X13 = torch.relu(self.bnd3(self.dconv3(X12)))
        X14 = torch.relu(self.bnd4(self.dconv4(torch.cat((X13, X6), dim=1))))
        X15 = torch.relu(self.bnd5(self.dconv5(X14)))
        X15 = torch.cat(
            (X15, torch.zeros((X15.shape[0], X15.shape[1], 1), device=X15.device)),
            dim=2,
        )
        X16 = torch.relu(self.bnd6(self.dconv6(torch.cat((X15, X4), dim=1))))
        X17 = torch.relu(self.bnd7(self.dconv7(X16)))
        X17 = torch.cat(
            (X17, torch.zeros((X17.shape[0], X17.shape[1], 1), device=X17.device)),
            dim=2,
        )
        X18 = torch.relu(self.bnd8(self.dconv8(torch.cat((X17, X2), dim=1))))
        X19 = self.dconv9(X18)
        if logits:
            return X19
        else:
            return self.softmax(X19)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(axis=-1, keepdims=True)
        if self.norm == "std":
            std = batch.std(axis=-1, keepdims=True).mean(axis=-2, keepdims=True)
            batch = batch / (std + 1e-10)
        elif self.norm == "peak":
            peak = (
                batch.abs()
                .max(axis=-2, keepdims=True)[0]
                .max(axis=-1, keepdims=True)[0]
            )
            batch = batch / (peak + 1e-10)

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        # Transpose predictions to correct shape
        batch = torch.transpose(batch, -1, -2)
        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            batch[:, :prenan] = np.nan
        if postnan > 0:
            batch[:, -postnan:] = np.nan
        return batch

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete thresholds using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = sbu.PickList()
        for phase in self.labels:
            if phase == "N":
                # Don't pick noise
                continue

            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )

        picks = sbu.PickList(sorted(picks))

        return sbu.ClassifyOutput(self.name, picks=picks)

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm

        return model_args
