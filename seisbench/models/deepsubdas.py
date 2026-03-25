from typing import Optional, Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

import seisbench.util as sbu

try:
    import torchvision
except ImportError:
    torchvision = sbu.MissingOptionalDependency("torchvision", "torchvision")

from .das_base import (
    DASAnnotateCallback,
    DASModel,
    DASPickingCallback,
    PatchingStructure,
)


class DeepSubDAS(DASModel):
    """
    The DeepSubDAS model by Xiao et al. (2025).

    The model supports loading pretrained weights for the underlying DeepLabV3 model. For available weights see the
    torchvision documentation at https://docs.pytorch.org/vision/main/models/deeplabv3.html. Note that this is different
    from the :py:func:`from_pretrained` method. While the former loads a generic segmentation model that might be a
    good starting point for training new models, the latter loads models specifically trained for seismic phase picking.

    :param deeplab_weights: The weights to use for the DeepLabV3 model.
    :param deeplab_weights_backbone: The weights to use for the DeepLabV3 backbone.
    :param deeplab_backbone: The DeepLabV3 backbone to use. Currently, supports `resnet50`` and ``resnet101``.
    :param remove_common_mode: If true, removes mean along the channel axis for each sample.
    :param transpose_for_model: If true, transposes the input before passing it to the model. This is implemented for
                                to ensure compatibility with the original model.
    """

    def __init__(
        self,
        deeplab_weights: Optional[str] = None,
        deeplab_weights_backbone: Optional[str] = None,
        deeplab_backbone: str = "resnet101",
        remove_common_mode: bool = False,
        transpose_for_model: bool = False,
        **kwargs,
    ):
        citation = (
            "Xiao, H., Tilmann, F., van den Ende, M., Rivet, D., Loureiro, A., Tsuji, T., ... "
            "& Denolle, M. A. (2026). DeepSubDAS: an earthquake phase picker from submarine "
            "distributed acoustic sensing data. Geophysical Journal International, 245(2), ggag061."
        )
        patching_structure = PatchingStructure(
            in_samples=4000,
            in_channels=2000,
            out_samples=4000,
            out_channels=2000,
            range_samples=(0, 4000),
            range_channels=(0, 2000),
        )
        super().__init__(
            citation=citation,
            patching_structure=patching_structure,
            annotate_keys=["P", "S"],
            annotate_forward_kwargs={"logits": False},
            **kwargs,
        )
        self.deeplab_weights = deeplab_weights
        self.deeplab_weights_backbone = deeplab_weights_backbone
        self.deeplab_backbone = deeplab_backbone
        self.remove_common_mode = remove_common_mode
        self.transpose_for_model = transpose_for_model

        if deeplab_backbone == "resnet101":
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                weights=deeplab_weights,
                weights_backbone=deeplab_weights_backbone,
            )
        elif deeplab_backbone == "resnet50":
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=deeplab_weights,
                weights_backbone=deeplab_weights_backbone,
            )
        else:
            raise ValueError(f"Unknown or unsupported backbone {deeplab_backbone}")

        # Modify input and output layers
        self.model.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.classifier[4] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))

    def normalize_batch(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        x = (x - x.mean(dim=-2, keepdim=True)) / (x.std(dim=-2, keepdim=True) + eps)
        if self.remove_common_mode:  # Remove average along channel dimension
            x = x - x.mean(dim=-1, keepdim=True)
        return x

    def forward(self, x, logits=False, argdict: Optional[dict[str, Any]] = None):
        x = self.normalize_batch(x)
        x = x.unsqueeze(1)  # Add fake channel dimension
        if self.transpose_for_model:
            x = x.permute(0, 1, 3, 2)
        y = self.model(x)["out"]
        if self.transpose_for_model:
            y = y.permute(0, 1, 3, 2)
        if not logits:
            y = F.softmax(y, dim=1)
        return {"full": y, "P": y[:, 0], "S": y[:, 1]}

    def get_model_args(self):
        model_args = super().get_model_args()
        model_args = {
            **model_args,
            **{
                "deeplab_weights": self.deeplab_weights,
                "deeplab_weights_backbone": self.deeplab_weights_backbone,
                "deeplab_backbone": self.deeplab_backbone,
                "transpose_for_model": self.transpose_for_model,
                "remove_common_mode": self.remove_common_mode,
            },
        }
        del model_args["citation"]
        return model_args

    @property
    def classify_callback(self) -> Type[DASAnnotateCallback]:
        return DASPickingCallback
