import copy
from typing import Any, TYPE_CHECKING

import numpy as np
import seisbench

if TYPE_CHECKING:
    import h5py


class ProbabilisticDASLabeller:
    """
    Create supervised labels for DAS from picks with probabilistic representation.
    """

    def __init__(
        self,
        annotation_mapping: list[str] | dict[str, str],
        noise_map: bool = True,
        output_labels: list[str] | None = None,
        label_shape: str = "gaussian",
        sigma: int = 10,
        key: tuple[str, str] = ("X", "y"),
    ):
        self.annotation_mapping, self.output_labels = (
            self._normalize_label_specification(
                annotation_mapping, noise_map, output_labels
            )
        )
        self.noise_map = noise_map
        self.label_shape = label_shape
        self.sigma = sigma
        self.key = key

    def __str__(self):
        return f"ProbabilisticDASLabeller (label_shape={self.label_shape}, sigma={self.sigma})"

    def __call__(self, state_dict):
        record, metadata = state_dict[self.key[0]]
        y = self.label(record, metadata)
        state_dict[self.key[1]] = (y, copy.deepcopy(metadata))

    @staticmethod
    def _normalize_label_specification(
        annotation_mapping: list[str] | dict[str, str],
        noise_map: bool = True,
        output_labels: list[str] | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        if isinstance(annotation_mapping, list):
            annotation_mapping = {x: x for x in annotation_mapping}

        if output_labels is None:
            output_labels = sorted(list(set(annotation_mapping.values())))
            if noise_map:
                output_labels.append("noise")
        else:
            if "noise" not in output_labels:
                output_labels.append("noise")
                seisbench.logger.warning(
                    "'noise' label was not specified in output_labels, but noise_column=True. "
                    "The 'noise' column has been appended at the end of the output_labels."
                )

        return annotation_mapping, output_labels

    def label(
        self, record: np.ndarray | h5py.Dataset, metadata: dict[str, Any]
    ) -> np.ndarray:
        labels = np.zeros((len(self.output_labels),) + record.shape, dtype=np.float32)
        annotations = metadata["__annotations__"]

        output_map = {label: i for i, label in enumerate(self.output_labels)}

        for ann_name, ann_values in annotations.items():
            if ann_name not in self.annotation_mapping:
                continue

            ann_type = self.annotation_mapping[ann_name]
            labels[output_map[ann_type]] += self._build_single_label_map(
                ann_values, record.shape
            )

        labels = np.clip(labels, 0, 1)
        if self.noise_map:
            labels /= np.clip(np.sum(labels, axis=0, keepdims=True), 1, np.inf)
            labels[-1] = 1 - np.sum(labels, axis=0)

        return labels

    def _build_single_label_map(
        self, ann_values: np.ndarray, shape: tuple[int, ...]
    ) -> np.ndarray:
        coord = np.arange(shape[0]).repeat(shape[1]).reshape(shape)
        rel_coord = (coord - ann_values.reshape(1, -1)) / self.sigma

        if self.label_shape == "gaussian":
            label = np.exp(-(rel_coord**2) / 2)
        elif self.label_shape == "triangle":
            label = np.clip(1 - np.abs(rel_coord), 0, 1).astype(np.float32)
        elif self.label_shape == "box":
            label = (np.abs(rel_coord) < 1).astype(np.float32)
        else:
            raise ValueError(f"Unknown label shape: {self.label_shape}")

        # Handle NaN columns
        label[:, np.isnan(ann_values)] = 0

        return label
