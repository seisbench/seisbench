from typing import Any, Optional

import numpy as np
import torch
import seisbench

from .das_base import DASModel, PatchingStructure
from .base import WaveformModel


class DASWaveformModelWrapper(DASModel):
    """
    This class is a wrapper to allow applying WaveformModels (trained for regular seismic data) to DAS datasets.
    The models are applied channel by channel.

    Example usage:

    .. code-block:: python

        base_model = PhaseNet.from_pretrained("instance")
        model = DASWaveformModelWrapper(base_model)


    :param model: The WaveformModel to apply to DAS data
    :param component_strategy: The strategy to transform the single-component DAS channels into three-component data for
                               the model. Supports ``clone`` (provide same channel for each compoennt) and ``pad``
                               (provide channel as first component, zero-padding for second and third component).

    """

    _annotate_args = DASModel._annotate_args.copy()

    def __init__(
        self, model: WaveformModel, component_strategy: str = "clone", **kwargs
    ):
        if not isinstance(model, WaveformModel):
            raise ValueError("Can only wrap WaveformModels.")
        if not model.output_type == "array":
            raise ValueError("Only 'array' models are supported.")

        if model.sampling_rate is not None:
            dt_range = (0.5 / model.sampling_rate, 2.0 / model.sampling_rate)
        else:
            dt_range = None

        annotate_keys = [x for x in model.labels if x != "N"]  # Drop noise output

        super().__init__(
            citation=model.citation,
            dt_range=dt_range,
            annotate_keys=annotate_keys,
            filter_samples=self._get_filter_args(model, dt_range),
            **kwargs,
        )

        self.model = model
        self.component_strategy = component_strategy
        self.default_args.update(model.default_args)

        model_annotate_args = model._annotate_args.copy()
        del model_annotate_args[
            "batch_size"
        ]  # Otherwise we get an unreasonable batch size
        self._annotate_args.update(model_annotate_args)
        if "overlap" in self._annotate_args:
            del self._annotate_args["overlap"]  # Use overlap_samples instead

    @property
    def component_strategy(self):
        return self._component_strategy

    @component_strategy.setter
    def component_strategy(self, value):
        if value not in ["clone", "pad"]:
            raise ValueError("component_strategy must be either 'clone' or 'pad'.")
        self._component_strategy = value

    @staticmethod
    def _get_filter_args(
        model: WaveformModel, dt_range: Optional[tuple[float, float]]
    ) -> Optional[tuple[str, dict[str, Any]]]:
        if model.filter_args is None:
            return None
        else:
            if len(model.filter_args) != 1 or isinstance(model.filter_args, dict):
                seisbench.logger.warning(
                    "Automatic filter inference failed due to incompatible filter specification. "
                    "Will not apply any filter."
                )
                return None

            # As the DAS model allows for a wider frequency range than the WaveformModel,
            # we sometimes need to adjust the filter frequency.
            if dt_range is not None:
                max_freq = (
                    0.999999 * 0.5 / dt_range[1]
                )  # No filter frequency can be above half the Nyquist
            else:
                max_freq = np.inf

            filter_name = model.filter_args[0]
            filter_kwargs = model.filter_kwargs
            corners = filter_kwargs["corners"]
            if filter_kwargs.get("zerophase", False):
                seisbench.logger.warning(
                    "Zero phase filtering not supported for DASWaveformModelWrapper. "
                    "Doubling filter order."
                )
                corners *= 2

            if filter_name == "bandpass":
                return "iirfilter", {
                    "N": corners,
                    "btype": "bandpass",
                    "ftype": "butter",
                    "Wn": [
                        filter_kwargs["freqmin"],
                        min(filter_kwargs["freqmax"], max_freq),
                    ],
                }
            elif filter_name == "bandstop":
                return "iirfilter", {
                    "N": corners,
                    "btype": "bandstop",
                    "ftype": "butter",
                    "Wn": [
                        filter_kwargs["freqmin"],
                        min(filter_kwargs["freqmax"], max_freq),
                    ],
                }
            elif filter_name == "lowpass":
                return "iirfilter", {
                    "N": corners,
                    "btype": "lowpass",
                    "ftype": "butter",
                    "Wn": min(filter_kwargs["freq"], max_freq),
                }
            elif filter_name == "highpass":
                return "iirfilter", {
                    "N": corners,
                    "btype": "highpass",
                    "ftype": "butter",
                    "Wn": min(filter_kwargs["freq"], max_freq),
                }
            else:
                seisbench.logger.warning(
                    "Automatic filter inference failed due to unsupported filter type "
                    f"('{filter_name}'). Will not apply any filter."
                )
                return None

    def get_patching_structure(
        self, data_shape: tuple[float, float], argdict: dict[str, Any]
    ) -> PatchingStructure:
        n_samples = int(np.floor(data_shape[0]))
        n_channels = int(np.floor(data_shape[1]))
        in_samples, pred_samples = self.model._get_in_pred_samples(np.empty(n_samples))
        in_channels = min(1024, n_channels)

        overlap_samples = self._argdict_get_with_default(argdict, "overlap_samples")
        if overlap_samples < 1:
            overlap_samples = int(overlap_samples * in_samples)

        blinding = self._argdict_get_with_default(argdict, "blinding")

        return PatchingStructure(
            in_channels=in_channels,
            out_channels=in_channels,
            range_channels=(0, in_channels),
            overlap_channels=0,
            out_samples=pred_samples[1] - pred_samples[0] - blinding[0] - blinding[1],
            in_samples=in_samples,
            range_samples=(
                pred_samples[0] + blinding[0],
                pred_samples[1] - blinding[1],
            ),  # Note the slightly different convention here
            overlap_samples=overlap_samples,
        )

    def forward(self, x: torch.Tensor, argdict: Optional[dict[str, Any]] = None):
        # x shape: (batch, samples, channels_das)
        x_original_shape = x.shape
        x = x.permute(0, 2, 1)  # -> (batch, channels_das, samples)
        x = x.reshape((-1, 1, x.shape[2]))  # -> (batch * channels_das, 1, samples)

        # Waveform model input shape: (batch, channels_3c, samples)
        n_components = len(self.model.component_order)
        if self.component_strategy == "clone":
            x = x.repeat(1, n_components, 1)
        elif self.component_strategy == "pad":
            x = torch.concatenate(
                [x] + (n_components - 1) * [torch.zeros_like(x)], dim=1
            )
        else:
            raise ValueError(f"Unknown strategy {self.component_strategy}")

        preprocessed = self.model.annotate_batch_pre(x, argdict=argdict)
        if isinstance(preprocessed, tuple):  # Contains piggyback information
            assert len(preprocessed) == 2
            preprocessed, piggyback = preprocessed
        else:
            piggyback = None

        preds = self.model(preprocessed)

        preds = self.model.annotate_batch_post(
            preds, piggyback=piggyback, argdict=argdict
        )

        output = {}
        for i, key in enumerate(self.model.labels):
            if key == "N":
                continue
            ann = preds[:, :, i].reshape(
                x_original_shape[0], x_original_shape[2], preds.shape[1]
            )  # -> (batch, channels_das, samples)
            ann = ann.permute(0, 2, 1)  # -> (batch, samples, channels_das)

            blinding = self._argdict_get_with_default(argdict, "blinding")
            b0, b1 = blinding[0], ann.shape[1] - blinding[1]
            ann = ann[:, b0:b1, :]

            output[key] = ann

        return output

    def save(self, *args, **kwargs):
        """
        This model does not provide a save function. Instead, save the underlying WaveformModel.
        """
        raise NotImplementedError(
            "Saving not supported for this type of model."
            "Instead, save the underlying WaveformModel."
        )
