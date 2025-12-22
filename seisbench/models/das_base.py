import asyncio
import bisect
import copy
import inspect
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from operator import attrgetter
from typing import Any, AsyncGenerator, Optional, Type

import numpy as np
import pandas as pd
import scipy.signal
from tqdm import tqdm

from scipy.signal import resample_poly

import torch
import torch.nn as nn
import seisbench
import seisbench.util as sbu
from .base import SeisBenchModel

try:
    import xdas
    from xdas import DataArray
except ImportError:
    xdas = sbu.MissingOptionalDependency("xdas", "xdas")
    DataArray = sbu.MissingOptionalDependency("xdas", "xdas")


# Type conversion map
TORCH_TO_NUMPY = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    return np.dtype(TORCH_TO_NUMPY[dtype])


@dataclass
class PatchCoordinate:
    """
    Coordinates of a patch in the input or output array.
    Denotes the upper-left corner of the patch and the dimensions along each axis.
    Note that coordinates can take non-integer values due to transformations.
    Callbacks should be able to handle this, e.g., by casting to int.
    """

    sample: float
    channel: float
    w_sample: int
    w_channel: int

    def __array__(self):
        return np.array([self.sample, self.channel, self.w_sample, self.w_channel])

    @property
    def sample_int(self) -> int:
        return int(np.round(self.sample))

    @property
    def channel_int(self) -> int:
        return int(np.round(self.channel))


@dataclass
class PatchingStructure:
    in_samples: int  # Number of input samples per patch along the time axis
    in_channels: int  # Number of input samples per patch along the channel axis
    out_samples: int  # Number of output samples per patch along the time axis
    out_channels: int  # Number of output samples per patch along the channel axis
    range_samples: tuple[
        int, int
    ]  # Range of the input covered by the output along the time axis
    range_channels: tuple[
        int, int
    ]  # Range of the input covered by the output along the channel axis
    overlap_samples: Optional[int] = (
        None  # Overlap between adjacent patches along the time axis
    )
    overlap_channels: Optional[int] = (
        None  # Overlap between adjacent patches along the channel axis
    )

    @property
    def shift_samples(self):
        return self.in_samples - self.overlap_samples

    @property
    def shift_channels(self):
        return self.in_channels - self.overlap_channels


class VirtualTransformedDataArray:
    """
    This class wraps a :py:class:`xdas.DataArray` and allows to apply a transformation to it on the fly. It is used to
    allow loading data from disk in chunks and only applying the transformations to the current chunk. This way, the
    total memory consumption is independent of the size of the underlying data.

    For resampling, the class uses ``scipy.signal.resample_poly``, which internally performs an upsampling, a zero-phase
    FIR filter and a downsampling step. To avoid boundary artifacts, extra samples are loaded for the filtering and
    truncated afterwards (at most 11 extra samples per side).

    After resampling, the class can apply an IIR filter along the sample axis. Only causal filters are supported because
    acausal filters would require loading the whole data at once. Only IIR filters are supported due to their higher
    computational efficiencies and because the number of filter states to cache between consecutive chunks is much
    lower. No filters along the channel axis are implemented. Instead, consider using an F-k filter.

    :param data: :py:class:`xdas.DataArray` to be transformed
    :param patching_structure: The structure of the patches to cut for annotation.
    :param resample_samples: Tuple of integers (up, down) defining the resampling factors along the sample axis.
    :param resample_channels: Tuple of integers (up, down) defining the resampling factors along the channel axis.
    :param filter_samples: Tuple of filter type and keyword arguments to pass to the filter. The filter type must be
                           the name of a filter design function in scipy.signal, e.g., "butter" or "cheby1". The filter
                           must support the ``output`` keyword argument, as this implementation relies on second-order
                           sections. The filter corners should be specified in Hz. The class will automatically pass
                           the sampling rate to the ``fs`` argument of the filter creation. No filter is applied if this
                           argument is None.
    :param channel_coord_name: Name of the coordinate in the data array that contains the channel coordinates.
    """

    def __init__(
        self,
        data: DataArray,
        patching_structure: PatchingStructure,
        resample_samples: tuple[int, int] = (1, 1),
        resample_channels: tuple[int, int] = (1, 1),
        filter_samples: Optional[tuple[str, dict[str, Any]]] = None,
        force_dtype: Optional[np.dtype] = None,
        channel_coord_name: Optional[str] = None,
    ):
        self.data = data

        if channel_coord_name is None:
            self.channel_coord_name = self.guess_channel_coord_name(data)
        else:
            self.channel_coord_name = channel_coord_name

        self.patching_structure = patching_structure
        gcd_samples = math.gcd(*resample_samples)
        self.resample_samples = (
            resample_samples[0] // gcd_samples,
            resample_samples[1] // gcd_samples,
        )
        gcd_channels = math.gcd(*resample_channels)
        self.resample_channels = (
            resample_channels[0] // gcd_channels,
            resample_channels[1] // gcd_channels,
        )
        self.filter_samples = filter_samples
        self.force_dtype = force_dtype

        self._check_transpose()
        self._validate_data()
        self._adjust_overlap()
        self._reset_filter()

    @staticmethod
    def guess_channel_coord_name(data: xdas.DataArray) -> str:
        for cand in ["channel", "distance"]:
            if cand in data.coords:
                return cand

        raise ValueError(
            "Could not identify channel name in data.coords. Please provide it explicitly."
        )

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of the transformed data. Always in ``(samples, channels)`` dimension order.

        Truncates the right end of the output to ensure ``(output_samples - in_samples)`` is divisible by the
        upsampling. The same is done for the channel axis. This is necessary to avoid fractional window offsets.
        """
        if self.transpose:
            source_channels, source_samples = self.data.shape
        else:
            source_samples, source_channels = self.data.shape

        samples_up = source_samples * self.resample_samples[0]
        out_samples = samples_up // self.resample_samples[1] + bool(
            samples_up % self.resample_samples[1]
        )

        channels_up = source_channels * self.resample_channels[0]
        out_channels = channels_up // self.resample_channels[1] + bool(
            channels_up % self.resample_channels[1]
        )

        # Make sure that the coordinates of each window are divisible by the upsampling factor
        out_samples -= (
            out_samples - self.patching_structure.in_samples
        ) % self.resample_samples[0]
        out_channels -= (
            out_channels - self.patching_structure.in_channels
        ) % self.resample_channels[0]

        return out_samples, out_channels

    @property
    def coords(self) -> dict[str, xdas.Coordinate]:
        # Calculates the output coordinates by interpolating the input coordinates
        # Relies on the data validation for ensuring underlying data is evenly spaced
        return {
            "time": xdas.InterpCoordinate(
                data={
                    "tie_indices": [0, self.shape[0] - 1],
                    "tie_values": [
                        self.data.coords["time"][0].data,
                        self.data.coords["time"][-1].data,
                    ],
                },
                dim="time",
            ),
            self.channel_coord_name: xdas.InterpCoordinate(
                data={
                    "tie_indices": [0, self.shape[1] - 1],
                    "tie_values": [
                        self.data.coords[self.channel_coord_name][0].data,
                        self.data.coords[self.channel_coord_name][-1].data,
                    ],
                },
                dim=self.channel_coord_name,
            ),
        }

    @property
    def dtype(self) -> np.dtype:
        if self.force_dtype is not None:
            return self.force_dtype
        return self.data.dtype

    @property
    def dx(self) -> float:
        channel_coords = self.coords[self.channel_coord_name]
        return float((channel_coords[1] - channel_coords[0]).values)

    @property
    def dt(self) -> float:
        time_coords = self.coords["time"]
        return (time_coords[1] - time_coords[0]) / np.timedelta64(1, "s")

    @property
    def filter_sos(self) -> Optional[np.ndarray]:
        if self.filter_samples is None:
            return None
        if self._filter_sos is None:
            self._filter_sos = getattr(scipy.signal, self.filter_samples[0])(
                **self.filter_samples[1], output="sos", fs=1 / self.dt
            ).astype(self.dtype)
        return self._filter_sos

    @staticmethod
    def estimate_theoretical_output_shape(
        data: DataArray,
        resample_samples: tuple[int, int],
        resample_channels: tuple[int, int],
    ) -> tuple[float, float]:
        return (
            data.shape[0] * resample_samples[0] // resample_samples[1],
            data.shape[1] * resample_channels[0] // resample_channels[1],
        )

    def __len__(self) -> int:
        n_samples, n_channels = self._get_n_patches()
        return n_samples * n_channels

    def _get_n_patches(self) -> tuple[int, int]:
        patching_structure = self.patching_structure
        shape = self.shape
        n_samples = (
            (shape[0] - patching_structure.in_samples)
            // patching_structure.shift_samples
            +
            # Extra sample if it's not a perfect fit
            bool(
                (shape[0] - patching_structure.in_samples)
                % patching_structure.shift_samples
            )
            + 1
        )

        n_channels = (
            (shape[1] - patching_structure.in_channels)
            // patching_structure.shift_channels
            + bool(
                (shape[1] - patching_structure.in_channels)
                % patching_structure.shift_channels
            )
            + 1
        )

        return n_samples, n_channels

    def _check_transpose(self):
        # Check that the data has the correct dimension order and set transposed argument accordingly
        if self.data.dims == ("time", self.channel_coord_name):
            self.transpose = False
        elif self.data.dims == (self.channel_coord_name, "time"):
            self.transpose = True
        else:
            raise ValueError(
                "Input array has unsupported dimension order. "
                f"Only ('time', '{self.channel_coord_name}') and ('{self.channel_coord_name}', 'time') are supported. "
            )

    def _validate_data(self):
        patching_structure = self.patching_structure
        # Check that the data is large enough
        if (
            self.shape[0] < patching_structure.in_samples
            or self.shape[1] < patching_structure.in_channels
        ):
            raise ValueError("Input array is too small to process.")

        # Check that the coordinates are uniformly spaced
        for key in "time", self.channel_coord_name:
            if self.data.coords[key].isinterp():
                simplified_coords = self.data.coords[
                    key
                ].simplify()  # Simplify but with zero tolerance
                if len(simplified_coords.tie_indices) > 2:
                    raise ValueError(
                        f"Coordinates for {key} are not uniformly spaced. "
                        f"You can try calling 'simplify' on the coordinates "
                        f"and provide an appropriate tolerance value to obtain uniformly spaced coordinates. "
                        f"Alternatively, you can splice the array into multiple subarrays with uniformly spaced"
                        f"coordinates and pass each one individually to the model."
                    )
            elif self.data.coords[key].isdense():
                diffs = np.diff(self.data.coords[key].data)
                if np.issubdtype(self.data.coords[key].data.dtype, np.datetime64):
                    atol = np.timedelta64(10, "us")
                else:
                    atol = 1e-8
                if not np.allclose(diffs, diffs[0], atol=atol):
                    raise ValueError(
                        f"Coordinates for {key} are not uniformly spaced. "
                        f"To pass them to the model, you either need to interpolate them or slice the array into "
                        f"multiple subarrays with uniformly spaced coordinates and pass each one individually to "
                        f"the model."
                    )
            else:
                raise ValueError(
                    f"Coordinates type for {key} is currently not supported. "
                    f"Only interpolated and dense coordinates are supported. "
                    f"Please convert your coordinates to dense or interpolated."
                )

    def _adjust_overlap(self):
        # Adjust overlap in patching structure to be compatible with the resampling
        # The shift value needs to be divisible by the upsampling factor
        self.patching_structure.overlap_samples += (
            self.patching_structure.shift_samples % self.resample_samples[0]
        )
        self.patching_structure.overlap_channels += (
            self.patching_structure.shift_channels % self.resample_channels[0]
        )

    async def __aiter__(self) -> AsyncGenerator[tuple[Any, PatchCoordinate], Any]:
        n_samples, n_channels = self._get_n_patches()
        self._reset_filter()
        for idx_samples in range(n_samples):
            for idx_channels in range(n_channels):
                coord = self._patch_coords(idx_samples, idx_channels)

                patch = await asyncio.to_thread(self._load_and_resample_patch, coord)
                if self.filter_samples is not None:
                    patch = await asyncio.to_thread(
                        self._filter_patch, patch, idx_samples, idx_channels
                    )

                yield patch, coord

    def _patch_coords(self, idx_samples: int, idx_channels: int) -> PatchCoordinate:
        patching_structure = self.patching_structure
        p_samples = idx_samples * (
            patching_structure.in_samples - patching_structure.overlap_samples
        )
        p_samples = min(p_samples, self.shape[0] - patching_structure.in_samples)
        p_channels = idx_channels * (
            patching_structure.in_channels - patching_structure.overlap_channels
        )
        p_channels = min(p_channels, self.shape[1] - patching_structure.in_channels)
        return PatchCoordinate(
            p_samples,
            p_channels,
            patching_structure.in_samples,
            patching_structure.in_channels,
        )

    def _load_and_resample_patch(self, coord: PatchCoordinate) -> np.ndarray:
        # Calculate source coordinates
        source_sample = (
            coord.sample_int // self.resample_samples[0]
        ) * self.resample_samples[1]
        source_channel = (
            coord.channel_int // self.resample_channels[0]
        ) * self.resample_channels[1]
        source_w_sample = int(
            np.ceil(
                (coord.w_sample / self.resample_samples[0]) * self.resample_samples[1]
            )
        )
        source_w_channel = int(
            np.ceil(
                (coord.w_channel / self.resample_channels[0])
                * self.resample_channels[1]
            )
        )

        # Add extra samples for resampling - makes sure padding results in an integer number of extra samples
        # Extra samples are designed to fit the FIR filter length in resample_poly
        extra_samples = int(np.ceil(20 * max(self.resample_samples) + 1))
        extra_samples += (
            self.resample_samples[1] - extra_samples % self.resample_samples[1]
        )
        extra_channels = int(np.ceil(20 * max(self.resample_channels) + 1))
        extra_channels += (
            self.resample_channels[1] - extra_channels % self.resample_channels[1]
        )

        # Ensure extra samples don't lead to negative indices
        pad_source_sample = max(0, source_sample - extra_samples)
        pad_source_channel = max(0, source_channel - extra_channels)

        # Shift the sample to the right to make sure the extra samples are divisible by the downsampling
        # This only has an effect if the pad_source_sample was below 0
        pad_source_sample += (
            source_sample - pad_source_sample
        ) % self.resample_samples[1]
        pad_source_channel += (
            source_channel - pad_source_channel
        ) % self.resample_channels[1]

        # Load data
        if self.transpose:
            patch = self.data.data[
                pad_source_channel : pad_source_channel
                + source_w_channel
                + 2 * extra_channels,
                pad_source_sample : pad_source_sample
                + source_w_sample
                + 2 * extra_samples,
            ].T
        else:
            patch = self.data.data[
                pad_source_sample : pad_source_sample
                + source_w_sample
                + 2 * extra_samples,
                pad_source_channel : pad_source_channel
                + source_w_channel
                + 2 * extra_channels,
            ]

        patch = np.asarray(
            patch
        )  # Triggers the actual loading in case of a virtual array

        # Resample samples and truncate padding
        patch = resample_poly(
            patch, self.resample_samples[0], self.resample_samples[1], axis=0
        )
        pad_samples = (
            (source_sample - pad_source_sample)
            // self.resample_samples[1]
            * self.resample_samples[0]
        )
        patch = patch[pad_samples : pad_samples + coord.w_sample]

        # Resample channels and truncate padding
        patch = resample_poly(
            patch, self.resample_channels[0], self.resample_channels[1], axis=1
        )
        pad_channels = (
            (source_channel - pad_source_channel)
            // self.resample_channels[1]
            * self.resample_channels[0]
        )
        patch = patch[:, pad_channels : pad_channels + coord.w_channel]

        patch = np.asarray(patch, dtype=self.dtype)

        return patch

    def _reset_filter(self):
        # Make sure the filter is recalculated
        self._filter_sos = None
        sos = self.filter_sos

        if self.filter_samples is None:
            self._sos_zi_base = None
            self._current_filter_zi = None
            self._next_filter_zi = None
        else:
            self._sos_zi_base = scipy.signal.sosfilt_zi(sos)  # Needs to be scaled
            self._current_filter_zi = None
            self._next_filter_zi = np.empty(
                (*self._sos_zi_base.shape, self.shape[1]), dtype=self.dtype
            )

    def _filter_patch(
        self, patch: np.ndarray, idx_samples: int, idx_channels: int
    ) -> np.ndarray:
        # Filter output data using an IIR filter (along sample axis) and cache overlap values
        # Assumes that patches are passed to the function in order.
        # This assumption is not checked, so the output behavior is undefined if it is violated.
        sos = self.filter_sos
        if sos is None:
            return patch

        n_samples, n_channels = self._get_n_patches()
        coord = self._patch_coords(idx_samples, idx_channels)
        next_row_coord = self._patch_coords(idx_samples + 1, idx_channels)

        if idx_channels == 0:
            self._current_filter_zi = self._next_filter_zi
            self._next_filter_zi = np.empty(
                (*self._sos_zi_base.shape, self.shape[1]), dtype=self.dtype
            )

        if idx_samples == 0:
            # Initialize filter state variables
            # Scale with initial amplitude of the channel
            self._current_filter_zi[
                ..., coord.channel_int : coord.channel_int + coord.w_channel
            ] = self._sos_zi_base[:, :, None] * patch[0, :]

        # Filter to next row
        zi = self._current_filter_zi[
            ..., coord.channel_int : coord.channel_int + coord.w_channel
        ]

        if idx_samples + 1 < n_samples:
            split = next_row_coord.sample_int - coord.sample_int
        else:
            split = patch.shape[0]

        part1 = patch[:split]
        part2 = patch[split:]

        part1_filtered, zf = scipy.signal.sosfilt(sos, part1, zi=zi, axis=0)

        # Store zi values
        self._next_filter_zi[
            ..., coord.channel_int : coord.channel_int + coord.w_channel
        ] = zf

        # There is no second part
        if part2.shape[0] == 0:
            return part1_filtered

        # Filter until the end of the patch
        part2_filtered, _ = scipy.signal.sosfilt(sos, part2, zi=zf, axis=0)

        return np.concatenate([part1_filtered, part2_filtered], axis=0)


class DASAnnotateCallback(ABC):
    """
    This abstract class describes the interface for callbacks used in the DAS annotate method.
    Callbacks will get streaming outputs from the annotate method, containing the different chunks after processing
    with the deep learning model. Different callbacks are available, e.g., for picking or for writing the full output.
    To implement a new callback, inherit from this class and implement the methods. Callbacks are stateful, allowing
    them, for example, to handle overlaps between adjacent chunks.

    .. warning:

        As callbacks are stateful, they should not be used in parallel.

    """

    def setup(
        self,
        data: VirtualTransformedDataArray,
        patching_structure: PatchingStructure,
        annotate_keys: list[str],
    ) -> None:
        """
        Setup step for the callback. This is called before the first chunk is processed and can be used to initialize
        state variables, e.g., the shape of the output or arrays for intermediate results.

        The setup step is optional, however, it is usually good practice to reset all state variables in the setup step.
        """
        pass

    def finalize(self) -> None:
        """
        Finalize step for the callback. This is called after the last chunk is processed and can be used to generate
        the final results based on the intermediate results processed in each chunk.

        The finalize step is optional.
        """
        pass

    @abstractmethod
    def handle_patch(
        self,
        annotations: dict[str, np.ndarray],
        in_coords: PatchCoordinate,
        out_coords: PatchCoordinate,
    ) -> None:
        """
        This method is called for each patch of the output after processing it with the deep learning model.
        Results inferred from this step should be stored in class variables.
        """
        pass

    @abstractmethod
    def get_results_dict(self) -> dict[str, Any]:
        """
        This method returns a dictionary with the results of the callback. It is used to generate the ClassifyOutput
        when using the callback through classify.
        """
        pass


class MultiCallback(DASAnnotateCallback):
    def __init__(self, callbacks: list[DASAnnotateCallback]):
        self.callbacks = callbacks

    def setup(
        self,
        data: VirtualTransformedDataArray,
        patching_structure: PatchingStructure,
        annotate_keys: list[str],
    ) -> None:
        for callback in self.callbacks:
            callback.setup(data, patching_structure, annotate_keys)

    def handle_patch(
        self,
        annotations: dict[str, np.ndarray],
        in_coords: PatchCoordinate,
        out_coords: PatchCoordinate,
    ) -> None:
        for callback in self.callbacks:
            callback.handle_patch(annotations, in_coords, out_coords)

    def finalize(self) -> None:
        for callback in self.callbacks:
            callback.finalize()

    def get_results_dict(self) -> dict[str, Any]:
        raise NotImplementedError(
            "MultiCallback does not support get_results_dict. "
            "Instead, query the results from each individual callback."
        )


class DASPickingCallback(DASAnnotateCallback):
    """
    Pick arrivals from probability curves using scipy.signal.find_peaks. The picking is performed independently on each
    channel, i.e., no continuity is assumed between channels.

    :param thresholds: Confidence thresholds for picking. Can be a single value for all phases,
                       or a dictionary with thresholds per phase.
    :param min_time_separation: Minimum time separation between two picks of the same phase in seconds.
    """

    def __init__(
        self,
        thresholds: float | dict[str, float] = 0.2,
        min_time_separation: float = 1.0,
    ):
        self._picks: dict[str, list[sbu.DASPick]] = {}
        self._picks_per_channel_idx: dict[tuple[str, int], list[sbu.DASPick]] = (
            defaultdict(list)
        )
        self.thresholds = thresholds
        self.min_time_separation = min_time_separation

        self._thresholds: dict[str, float] = {}
        self._annotate_keys: list[str] = []
        self._dt = 0.0
        self._output_coords = None
        self.channel_coord_name = None

    def setup(
        self,
        data: VirtualTransformedDataArray,
        patching_structure: PatchingStructure,
        annotate_keys: list[str],
    ) -> None:
        self.channel_coord_name = data.channel_coord_name
        self._annotate_keys = annotate_keys
        self._picks = {key: [] for key in annotate_keys}
        self._picks_per_channel_idx = defaultdict(list)
        self._dt = data.dt

        _, self._output_coords = DASModel.calc_output_shape_and_coordinates(
            data, patching_structure
        )

        if isinstance(self.thresholds, float):
            self._thresholds = {phase: self.thresholds for phase in annotate_keys}
        else:
            self._thresholds = self.thresholds

        for key in annotate_keys:
            if key not in self._thresholds:
                raise ValueError(f"Threshold for key {key} not specified.")

    def handle_patch(
        self,
        annotations: dict[str, np.ndarray],
        in_coords: PatchCoordinate,
        out_coords: PatchCoordinate,
    ) -> None:
        min_separation_samples = int(self.min_time_separation / self._dt)

        for key in self._annotate_keys:
            threshold = self._thresholds[key]
            ann = annotations[key]

            for channel_idx in range(ann.shape[1]):
                if (ann[:, channel_idx] > threshold).any():
                    peaks, peak_properties = scipy.signal.find_peaks(
                        ann[:, channel_idx],
                        height=threshold,
                        distance=min_separation_samples,
                    )
                    for peak, confidence in zip(peaks, peak_properties["peak_heights"]):
                        self._picks_per_channel_idx[(key, channel_idx)].append(
                            sbu.DASPick(
                                time=self._translate_coords(
                                    out_coords.sample + peak, "time"
                                ),
                                channel=self._translate_coords(
                                    out_coords.channel + channel_idx,
                                    self.channel_coord_name,
                                ),
                                confidence=confidence,
                                phase=key,
                            )
                        )

    def _translate_coords(self, idx: float, coord_name: str) -> float | np.datetime64:
        coord = self._output_coords[coord_name]
        v0 = coord.get_value(int(idx))
        v1 = coord.get_value(min(int(idx) + 1, len(coord) - 1))

        return v0 + (v1 - v0) * (idx - int(idx))

    def finalize(self) -> None:
        min_separation = np.timedelta64(int(self.min_time_separation * 1e9), "ns")
        for (phase, _), picks in self._picks_per_channel_idx.items():
            sorted_by_time = sorted(picks, key=attrgetter("time"))
            sorted_by_confidence = sorted(
                picks, key=attrgetter("confidence"), reverse=True
            )

            for pick in sorted_by_confidence:
                idx1 = bisect.bisect_left(
                    sorted_by_time,
                    pick.time - min_separation,
                    lo=0,
                    hi=len(sorted_by_time),
                    key=attrgetter("time"),
                )
                idx2 = bisect.bisect_right(
                    sorted_by_time,
                    pick.time + min_separation,
                    lo=0,
                    hi=len(sorted_by_time),
                    key=attrgetter("time"),
                )

                if any(
                    other.confidence > pick.confidence
                    for other in sorted_by_time[idx1:idx2]
                ):
                    continue

                self._picks[phase].append(pick)

    def get_results_dict(self) -> dict[str, Any]:
        return self._picks

    def get_results_dataframe(self) -> pd.DataFrame:
        return pd.concat([pd.DataFrame(key) for key in self._picks.values()])


class WriterBuffer:
    """
    A buffer to handle intersections between overlapping output data. The buffer expects data in patches of equal size.
    The patch order needs to be left to right (samples), top to bottom (channels), i.e., first all samples for a range
    of channels need to be processed before the next row can be processed.

    The buffer keeps up to two rows in memory and writes slices along the sample axis once they are fully predicted.
    """

    def __init__(
        self,
        data: VirtualTransformedDataArray,
        stacking: str,
        output_shape: tuple[int, int],
    ):
        self.stacking = stacking
        self.output_shape = output_shape
        self._dtype = data.dtype
        self._last_channel: int | None = None  # First channel of the channel buffer
        self._previous_sample: int | None = (
            None  # First sample of the previous sample buffer
        )
        self._current_sample: int | None = (
            None  # First sample of the current sample buffer
        )

        self._channel_buffer = None  # Buffer along the current line
        self._previous_sample_buffer = None  # Line above, always complete
        self._current_sample_buffer = None  # Current line, potentially incomplete

        if self.stacking == "avg":
            self._channel_buffer_count = None
            self._previous_sample_buffer_count = None
            self._current_sample_buffer_count = None

    @property
    def stacking(self) -> str:
        return self._stacking

    @stacking.setter
    def stacking(self, value: str):
        if value not in ["avg", "max"]:
            raise ValueError("Stacking must be either 'avg' or 'max'.")
        self._stacking = value

    def add_data(
        self, data: np.ndarray, out_coords: PatchCoordinate
    ) -> Optional[tuple[np.ndarray, PatchCoordinate]]:
        if self._last_channel is not None and self._last_channel >= out_coords.channel:
            raise ValueError(
                "Channel indices within a row must be strictly increasing."
            )
        if (
            self._current_sample is not None
            and self._current_sample != out_coords.sample
        ):
            raise ValueError("Can't start new row without finishing the previous one.")

        output = None

        if self._last_channel is None:
            if out_coords.channel > 0:
                raise ValueError("Each row needs to start at channel 0.")
            self._channel_buffer = data.copy()
            if self.stacking == "avg":
                self._channel_buffer_count = np.ones_like(
                    self._channel_buffer, dtype=np.uint32
                )
            finalized_segment = None
            self._last_channel = 0
        else:
            # Update channel buffer
            boundary_channels = (
                out_coords.channel_int - self._last_channel
            )  # channels that are only in one of the data
            tmp_buffer = np.empty(
                (out_coords.w_sample, out_coords.w_channel), dtype=self._dtype
            )

            finalized_segment = self._channel_buffer[:, :boundary_channels]
            if self.stacking == "avg":
                finalized_segment_count = self._channel_buffer_count[
                    :, :boundary_channels
                ]

            tmp_buffer[:, -boundary_channels:] = data[
                :, -boundary_channels:
            ]  # Right - new data
            # Middle - manage overlap
            if self.stacking == "max":
                tmp_buffer[:, :-boundary_channels] = np.maximum(
                    self._channel_buffer[:, boundary_channels:],
                    data[:, :-boundary_channels],
                )
            elif self.stacking == "avg":
                tmp_buffer[:, :-boundary_channels] = (
                    self._channel_buffer[:, boundary_channels:]
                    + data[:, :-boundary_channels]
                )
                tmp_buffer_count = np.zeros_like(tmp_buffer, dtype=np.uint32)
                tmp_buffer_count[:, :-boundary_channels] = self._channel_buffer_count[
                    :, boundary_channels:
                ]
                tmp_buffer_count += 1
                self._channel_buffer_count = tmp_buffer_count

            self._channel_buffer = tmp_buffer

        # Row completed - Extend finalized segment
        if out_coords.channel + out_coords.w_channel == self.output_shape[1]:
            if (
                finalized_segment is None
            ):  # First entry in the row, i.e., row has only a single patch
                finalized_segment = self._channel_buffer
                if self.stacking == "avg":
                    finalized_segment_count = self._channel_buffer_count
            else:
                finalized_segment = np.concatenate(
                    [finalized_segment, self._channel_buffer], axis=1
                )
                if self.stacking == "avg":
                    finalized_segment_count = np.concatenate(
                        [finalized_segment_count, self._channel_buffer_count], axis=1
                    )

        # Update sample buffer - Note that horizontal overlap is already handled
        if self._current_sample is None:  # First entry in the row
            self._current_sample_buffer = np.empty(
                (out_coords.w_sample, self.output_shape[1]), dtype=self._dtype
            )
            if self.stacking == "avg":
                self._current_sample_buffer_count = np.zeros_like(
                    self._current_sample_buffer, dtype=np.uint32
                )
            self._current_sample = out_coords.sample_int

        if finalized_segment is not None:
            self._current_sample_buffer[
                :, self._last_channel : self._last_channel + finalized_segment.shape[1]
            ] = finalized_segment
            if self.stacking == "avg":
                self._current_sample_buffer_count[
                    :,
                    self._last_channel : self._last_channel
                    + finalized_segment.shape[1],
                ] = finalized_segment_count

        self._last_channel = out_coords.channel_int

        if (
            out_coords.channel + out_coords.w_channel == self.output_shape[1]
        ):  # Row completed - Write sample buffer
            if self._previous_sample_buffer is None:
                finalized_segment = None
                finalized_segment_count = None
                finalized_sample = None
                self._previous_sample_buffer = self._current_sample_buffer
                if self.stacking == "avg":
                    self._previous_sample_buffer_count = (
                        self._current_sample_buffer_count
                    )
                self._previous_sample = self._current_sample
            else:
                boundary_samples = self._current_sample - self._previous_sample
                finalized_segment = self._previous_sample_buffer[:boundary_samples]
                if self.stacking == "avg":
                    finalized_segment_count = self._previous_sample_buffer_count[
                        :boundary_samples
                    ]

                tmp_buffer = np.empty(
                    (out_coords.w_sample, self.output_shape[1]), dtype=self._dtype
                )
                tmp_buffer[-boundary_samples:] = self._current_sample_buffer[
                    -boundary_samples:
                ]  # Bottom - New data

                # Middle - manage overlap
                if self.stacking == "max":
                    tmp_buffer[:-boundary_samples] = np.maximum(
                        self._previous_sample_buffer[boundary_samples:],
                        self._current_sample_buffer[:-boundary_samples],
                    )
                elif self.stacking == "avg":
                    tmp_buffer[:-boundary_samples] = (
                        self._previous_sample_buffer[boundary_samples:]
                        + self._current_sample_buffer[:-boundary_samples]
                    )
                    tmp_buffer_count = np.zeros_like(tmp_buffer, dtype=np.uint32)
                    tmp_buffer_count[:-boundary_samples] = (
                        self._previous_sample_buffer_count[boundary_samples:]
                    )
                    tmp_buffer_count += self._current_sample_buffer_count
                    self._previous_sample_buffer_count = tmp_buffer_count

                finalized_sample = self._previous_sample
                self._previous_sample_buffer = tmp_buffer
                self._previous_sample = self._current_sample

            # Clear buffers for the current row
            self._current_sample_buffer = None
            self._current_sample_buffer_count = None
            self._current_sample = None
            self._channel_buffer = None
            self._channel_buffer_count = None
            self._last_channel = None

            if (
                out_coords.sample + out_coords.w_sample == self.output_shape[0]
            ):  # Last row
                if finalized_segment is None:
                    finalized_segment = self._previous_sample_buffer
                    if self.stacking == "avg":
                        finalized_segment_count = self._previous_sample_buffer_count
                else:
                    finalized_segment = np.concatenate(
                        [finalized_segment, self._previous_sample_buffer], axis=0
                    )
                    if self.stacking == "avg":
                        finalized_segment_count = np.concatenate(
                            [
                                finalized_segment_count,
                                self._previous_sample_buffer_count,
                            ],
                            axis=0,
                        )

                self._previous_sample_buffer = None
                self._previous_sample_buffer_count = None

            if finalized_segment is not None:
                if self.stacking == "avg":
                    finalized_segment = finalized_segment / finalized_segment_count
                else:
                    # Copy to avoid passing out pointers to a view of a large array
                    finalized_segment = finalized_segment.copy()

                if finalized_sample is None:
                    finalized_sample = 0

                output = (
                    finalized_segment,
                    PatchCoordinate(
                        finalized_sample,
                        0,
                        finalized_segment.shape[0],
                        finalized_segment.shape[1],
                    ),
                )

        return output

    def finalize(self) -> None:
        if any(
            buffer is not None
            for buffer in [
                self._channel_buffer,
                self._channel_buffer_count,
                self._current_sample_buffer,
                self._current_sample_buffer_count,
                self._previous_sample_buffer,
                self._previous_sample_buffer_count,
            ]
        ):
            raise ValueError("Incomplete return data.")


class WriterCallback(DASAnnotateCallback):
    """
    Writes the raw predictions of the model to disk. The callback implements streaming processing to avoid excessive
    memory usage, while ensuring correct splicing at the overlaps between adjacent patches.

    The output writing relies on the
    `xdas DataArrayWriter <https://xdas.readthedocs.io/en/latest/_autosummary/xdas.processing.DataArrayWriter.html>`_ .
    This means that the output will be written in multiple files using one output folder per annotation key.
    To load the files for key ``x`` use ``xdas.open_mfdataarray("output_path/x/*")``. Note that the time coordinate will
    have minor discontinuities due to the chunked writing. These can be fixed by calling
    ``data.coords["time"] = data.coords["time"].simplify(tolerance=np.timedelta64(1, "us"))``.
    """

    def __init__(self, output_path: Path | str, stacking: str = "avg"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=False, exist_ok=True)
        if stacking not in ["avg", "max"]:
            raise ValueError("Stacking must be either 'avg' or 'max'.")
        self.stacking = stacking

        self._writer_buffers = None
        self._data_writers = None
        self._output_coords = None
        self._annotate_keys = None
        self._sample_buffer = None
        self._channel_buffer = None

    def setup(
        self,
        data: VirtualTransformedDataArray,
        patching_structure: PatchingStructure,
        annotate_keys: list[str],
    ) -> None:
        output_shape, self._output_coords = DASModel.calc_output_shape_and_coordinates(
            data, patching_structure
        )
        self.channel_coord_name = data.channel_coord_name

        self._annotate_keys = annotate_keys
        self._data_writers = {}
        self._writer_buffers = {}
        for key in annotate_keys:
            annotation_path = self.output_path / key
            annotation_path.mkdir(parents=False, exist_ok=False)
            self._data_writers[key] = xdas.processing.DataArrayWriter(annotation_path)
            self._writer_buffers[key] = WriterBuffer(
                data, stacking=self.stacking, output_shape=output_shape
            )

    def finalize(self) -> None:
        for writer in self._data_writers.values():
            writer.result()  # Closes writer
        self._data_writers = None

    def handle_patch(
        self,
        annotations: dict[str, np.ndarray],
        in_coords: PatchCoordinate,
        out_coords: PatchCoordinate,
    ) -> None:
        for key in self._annotate_keys:
            segment = self._writer_buffers[key].add_data(annotations[key], out_coords)
            if segment is not None:
                seg_annotation, seg_coords = segment
                self._data_writers[key].write(
                    xdas.DataArray(
                        data=seg_annotation,
                        coords={
                            "time": self._output_coords["time"][
                                seg_coords.sample_int : seg_coords.sample_int
                                + seg_coords.w_sample
                            ],
                            self.channel_coord_name: self._output_coords[
                                self.channel_coord_name
                            ][
                                seg_coords.channel_int : seg_coords.channel_int
                                + seg_coords.w_channel
                            ],
                        },
                    )
                )

    def get_results_dict(self) -> dict[str, Any]:
        return {key: self.output_path / key for key in self._annotate_keys}


class InMemoryCollectionCallback(DASAnnotateCallback):
    """
    Collects the raw predictions of the model in memory and splices the DAS array back together from the individual
    patches. To avoid memory overflows, this callback should only be used for small datasets.
    """

    def __init__(self, stacking: str = "avg"):
        self.annotations = None
        self.stacking = stacking

        self._annotate_keys = None
        self._output_coords = None
        self._write_buffers = None
        self._outputs = None

    def setup(
        self,
        data: VirtualTransformedDataArray,
        patching_structure: PatchingStructure,
        annotate_keys: list[str],
    ) -> None:
        output_shape, self._output_coords = DASModel.calc_output_shape_and_coordinates(
            data, patching_structure
        )
        self._write_buffers = {
            key: WriterBuffer(data, stacking=self.stacking, output_shape=output_shape)
            for key in annotate_keys
        }
        self._outputs = {
            key: np.empty(output_shape, dtype=data.dtype) for key in annotate_keys
        }
        self._annotate_keys = annotate_keys

    def handle_patch(
        self,
        annotations: dict[str, np.ndarray],
        in_coords: PatchCoordinate,
        out_coords: PatchCoordinate,
    ) -> None:
        for key in self._annotate_keys:
            segment = self._write_buffers[key].add_data(annotations[key], out_coords)
            if segment is not None:
                seg_annotation, seg_coords = segment
                self._outputs[key][
                    seg_coords.sample_int : seg_coords.sample_int + seg_coords.w_sample,
                    seg_coords.channel_int : seg_coords.channel_int
                    + seg_coords.w_channel,
                ] = seg_annotation

    def finalize(self) -> None:
        self.annotations = {}
        for key in self._annotate_keys:
            self.annotations[key] = DataArray(
                data=self._outputs[key], coords=self._output_coords
            )

    def get_results_dict(self) -> dict[str, Any]:
        return self.annotations


class FKFilter(nn.Module):
    """
    An F-k filter implemented in PyTorch. The filter processes batched data, i.e., the input format should be
    (batch, samples, channels).

    :param dx: Channel spacing in space
    :param dt: Sample spacing in time
    :param v_min: Minimum velocity to be considered in the filter. If None, no filtering is applied.
    :param v_max: Maximum velocity to be considered in the filter. If None, no filtering is applied.
    :param mode: Either "pass" or "reject". If "pass" all velocities between v_min and v_max are retained. If "reject",
                 all frequencies outside this band.
    """

    def __init__(
        self,
        dt: float,
        dx: float,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        mode: str = "pass",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.dx = dx
        self.v_min = v_min
        self.v_max = v_max
        self.mode = mode

        if mode not in ["pass", "reject"]:
            raise ValueError("mode must be 'pass' or 'reject'")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        _, nt, nx = data.shape

        Fk = torch.fft.fft2(data)

        f = torch.fft.fftfreq(nt, self.dt)  # 1/s
        k = torch.fft.fftfreq(nx, self.dx)  # 1/m

        F, K = torch.meshgrid(f, k, indexing="ij")

        c = torch.abs(F / (K + 1e-10))

        mask = torch.ones(Fk.shape[1:], dtype=torch.bool)
        if self.v_min is not None:
            mask = mask & (c >= self.v_min)
        if self.v_max is not None:
            mask = mask & (c < self.v_max)

        if self.mode == "reject":
            mask = ~mask

        Fk_filtered = Fk * mask

        return torch.fft.ifft2(Fk_filtered).real


class DASModel(SeisBenchModel, ABC):
    """
    This is the base class for all models processing DAS data.

    .. admonition:: Hint

        If you are an end-user looking to apply pretrained models, you most likely won't interact with this class
        directly. Instead, you will use classes inheriting from this class and their :py:func:`annotate` and
        :py:func:`classify` functions. If you aim to develop your own model, you should inherit from this class and
        have a look at the details below.

    .. admonition:: Hint

        When calling ``annotate`` or ``classify``, the model can perform automatic resampling along both axis. This
        ensures that the model can be flexibly applied to data of different sampling rates and channel spacings.
        However, as models are typically stable with respect to small changes in sampling rate and channel spacing,
        this class allows for a range of sampling rates and channel spacings to be specified. When called on data that
        does not fall into this ratio, the model will search for the smallest set of integers for upsampling and
        downsampling. The resampling is done using ``scipy.signal.resample_poly``. To get the exact resampling ratio
        for a particular input array, check the function :py:func:`get_resample_ratio`.

    :param patching_structure: The structure of the patches to cut for annotation. If None, the function
                               :py:func:`get_patching_structure` needs to be implemented, allowing to dynamically adjust
                               the patching structure to the input data.
    :param dt_range: Admissible range for the time step of data to be processed. This value is only taken into account
                     for the execution of the ``annotate``/``classify`` functions. See the above hint on the resampling
                     behavior. Values are in seconds.
    :param dx_range: Same as ``dt_range`` but along the channel axis. Values are in meters.
    :param buffer_queue_size: Maximum number of chunks to keep in the intermediate buffers.
    :param annotate_forward_kwargs: Additional keyword arguments to pass to the ``forward`` method of the model when
                                    running ``annotate``/``classify``.
    :param annotate_keys: List of annotation keys to read from the output.
    :param default_args: Default arguments for the optional keyword arguments of ``annotate``/``classify``.
    :param fk_filter_args: Arguments for the F-k filter. See :py:class:`FKFilter` for details.
    :param filter_samples: Filter to apply along the sample axis. See :py:class:`VirtualTransformedDataArray`
                           for details.
    """

    _annotate_args = {
        "batch_size": ("Batch size for the model", 2),
        "pbar": ("Show progress bar", True),
        "overlap_samples": (
            "Overlap between patches along the sample axis. "
            "Values between 0 and 1 are treated as fractions of the patch length. "
            "Values above 1 a sample counts.",
            0.5,
        ),
        "overlap_channels": (
            "Overlap between patches along the channel axis. "
            "Values between 0 and 1 are treated as fractions of the patch length. "
            "Values above 1 a channel counts.",
            0.5,
        ),
        "channel_coord_name": (
            "Name of the channel coordinate in the input data. "
            "The same will be used in the output. "
            "If None, the name is inferred automatically using a list of candidates.",
            None,
        ),
    }

    def __init__(
        self,
        dt_range: Optional[tuple[float, float]] = None,
        dx_range: Optional[tuple[float, float]] = None,
        patching_structure: Optional[PatchingStructure] = None,
        buffer_queue_size: int = 8,
        annotate_forward_kwargs: Optional[dict[str, Any]] = None,
        annotate_keys: Optional[list[str]] = None,
        default_args: Optional[dict[str, Any]] = None,
        fk_filter_args: Optional[dict[str, Any]] = None,
        filter_samples: Optional[tuple[str, dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dt_range = dt_range
        self.dx_range = dx_range
        self.patching_structure = patching_structure
        self.buffer_queue_size = buffer_queue_size

        if annotate_forward_kwargs is None:
            annotate_forward_kwargs = {}
        self.annotate_forward_kwargs = annotate_forward_kwargs

        if annotate_keys is None:
            annotate_keys = []
        self.annotate_keys = annotate_keys

        if default_args is None:
            default_args = {}
        self.default_args = default_args

        self.fk_filter_args = fk_filter_args
        self.filter_samples = filter_samples

    def get_model_args(self):
        # This function does not export technical arguments that are usually set in the subclass.
        model_args = super().get_model_args()
        model_args = {
            **model_args,
            **{
                "buffer_queue_size": self.buffer_queue_size,
            },
        }
        return model_args

    def get_patching_structure(
        self, data_shape: tuple[float, float], argdict: dict[str, Any]
    ) -> PatchingStructure:
        """
        To enable dynamic window sizes, depending on the shape of the input record, this function can be overwritten.
        By default, returns the predefined patching structure. In addition, this function allows to overwrite the
        overlap dynamically.

        The ``data_shape`` is provided for adaptive models. Note that the data shape can have float coordinates due to
        in-memory resampling of the data. The actual output shape can only be inferred once the patching structure has
        been defined, as the number of truncated samples depends on the patching structure. Therefore, models should be
        flexible towards the case of slightly smaller data shapes than the theoretical one.
        """
        if self.patching_structure is None:
            raise ValueError(
                "Patching structure needs to be defined. "
                "This can happen either by setting it in the constructor or by implementing "
                "_get_patch_structure in a subclass."
            )
        patching_structure = copy.deepcopy(self.patching_structure)

        if patching_structure.overlap_samples is None:
            overlap_samples = self._argdict_get_with_default(argdict, "overlap_samples")
            if overlap_samples < 1:
                overlap_samples = int(overlap_samples * patching_structure.in_samples)
            patching_structure.overlap_samples = overlap_samples
        if patching_structure.overlap_channels is None:
            overlap_channels = self._argdict_get_with_default(
                argdict, "overlap_channels"
            )
            if overlap_channels < 1:
                overlap_channels = int(
                    overlap_channels * patching_structure.in_channels
                )
            patching_structure.overlap_channels = overlap_channels

        return patching_structure

    def annotate(self, *args, **kwargs) -> None:
        asyncio.run(self.annotate_async(*args, **kwargs))

    def classify(self, *args, **kwargs) -> sbu.ClassifyOutput:
        return asyncio.run(self.classify_async(*args, **kwargs))

    async def annotate_async(
        self, data: DataArray, callback: DASAnnotateCallback, **kwargs
    ) -> None:
        self._verify_argdict(kwargs)
        # Kwargs overwrite default args
        argdict = self.default_args.copy()
        argdict.update(kwargs)

        channel_coord_name = self._argdict_get_with_default(
            argdict, "channel_coord_name"
        )
        resample_samples, resample_channels = self.get_resample_ratios(
            data, channel_coord_name
        )

        patching_structure = self.get_patching_structure(
            VirtualTransformedDataArray.estimate_theoretical_output_shape(
                data, resample_samples, resample_channels
            ),
            argdict,
        )

        virtual_data = VirtualTransformedDataArray(
            data,
            patching_structure,
            resample_samples=resample_samples,
            resample_channels=resample_channels,
            filter_samples=self.filter_samples,
            force_dtype=torch_dtype_to_numpy(self.dtype),
            channel_coord_name=self._argdict_get_with_default(
                argdict, "channel_coord_name"
            ),
        )

        if self.fk_filter_args is None:
            fk_filter = None
        else:
            fk_filter = FKFilter(
                dt=virtual_data.dt, dx=virtual_data.dx, **self.fk_filter_args
            )

        callback.setup(virtual_data, patching_structure, self.annotate_keys)

        queue_chunks = asyncio.Queue(self.buffer_queue_size)
        queue_output = asyncio.Queue(self.buffer_queue_size)

        loader_task = self._slice_and_preprocess(virtual_data, queue_chunks)
        processor_task = self._process_patch(
            queue_chunks, queue_output, patching_structure, fk_filter, argdict
        )
        postprocessor_task = self._postprocess_patch(
            queue_output, callback, len(virtual_data), argdict
        )

        await asyncio.gather(loader_task, processor_task, postprocessor_task)
        callback.finalize()

    async def _slice_and_preprocess(
        self,
        data: VirtualTransformedDataArray,
        queue_out: asyncio.Queue,
    ) -> None:
        async for patch, coord in data:
            await queue_out.put((patch, coord))

        await queue_out.put(None)

    async def _process_patch(
        self,
        queue_in: asyncio.Queue,
        queue_out: asyncio.Queue,
        patching_structure: PatchingStructure,
        fk_filter: Optional[FKFilter],
        argdict: dict[str, Any],
    ) -> None:
        batch_size = self._argdict_get_with_default(argdict, "batch_size")

        buffer = []
        while True:
            item = await queue_in.get()
            if item is None:
                if len(buffer) > 0:
                    annotation_items = await asyncio.to_thread(
                        self._predict_buffer,
                        buffer,
                        patching_structure,
                        fk_filter,
                        argdict,
                    )
                    for annotation_item in annotation_items:
                        await queue_out.put(annotation_item)
                break

            buffer.append(item)

            if len(buffer) == batch_size:
                annotation_items = await asyncio.to_thread(
                    self._predict_buffer, buffer, patching_structure, fk_filter, argdict
                )
                for annotation_item in annotation_items:
                    await queue_out.put(annotation_item)
                buffer = []

            queue_in.task_done()
        await queue_out.put(None)

    def _predict_buffer(
        self,
        buffer: list[tuple[np.ndarray, PatchCoordinate]],
        patching_structure: PatchingStructure,
        fk_filter: Optional[FKFilter],
        argdict: dict[str, Any],
    ) -> list[tuple[dict[str, np.ndarray], PatchCoordinate]]:
        data = [data for data, _ in buffer]
        coords = [coord for _, coord in buffer]

        data = torch.tensor(
            np.stack(data, axis=0), device=self.device, dtype=self.dtype
        )
        train_mode = self.training
        try:
            self.eval()
            with torch.no_grad():
                if fk_filter is not None:
                    data = fk_filter(data)
                annotations = self(
                    data, **self.annotate_forward_kwargs, argdict=argdict
                )
        finally:
            if train_mode:
                self.train()

        output = []
        for i, in_coord in enumerate(coords):
            out_coord = self._transform_patch_coordinates(in_coord, patching_structure)
            sample_annotations = {
                key: annotations[key][i].cpu().numpy() for key in self.annotate_keys
            }
            output.append((sample_annotations, in_coord, out_coord))

        return output

    def get_resample_ratios(
        self, data: DataArray, channel_coord_name: Optional[str]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Estimates integer ratios for resampling along the sample and channel axes to fall into the predefined ratios.
        """
        if channel_coord_name is None:
            channel_coord_name = VirtualTransformedDataArray.guess_channel_coord_name(
                data
            )

        if self.dt_range is None:
            resample_samples = (1, 1)
        else:
            time_coords = data.coords["time"]
            data_dt = (time_coords[1] - time_coords[0]) / np.timedelta64(1, "s")
            resample_samples = self._find_range(data_dt, *self.dt_range)

        if self.dx_range is None:
            resample_channels = (1, 1)
        else:
            channel_coords = data.coords[channel_coord_name]
            data_dx = float((channel_coords[1] - channel_coords[0]).values)
            resample_channels = self._find_range(data_dx, *self.dx_range)

        return resample_samples, resample_channels

    @staticmethod
    def _find_range(
        v: float, vmin: float, vmax: float, limit: int = 10
    ) -> tuple[int, int]:
        for up in range(1, limit + 1):
            for down in range(1, limit + 1):
                if vmin <= (v * down) / up <= vmax:
                    break
            else:
                continue
            break
        else:
            raise ValueError("No integer ratio found for resampling data.")

        return up, down

    @staticmethod
    def _transform_patch_coordinates(
        coord: PatchCoordinate, patching_structure: PatchingStructure
    ) -> PatchCoordinate:
        """
        Transform between coordinates in the input array and coordinates in the output array.
        """
        if (
            coord.w_sample != patching_structure.in_samples
            or coord.w_channel != patching_structure.in_channels
        ):
            raise ValueError("Patch coordinates do not match patching structure.")

        scale_samples = patching_structure.out_samples / (
            patching_structure.range_samples[1] - patching_structure.range_samples[0]
        )
        scale_channels = patching_structure.out_channels / (
            patching_structure.range_channels[1] - patching_structure.range_channels[0]
        )
        return PatchCoordinate(
            sample=coord.sample * scale_samples,
            channel=coord.channel * scale_channels,
            w_sample=patching_structure.out_samples,
            w_channel=patching_structure.out_channels,
        )

    @staticmethod
    def calc_output_shape_and_coordinates(
        da: VirtualTransformedDataArray, patching_structure: PatchingStructure
    ) -> tuple[tuple[int, int], dict[str, xdas.InterpCoordinate]]:
        """
        Calculates the shape and coordinate axis of the output array after processing with the given patching structure.
        In case the output shape would be fractional, an extra sample is added to the output array along the
        corresponding axis.
        """
        scale_samples = patching_structure.out_samples / (
            patching_structure.range_samples[1] - patching_structure.range_samples[0]
        )
        scale_channels = patching_structure.out_channels / (
            patching_structure.range_channels[1] - patching_structure.range_channels[0]
        )
        truncated_samples = (
            patching_structure.in_samples
            - patching_structure.range_samples[1]
            + patching_structure.range_samples[0]
        )
        truncated_channels = (
            patching_structure.in_channels
            - patching_structure.range_channels[1]
            + patching_structure.range_channels[0]
        )

        output_shape = (
            np.ceil((da.shape[0] - truncated_samples) * scale_samples).astype(int),
            np.ceil((da.shape[1] - truncated_channels) * scale_channels).astype(int),
        )
        output_coords = {
            "time": xdas.InterpCoordinate(
                data={
                    "tie_indices": [0, output_shape[0] - 1],
                    "tie_values": [
                        da.coords["time"][patching_structure.range_samples[0]].data,
                        da.coords["time"][
                            -(
                                patching_structure.in_samples
                                - patching_structure.range_samples[1]
                            )
                            - 1
                        ].data,
                    ],
                },
                dim="time",
            ),
            da.channel_coord_name: xdas.InterpCoordinate(
                data={
                    "tie_indices": [0, output_shape[1] - 1],
                    "tie_values": [
                        da.coords[da.channel_coord_name][
                            patching_structure.range_channels[0]
                        ].data,
                        da.coords[da.channel_coord_name][
                            -(
                                patching_structure.in_channels
                                - patching_structure.range_channels[1]
                            )
                            - 1
                        ].data,
                    ],
                },
                dim=da.channel_coord_name,
            ),
        }

        return output_shape, output_coords

    async def _postprocess_patch(
        self,
        queue_in: asyncio.Queue,
        callback: DASAnnotateCallback,
        n_patches: int,
        argdict: dict[str, Any],
    ) -> None:
        if self._argdict_get_with_default(argdict, "pbar"):
            pbar = tqdm(total=n_patches)
        else:
            pbar = None

        while True:
            item = await queue_in.get()
            if item is None:
                break
            await asyncio.to_thread(callback.handle_patch, *item)
            if pbar is not None:
                pbar.update(1)
            queue_in.task_done()
        if pbar is not None:
            pbar.close()

    async def classify_async(self, data: DataArray, **kwargs) -> sbu.ClassifyOutput:
        """
        The classify method is used to process the data and apply the default callback.
        The ``kwargs`` are split into two groups: those that are passed to the callback and those that are passed to the
        annotate method.
        """
        callback_cls = self.classify_callback

        callback_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(callback_cls).parameters
        }
        annotate_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in inspect.signature(callback_cls).parameters
        }

        callback = callback_cls(**callback_kwargs)

        await self.annotate_async(data, callback, **annotate_kwargs)
        return sbu.ClassifyOutput(self.name, **callback.get_results_dict())

    @property
    def classify_callback(self) -> Type[DASAnnotateCallback]:
        """
        Return the default callback for this model. For example, for picking models, this would be a DASPickingCallback.
        The class will then be instantiated and used to process the output of the annotate method.
        Constructor arguments will be extracted from the ``kwargs`` passed to ``classify``.
        """
        raise NotImplementedError()

    def _argdict_get_with_default(self, argdict, key):
        return argdict.get(key, self._annotate_args.get(key)[1])

    def _verify_argdict(self, argdict):
        for key in argdict.keys():
            if not any(
                re.fullmatch(pattern.replace("*", ".*"), key)
                for pattern in self._annotate_args.keys()
            ):
                seisbench.logger.warning(f"Unknown argument '{key}' will be ignored.")
