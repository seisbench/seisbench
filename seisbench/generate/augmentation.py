import numpy as np
import scipy.signal
import copy
import re

from abc import abstractmethod, ABC
from seisbench.util.ml import gaussian_pick
from seisbench import config
import seisbench


class FixedWindow:
    """
    A simple windower that returns fixed windows.
    In addition, the windower rewrites all metadata ending in "_sample" to point to the correct sample after window selection.
    Window start and length can be set either at initialization or separately in each call.
    The later is primarily intended for more complicated windowers inheriting from FixedWindow.
    :param p0: Start position of the trace.
    If p0 is negative, this will be treated as identifying a sample before the start of the trace.
    This is in contrast to standard list indexing with negative indices in Python, which counts items from the end of the list.
    Negative p0 is not possible with the strategy "fail".

    :param windowlen: Window length
    :param strategy: Strategy to mitigate insufficient data. Options are:

        - "fail": Raises a ValueError
        - "pad": Adds zero padding to the window
        - "move": Moves the start to the closest possible position to achieve sufficient trace length.
          The resulting trace will be aligned to the input trace on one of the ends,
          depending if parts before (left aligned) or after the trace (right aligned) were requested.
          Will fail if total trace length is shorter than requested window length.
        - "variable": Returns shorter length window, resulting in possibly varying window size.
          Might return empty window if requested window is completely outside target range.

    :param axis: Axis along which the window selection should be performed
    :param key: The keys for reading from and writing to the state dict.
        If key is a single string, the corresponding entry in state dict is modified.
        Otherwise, a 2-tuple is expected, with the first string indicating the key to read from and the second one the key to write to.
    """

    def __init__(self, p0=None, windowlen=None, strategy="fail", axis=-1, key="X"):
        self.p0 = p0
        self.windowlen = windowlen
        self.strategy = strategy
        self.axis = axis
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

        if self.strategy not in ["fail", "pad", "move", "variable"]:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Options are 'fail', 'pad', 'move', 'variable'"
            )

    def __call__(self, state_dict, p0=None, windowlen=None):
        p0, windowlen = self._validate_parameters(p0, windowlen)

        x, metadata = state_dict[self.key[0]]

        if self.key[0] != self.key[1]:
            # Ensure metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)

        left_padding = None
        p0_padding_offset = 0
        right_padding = None

        if p0 < 0 and self.strategy == "pad":
            left_pad_shape = list(x.shape)
            p0_padding_offset = max(-p0, 0)
            left_pad_shape[self.axis] = min(p0_padding_offset, windowlen)
            left_padding = np.zeros_like(x, shape=left_pad_shape)

            windowlen += p0  # Shorten target window len
            windowlen = max(0, windowlen)  # Windowlen needs to be positive
            p0 = 0

        if x.shape[self.axis] < p0 + windowlen:
            if self.strategy == "fail":
                raise ValueError(
                    f"Requested window length ({windowlen}) is longer than available length after p0 ({x.shape[self.axis] - p0})."
                )
            elif self.strategy == "pad":
                p0 = min(p0, x.shape[self.axis])
                old_windowlen = windowlen
                windowlen = x.shape[self.axis] - p0

                right_pad_shape = list(x.shape)
                right_pad_shape[self.axis] = old_windowlen - windowlen
                right_padding = np.zeros_like(x, shape=right_pad_shape)
            elif self.strategy == "move":
                p0 = x.shape[self.axis] - windowlen
                if p0 < 0:
                    raise ValueError(
                        f"Total trace length ({x.shape[self.axis]}) is shorter than requested window length ({windowlen})."
                    )
            elif self.strategy == "variable":
                p0 = min(p0, x.shape[self.axis])
                windowlen = x.shape[self.axis] - p0

        window = np.take(x, range(p0, p0 + windowlen), axis=self.axis)

        if left_padding is not None:
            window = np.concatenate([left_padding, window], axis=self.axis)
        if right_padding is not None:
            window = np.concatenate([window, right_padding], axis=self.axis)

        if left_padding is not None:
            p0 -= p0_padding_offset

        for key in metadata.keys():
            if key.endswith("_sample"):
                try:
                    metadata[key] = metadata[key] - p0
                except TypeError:
                    seisbench.logger.info(
                        f"Failed to do window adjustment for column {key} "
                        f"due to type error. "
                    )

        state_dict[self.key[1]] = (window, metadata)

    def _validate_parameters(self, p0, windowlen):
        if p0 is None:
            p0 = self.p0
        if windowlen is None:
            windowlen = self.windowlen

        if p0 is None:
            raise ValueError("Start position must be set in either init or call.")
        if windowlen is None:
            raise ValueError("Window length must be set in either init or call.")

        if p0 < 0 and self.strategy == "fail":
            raise ValueError("Negative indexing is not supported for strategy fail.")

        if p0 < 0 and self.strategy == "move":
            p0 = 0

        if p0 < 0 and self.strategy == "variable":
            windowlen += p0  # Shorten target window len
            windowlen = max(0, windowlen)  # Windowlen needs to be positive
            p0 = 0

        return p0, windowlen

    def __str__(self):
        return f"FixedWindow (p0={self.p0}, windowlen={self.windowlen}, strategy={self.strategy}, axis={self.axis})"


class SlidingWindow(FixedWindow):
    """
    Generates sliding windows and adds a new axis for windows as first axis.
    All metadata entries are converted to arrays of entries.
    Only complete windows are returned and a possible remainder is truncated.
    In particular, if the available data is shorter than the number of windows, an empty array is returned.

    :param timestep: Difference between two consecutive window starts
    :param windowlen: Length of the output window
    :param kwargs: All kwargs are passed directly to FixedWindow

    """

    def __init__(self, timestep, windowlen, **kwargs):
        self.timestep = timestep
        super().__init__(windowlen=windowlen, **kwargs)

    def __call__(self, state_dict):
        windowlen = self.windowlen
        timestep = self.timestep

        x, metadata = state_dict[self.key[0]]

        if self.key[0] != self.key[1]:
            # Ensure metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)

        n_windows = (x.shape[self.axis] - windowlen) // timestep + 1

        if n_windows == 0:
            target_shape = list(x.shape)
            target_shape[self.axis] = windowlen
            x_out = np.zeros_like(x, shape=[0] + target_shape)
            state_dict[self.key[1]] = (x_out, metadata)
            return

        window_outputs = []
        window_metadatas = []

        for i in range(n_windows):
            tmp_state_dict = {self.key[0]: (x, copy.deepcopy(metadata))}
            super().__call__(tmp_state_dict, p0=i * timestep)
            window_outputs.append(tmp_state_dict[self.key[1]][0])
            window_metadatas.append(tmp_state_dict[self.key[1]][1])

        x_out = np.stack(window_outputs, axis=0)
        collected_metadata = {}
        for key in window_metadatas[0].keys():
            collected_metadata[key] = np.array(
                [window_metadata[key] for window_metadata in window_metadatas]
            )
        state_dict[self.key[1]] = (x_out, collected_metadata)

    def __str__(self):
        return f"SlidingWindow (windowlen={self.windowlen}, timestep={self.timestep})"


class WindowAroundSample(FixedWindow):
    """
    Creates a window around a sample defined in the metadata.
    :param metadata_keys: Metadata key or list of metadata keys to use for window selection.
    The corresponding metadata entries are assumed to be in sample units.
    The generator will fail if for a sample all values are NaN.

    :param samples_before: The number of samples to include before the target sample.
    :param selection: Selection strategy in case multiple metadata keys are provided and have non-NaN values.
        Options are:
        - "first": use the first available key
        - "random": use uniform random selection among the keys
    :param kwargs: Parameters passed to the init method of FixedWindow.

    """

    def __init__(self, metadata_keys, samples_before=0, selection="first", **kwargs):
        if isinstance(metadata_keys, str):
            self.metadata_keys = [metadata_keys]
        else:
            self.metadata_keys = metadata_keys
        self.samples_before = samples_before
        self.selection = selection

        if selection not in ["first", "random"]:
            raise ValueError(f"Unknown selection strategy '{self.selection}'")

        super().__init__(**kwargs)

    def __call__(self, state_dict, windowlen=None):
        _, metadata = state_dict[self.key[0]]
        cand = [
            metadata[key] for key in self.metadata_keys if not np.isnan(metadata[key])
        ]

        if len(cand) == 0:
            raise ValueError("Found no possible candidate for window selection")

        if self.selection == "first":
            p0 = cand[0]
        elif self.selection == "random":
            p0 = np.random.choice(cand)
        else:
            raise ValueError(f"Unknown selection strategy '{self.selection}'")

        p0 -= self.samples_before

        super(WindowAroundSample, self).__call__(
            state_dict, p0=int(p0), windowlen=windowlen
        )

    def __str__(self):
        return f"WindowAroundSample (metadata_keys={self.metadata_keys}, samples_before={self.samples_before}, selection={self.selection})"


class RandomWindow(FixedWindow):
    """
    Selects a window within the range [low, high) randomly with a uniform distribution.
    If there are no candidates fulfilling the criteria, the window selection depends on the strategy.
    For "fail" or "move" a ValueError is raise.
    For "variable" only the part between low and high is returned. If high < low, this part will be empty.
    For "pad" the same as for "variable" will be returned, but padded to the correct length.
    The padding will be added randomly to both sides.

    :param low: The lowest allowed index for the start sample.
        The sample at this position can be included in the output.
    :param high: The highest allowed index for the end.
        The sample at position high can *not* be included in the output
    :param kwargs: Parameters passed to the init method of FixedWindow.
    """

    def __init__(self, low=None, high=None, windowlen=None, **kwargs):
        super().__init__(windowlen=windowlen, **kwargs)

        self.low = low
        self.high = high

        if high is not None:
            if low is None:
                low = 0  # Temporary value for validity check that is not stored

            if windowlen is not None:
                if high - low < windowlen and self.strategy in ["fail", "move"]:
                    raise ValueError(
                        "Difference between low and high must be at least window length."
                    )
            elif low >= high:
                raise ValueError("Low value needs to be smaller than high value.")

    def __call__(self, state_dict, windowlen=None):
        x, _ = state_dict[self.key[0]]
        _, windowlen = self._validate_parameters(0, windowlen)

        low, high = self.low, self.high
        if high is None:
            high = x.shape[self.axis]
        if low is None:
            low = 0

        if low > high - windowlen:
            if self.strategy == "pad":
                # Get larger interval around target window with random location
                # The parent FixedWindow will handle possible cases in which the waveform is too short
                p0 = np.random.randint(high - windowlen, low + 1)
                super().__call__(state_dict, p0=p0, windowlen=windowlen)
                # Get output and overwrite padded range with zeros
                x, metadata = state_dict[self.key[1]]

                def fill_range_with_zeros(a, b):
                    # Fills range of array with zeros
                    # Automatically selects the correct axis
                    axis = self.axis
                    if axis < 0:
                        axis = x.ndim + axis

                    ind = np.arange(a, b, dtype=int)
                    for _ in range(axis):
                        ind = np.expand_dims(ind, 0)
                    for _ in range(axis + 1, x.ndim):
                        ind = np.expand_dims(ind, -1)

                    np.put_along_axis(x, ind, 0, axis)

                fill_range_with_zeros(0, low - p0)
                fill_range_with_zeros(high - p0, x.shape[self.axis])
                return

            elif self.strategy == "variable":
                # Return interval between high and low
                p0 = low
                windowlen = max(0, high - low)
                super().__call__(state_dict, p0=p0, windowlen=windowlen)
                return

            else:
                raise ValueError("No possible candidates for random window selection.")

        p0 = np.random.randint(low, high - windowlen + 1)
        super().__call__(state_dict, p0=p0, windowlen=windowlen)

    def __str__(self):
        return f"RandomWindow (low={self.low}, high={self.high})"


class Normalize:
    """
    A normalization augmentation that allows demeaning, detrending and amplitude normalization (in this order).

    :param demean_axis: The axis (single axis or tuple) which should be jointly demeaned. None indicates no demeaning.
    :param detrend_axis: The axis along with detrending should be applied.
    :param amp_norm: The axis (single axis or tuple) which should be jointly amplitude normalized. None indicates no normalization.
    :param amp_norm_type: Type of amplitude normalization. Supported types:
        - "peak": division by the absolute peak of the trace
        - "std": division by the standard deviation of the trace
    :param eps: Epsilon value added in amplitude normalization to avoid division by zero
    :param key: The keys for reading from and writing to the state dict.
        If key is a single string, the corresponding entry in state dict is modified.
        Otherwise, a 2-tuple is expected, with the first string indicating the key to read from and the second one the key to write to.
    """

    def __init__(
        self,
        demean_axis=None,
        detrend_axis=None,
        amp_norm_axis=None,
        amp_norm_type="peak",
        eps=1e-10,
        key="X",
    ):
        self.demean_axis = demean_axis
        self.detrend_axis = detrend_axis
        self.amp_norm_axis = amp_norm_axis
        self.amp_norm_type = amp_norm_type
        self.eps = eps
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

        if self.amp_norm_type not in ["peak", "std"]:
            raise ValueError(
                f"Unrecognized amp_norm_type '{self.amp_norm_type}'. Available types are: 'peak', 'std'."
            )

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]

        x = self._demean(x)
        x = self._detrend(x)
        x = self._amp_norm(x)

        state_dict[self.key[1]] = (x, metadata)

    def _demean(self, x):
        if self.demean_axis is not None:
            x = x - np.mean(x, axis=self.demean_axis, keepdims=True)
        return x

    def _detrend(self, x):
        if self.detrend_axis is not None:
            x = scipy.signal.detrend(x, axis=self.detrend_axis)
        return x

    def _amp_norm(self, x):
        if self.amp_norm_axis is not None:
            if self.amp_norm_type == "peak":
                x = x / (np.amax(x, axis=self.amp_norm_axis, keepdims=True) + self.eps)
            elif self.amp_norm_type == "std":
                x = x / (np.std(x, axis=self.amp_norm_axis, keepdims=True) + self.eps)
        return x

    def __str__(self):
        desc = []
        if self.demean_axis is not None:
            desc.append(f"Demean (axis={self.demean_axis})")
        if self.detrend_axis is not None:
            desc.append(f"Detrend (axis={self.detrend_axis})")
        if self.amp_norm_axis is not None:
            desc.append(
                f"Amplitude normalization (type={self.amp_norm_type}, axis={self.amp_norm_axis})"
            )

        if desc:
            desc = ", ".join(desc)
        else:
            desc = "no normalizations"

        return f"Normalize ({desc})"


class Filter:
    """
    Implements a filter augmentation, closely based on scipy.signal.butter.
    Please refer to the scipy documentation for more detailed description of the parameters

    :param N: Filter order
    :param Wn: The critical frequency or frequencies
    :param btype: The filter type: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
    :param analog: When True, return an analog filter, otherwise a digital filter is returned.
    :param forward_backward: If true, filters once forward and once backward.
        This doubles the order of the filter and makes the filter zero-phase.
    :param axis: Axis along which the filter is applied.
    :param key: The keys for reading from and writing to the state dict.
        If key is a single string, the corresponding entry in state dict is modified.
        Otherwise, a 2-tuple is expected, with the first string indicating the key to read from and the second one the key to write to.

    """

    def __init__(
        self, N, Wn, btype="low", analog=False, forward_backward=False, axis=-1, key="X"
    ):
        self.forward_backward = forward_backward
        self.axis = axis
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self._filt_args = (N, Wn, btype, analog)

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        sampling_rate = metadata["trace_sampling_rate_hz"]

        sos = scipy.signal.butter(*self._filt_args, output="sos", fs=sampling_rate)
        if self.forward_backward:
            # Copy is required, because otherwise the stride of x is negative.T
            # This can break the pytorch collate function.
            x = scipy.signal.sosfiltfilt(sos, x, axis=self.axis).copy()
        else:
            x = scipy.signal.sosfilt(sos, x, axis=self.axis)

        state_dict[self.key[1]] = (x, metadata)

    def __str__(self):
        N, Wn, btype, analog = self._filt_args
        return (
            f"Filter ({btype}, order={N}, frequencies={Wn}, analog={analog}, "
            f"forward_backward={self.forward_backward}, axis={self.axis})"
        )


class FilterKeys:
    """
    Filters keys in the state dict.
    Can be used to remove keys from the output that can not be collated by pytorch or are not required anymore.
    Either included or excluded keys can be defined.

    :param include: Only these keys will be present in the output.
    :param exclude: All keys except these keys will be present in the output.

    """

    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

        if (self.include is None and self.exclude is None) or (
            self.include is not None and self.exclude is not None
        ):
            raise ValueError("Exactly one of include or exclude must be specified.")

    def __call__(self, state_dict):
        if self.exclude is not None:
            for key in self.exclude:
                del state_dict[key]

        elif self.include is not None:
            for key in set(state_dict.keys()) - set(self.include):
                del state_dict[key]

    def __str__(self):
        if self.exclude is not None:
            return f"Filter keys (excludes {', '.join(self.exclude)})"
        if self.include is not None:
            return f"Filter keys (includes {', '.join(self.include)})"


class ChangeDtype:
    """
    Copies the data while changing the data type to the provided one

    :param dtype: Target data type
    :param key: The keys for reading from and writing to the state dict.
        If key is a single string, the corresponding entry in state dict is modified.
        Otherwise, a 2-tuple is expected, with the first string indicating the key to read from and the second one the key to write to.
    """

    def __init__(self, dtype, key="X"):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.dtype = dtype

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]

        if self.key[0] != self.key[1]:
            metadata = copy.deepcopy(metadata)

        x = x.astype(self.dtype)
        state_dict[self.key[1]] = (x, metadata)

    def __str__(self):
        return f"ChangeDtype (dtype={self.dtype}, key={self.key})"


class SupervisedLabeller(ABC):
    """
    Supervised classification labels.
    Performs simple checks for standard supervised classification labels.

    :param label_type: The type of label either: 'multi_label', 'multi_class', 'binary'.
    :param dim: Dimension over which labelling will be applied.
    :param key: The keys for reading from and writing to the state dict.
                Expects a 2-tuple with the first string indicating the key to read from and the second one the key to
                write to. Defaults to ("X", "y")
    """

    def __init__(self, label_type, dim, key=("X", "y")):
        self.label_type = label_type
        self.dim = dim

        if not isinstance(key, tuple):
            raise TypeError("key needs to be a 2-tuple.")
        if not len(key) == 2:
            raise ValueError("key needs to have exactly length 2.")

        self.key = key

        if self.label_type not in ("multi_label", "multi_class", "binary"):
            raise ValueError(
                f"Unrecognized supervised classification label type '{self.label_type}'."
                f"Available types are: 'multi_label', 'multi_class', 'binary'."
            )

    @abstractmethod
    def label(self, X, metadata):
        # to be overwritten in subclasses
        return y

    @staticmethod
    def _swap_dimension_order(arr, current_dim, expected_dim):
        config_dim = tuple(current_dim.find(d) for d in (expected_dim))
        return np.transpose(arr, (config_dim))

    @staticmethod
    def _get_dimension_order_from_config(config, ndim):
        if ndim == 3:
            sample_dim = config["dimension_order"].find("N")
            channel_dim = config["dimension_order"].find("C")
            width_dim = config["dimension_order"].find("W")
        elif ndim == 2:
            sample_dim = None
            channel_dim = config["dimension_order"].find("C")
            width_dim = config["dimension_order"].find("W")
            channel_dim = 0 if channel_dim < width_dim else 1
            width_dim = int(not channel_dim)
        else:
            raise ValueError(
                f"Only computes dimension order given 3 dimensions (NCW), "
                f"or 2 dimensions (CW)."
            )

        return sample_dim, channel_dim, width_dim

    def _check_labels(self, y, metadata):
        if self.label_type == "multi_class" and self.label_method == "probabilistic":
            if (y.sum(self.dim) > 1).any():
                raise ValueError(
                    f"More than one label provided. For multi_class problems, only one label can be provided per input."
                )

        if self.label_type == "binary":
            if (y.sum(self.dim) > 1).any():
                raise ValueError(f"Binary labels should lie within 0-1 range.")

        for label in self.label_columns:
            if (isinstance(metadata[label], (int, np.integer))) and self.ndim == 3:
                raise ValueError(
                    f"Only provided single arrival in metadata {label} column to multiple windows. Check augmentation workflow."
                )

    def __call__(self, state_dict):
        X, metadata = state_dict[self.key[0]]
        self.ndim = len(X.shape)

        y = self.label(X, metadata)
        self._check_labels(y, metadata)
        state_dict[self.key[1]] = (y, copy.deepcopy(metadata))


# FIXME: Best place for this method? Common to both 'Labeller' child classes but too specific to but in SupervisedLabller parent (as deals with picks-only)
# Leaving here for now
def _auto_identify_picklabels(state_dict):
    return sorted(
        list(filter(re.compile("trace_.*_arrival_sample").match, state_dict.keys()))
    )


class ProbabilisticLabeller(SupervisedLabeller):
    """
    Create supervised labels from picks. The picks in example are represented
    probabilistically with a Gaussian
        \[
            X \sim \mathcal{N}(\mu,\,\sigma^{2})\,.
        \]
        and the noise class is automatically created as \[ \max \left(0, 1 - \sum_{n=1}^{c} x_{j} \right) \].

    All picks with NaN sample are treated as not present.

    :param label_columns: Specify the columns to use for pick labelling, defaults to None and columns are inferred from metadata
    :type label_columns: list, optional
    :param dim: Dimension over which labelling will be applied, defaults to 1
    :type dim: int, optional
    :param sigma: Variance of Gaussian representation in samples, defaults to 10
    :type sigma: int, optional
    """

    def __init__(
        self,
        label_columns=None,
        dim=1,
        sigma=10,
    ):
        self.label_method = "probabilistic"
        self.label_columns = label_columns
        self.sigma = sigma
        super().__init__(label_type="multi_class", dim=dim)

    def label(self, X, metadata):

        if not self.label_columns:
            self.label_columns = _auto_identify_picklabels(metadata)

        sample_dim, channel_dim, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        # Add Noise as class (class N + 1 where N is number of labels)
        self.labels = [label.split("_")[1] for label in self.label_columns]
        self.labels.append("Noise")

        if self.ndim == 2:
            y = np.zeros(shape=(len(self.label_columns) + 1, X.shape[width_dim]))
        elif self.ndim == 3:
            y = np.zeros(
                shape=(
                    X.shape[sample_dim],
                    len(self.label_columns) + 1,
                    X.shape[width_dim],
                )
            )

        # Construct pick labels
        for i, label in enumerate(self.label_columns):

            if isinstance(metadata[label], (int, np.integer, float)):
                # Handle single window case
                onset = metadata[label]
                y[i, :] = gaussian_pick(
                    onset=onset, length=X.shape[width_dim], sigma=self.sigma
                )
            else:
                # Handle multi-window case
                for j in range(X.shape[sample_dim]):
                    onset = metadata[label][j]
                    y[j, i, :] = gaussian_pick(
                        onset=onset, length=X.shape[width_dim], sigma=self.sigma
                    )

        y[np.isnan(y)] = 0  # Set non-present pick probabilites to 0

        y /= np.maximum(
            1, np.nansum(y, axis=channel_dim, keepdims=True)
        )  # Ensure total probability mass is at most 1

        # Construct noise label
        if self.ndim == 2:
            y[len(self.label_columns), :] = 1 - np.nansum(y, axis=channel_dim)
            y = self._swap_dimension_order(
                y,
                current_dim="CW",
                expected_dim=config["dimension_order"].replace("N", ""),
            )
        elif self.ndim == 3:
            y[:, len(self.label_columns), :] = 1 - np.nansum(y, axis=channel_dim)
            y = self._swap_dimension_order(
                y, current_dim="NCW", expected_dim=config["dimension_order"]
            )

        return y

    def __str__(self):
        return f"ProbabilisticLabeller (label_type={self.label_type}, dim={self.dim})"


class StandardLabeller(SupervisedLabeller):
    """
    Create supervised labels from picks. The entire example is labelled as a single class/pick.
    For cases where multiple picks overlap in the input window, a number of options can be specified:
      - 'label-first' Only use first pick as label example.
      - 'fixed-relevance' Use pick closest to centre point of window as example.
      - 'random' Use random choice as example label.

    :param label_columns: Specify the columns to use for pick labelling, defaults to None and columns are inferred from metadata
    :type label_columns: list, optional
    :param dim: Dimension over which labelling will be applied, defaults to 1
    :type dim: int, optional
    :param on_overlap: Method used to label when multiple picks present in window, defaults to "label-first"
    :type on_overlap: str, optional
    """

    def __init__(
        self,
        label_columns=None,
        dim=1,
        on_overlap="label-first",
    ):

        self.label_method = "standard"
        self.label_columns = label_columns
        self.on_overlap = on_overlap

        if self.on_overlap not in ("label-first", "fixed-relevance", "random"):
            raise ValueError(
                "Unexpected value for `on_overlap` argument. Accepted values are: "
                "`label-first`, `fixed-relevance`, `random`."
            )
        super().__init__(label_type="multi_class", dim=dim)

    def _get_pick_arrivals_as_array(self, metadata):
        """
        Convert picked arrivals to numpy array. Set arrivals outside
        window to null (NaN) values.
        """
        arrival_array = np.array(
            [metadata[col] for col in self.label_columns], dtype=float
        ).T
        arrival_array[arrival_array < 0] = np.nan
        nan_mask = np.isnan(arrival_array)
        return arrival_array, nan_mask

    def _label_first(self, row_id, arrival_array, nan_mask):
        """
        Label based on first arrival in time in window.
        If no arrivals present, label as noise (class 0)
        """
        arrival_array[nan_mask] = np.inf
        if np.isinf(arrival_array[row_id]).all():
            return 0
        else:
            first_arrival = np.nanargmin(arrival_array[row_id])
            return first_arrival + 1

    def _label_random(self, row_id, nan_mask):
        """
        Label by randomly choosing pick inside window.
        If no arrivals present, label as noise (class 0)
        """
        non_null_columns = np.argwhere(~nan_mask[row_id])
        if non_null_columns.any() > 0:
            return np.random.choice(non_null_columns.reshape(-1)) + 1
        else:
            return 0

    def _label_fixed_relavence(self, row_id, arrival_array, midpoint):
        """
        Label using closest pick to centre of window.
        If no arrivals present, label as noise (class 0)
        """
        if np.isnan(arrival_array[row_id]).all():
            return 0
        else:
            return np.nanargmin(abs(arrival_array[row_id] - midpoint)) + 1

    def label(self, X, metadata):

        if not self.label_columns:
            self.label_columns = _auto_identify_picklabels(metadata)

        sample_dim, _, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        # Add Noise as null class (class 0)
        self.labels = [label.split("_")[1] for label in self.label_columns]
        self.labels.insert(0, "Noise")

        # Construct pick labels
        y = np.zeros(shape=(X.shape[sample_dim], 1))

        arrival_array, nan_mask = self._get_pick_arrivals_as_array(metadata)
        n_arrivals = nan_mask.shape[1] - nan_mask.sum(axis=1)

        for row_id, arrivals in enumerate(arrival_array):
            # Label
            if n_arrivals[row_id] == 1:
                y[row_id] = np.nanargmin(arrivals) + 1
            # On potentially overlapping labels
            else:
                if self.on_overlap == "label-first":
                    y[row_id] = self._label_first(row_id, arrival_array, nan_mask)
                elif self.on_overlap == "random":
                    y[row_id] = self._label_random(row_id, nan_mask)
                elif self.on_overlap == "fixed-relevance":
                    midpoint = X.shape[width_dim] // 2
                    y[row_id] = self._label_fixed_relavence(
                        row_id, arrival_array, midpoint
                    )

        return y

    def __str__(self):
        return f"StandardLabeller (label_type={self.label_type}, dim={self.dim})"
