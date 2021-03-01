import numpy as np
import scipy.signal
import copy


class FixedWindow:
    def __init__(self, p0=None, windowlen=None, strategy="fail", axis=-1, key="X"):
        """
        A simple windower that returns fixed windows.
        In addition, the windower rewrites all metadata ending in "_sample" to point to the correct sample after window selection.
        Window start and length can be set either at initialization or separately in each call.
        The later is primarily intended for more complicated windowers inheriting from FixedWindow.
        :param p0: Start position of the trace
        :param windowlen: Window length
        :param strategy: Strategy to mitigate insufficient data. Options are:
        - "fail": Raises a ValueError
        - "pad": Adds zero padding to the right of the window
        - "move": Moves the start to the left to achieve sufficient trace length.
        The resulting trace will be aligned to the input trace on the right end.
        Will fail if total trace length is shorter than requested window length.
        - "variable": Returns shorter length window, resulting in possibly varying window size
        :param axis: Axis along which the window selection should be performed
        :param key: The keys for reading from and writing to the state dict.
        If key is a single string, the corresponding entry in state dict is modified.
        Otherwise, a 2-tuple is expected, with the first string indicating the key to read from and the second one the key to write to.
        """
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

        padding = None
        if x.shape[self.axis] < p0 + windowlen:
            if self.strategy == "fail":
                raise ValueError(
                    f"Requested window length ({windowlen}) is longer than available length after p0 ({x.shape[self.axis] - p0})."
                )
            elif self.strategy == "pad":
                p0 = min(p0, x.shape[self.axis])
                old_windowlen = windowlen
                windowlen = x.shape[self.axis] - p0
                pad_shape = list(x.shape)
                pad_shape[self.axis] = old_windowlen - windowlen
                padding = np.zeros_like(x, shape=pad_shape)
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

        if padding is not None:
            window = np.concatenate([window, padding], axis=self.axis)

        for key in metadata.keys():
            if key.endswith("_sample"):
                metadata[key] -= p0

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

        if p0 < 0:
            raise ValueError("Negative indexing is not supported.")

        return p0, windowlen

    def __str__(self):
        return f"FixedWindow (p0={self.p0}, windowlen={self.windowlen}, strategy={self.strategy}, axis={self.axis})"


class SlidingWindow(FixedWindow):
    def __init__(self, timestep, windowlen, **kwargs):
        """
        Generates sliding windows and adds a new axis for windows as first axis.
        All metadata entries are converted to arrays of entries.
        Only complete windows are returned and a possible remainder is truncated.
        In particular, if the available data is shorter than the number of windows, an empty array is returned.
        :param timestep: Difference between two consecutive window starts
        :param windowlen: Length of the output window
        :param kwargs: All kwargs are passed directly to FixedWindow
        """
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


class Normalize:
    def __init__(
        self,
        demean_axis=None,
        detrend_axis=None,
        amp_norm_axis=None,
        amp_norm_type="peak",
        key="X",
    ):
        """
        A normalization augmentation that allows demeaning, detrending and amplitude normalization (in this order).
        :param demean_axis: The axis (single axis or tuple) which should be jointly demeaned. None indicates no demeaning.
        :param detrend_axis: The axis along with detrending should be applied.
        :param amp_norm: The axis (single axis or tuple) which should be jointly amplitude normalized. None indicates no normalization.
        :param amp_norm_type: Type of amplitude normalization. Supported types:
        - "peak" - division by the absolute peak of the trace
        - "std" - division by the standard deviation of the trace
        :param key: The keys for reading from and writing to the state dict.
        If key is a single string, the corresponding entry in state dict is modified.
        Otherwise, a 2-tuple is expected, with the first string indicating the key to read from and the second one the key to write to.
        """
        self.demean_axis = demean_axis
        self.detrend_axis = detrend_axis
        self.amp_norm_axis = amp_norm_axis
        self.amp_norm_type = amp_norm_type
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
                x = x / np.amax(x, axis=self.amp_norm_axis, keepdims=True)
            elif self.amp_norm_type == "std":
                x = x / np.std(x, axis=self.amp_norm_axis, keepdims=True)
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
    def __init__(
        self, N, Wn, btype="low", analog=False, forward_backward=False, axis=-1, key="X"
    ):
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
        Otherwise, a 2-tuple is expected, with the first string indicating the key to read from and the second one the key to write to.
        """
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
    def __init__(self, include=None, exclude=None):
        """
        Filters keys in the state dict.
        Can be used to remove keys from the output that can not be collated by pytorch or are not required anymore.
        Either included or excluded keys can be defined.
        :param include: Only these keys will be present in the output.
        :param exclude: All keys except these keys will be present in the output.
        """
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
