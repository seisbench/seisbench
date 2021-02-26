import numpy as np
import scipy.signal


class SlidingWindow:
    def __init__(self, windowlen=600, timestep=200):
        self.windowlen = windowlen
        self.timestep = timestep

    def __call__(self, state_dict):
        windowlen = self.windowlen
        timestep = self.timestep

        n_windows = (state_dict["waveforms"].shape[1] - windowlen) // timestep
        X_windows = []
        y_windows = []

        for i in range(n_windows):
            # Data is transformed to [N, C, W]
            X_windows.append(
                state_dict["waveforms"].T[i * timestep : i * timestep + windowlen].T
            )
            y_windows.append(
                state_dict["waveforms"].T[i * timestep : i * timestep + windowlen].T
            )
        # Add sequential windows to state_dict
        state_dict["X"] = np.array(X_windows)
        state_dict["y"] = np.array(y_windows)

    def __str__(self):
        return f"SlidingWindow (windowlen={self.windowlen}, timestep={self.timestep})"


class FixedWindow:
    def __init__(
        self, p0=None, windowlen=None, strategy="fail", axis=-1, key=("waveforms", "X")
    ):
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

        x = state_dict[self.key[0]]

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

        for key in state_dict["metadata"].keys():
            if key.endswith("_sample"):
                state_dict["metadata"][key] -= p0

        state_dict[self.key[1]] = window

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
        return f"FixedWindow (p0={self.p0}, windowlen={self.windowlen})"


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
        It will also cast the input array to float if it is an int type to enable the normalization actions.
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
        x = state_dict[self.key[0]]
        if x.dtype.char in np.typecodes["AllInteger"]:
            # Cast int types to float to ensure all operations are mathematically possible
            x = x.astype(float)

        x = self._demean(x)
        x = self._detrend(x)
        x = self._amp_norm(x)

        state_dict[self.key[1]] = x

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
        If key is a single string, the corresponding entry in state dict is modified.
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
        x = state_dict[self.key[0]]
        sampling_rate = state_dict["metadata"]["trace_sampling_rate_hz"]

        sos = scipy.signal.butter(*self._filt_args, output="sos", fs=sampling_rate)
        if self.forward_backward:
            # Copy is required, because otherwise the stride of x is negative.T
            # This can break the pytorch collate function.
            x = scipy.signal.sosfiltfilt(sos, x, axis=self.axis).copy()
        else:
            x = scipy.signal.sosfilt(sos, x, axis=self.axis)

        state_dict[self.key[1]] = x

    def __str__(self):
        N, Wn, btype, analog = self._filt_args
        return (
            f"Filter ({btype}, order={N}, frequencies={Wn}, analog={analog}, "
            f"forward_backward={self.forward_backward}, axis={self.axis})"
        )
