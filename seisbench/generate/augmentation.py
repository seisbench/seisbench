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
        :param key: The key to extract from the state_dict for normalization
        """
        self.demean_axis = demean_axis
        self.detrend_axis = detrend_axis
        self.amp_norm_axis = amp_norm_axis
        self.amp_norm_type = amp_norm_type
        self.key = key

        if self.amp_norm_type not in ["peak", "std"]:
            raise ValueError(
                f"Unrecognized amp_norm_type '{self.amp_norm_type}'. Available types are: 'peak', 'std'."
            )

    def __call__(self, state_dict):
        x = state_dict[self.key]
        if x.dtype.char in np.typecodes["AllInteger"]:
            # Cast int types to float to ensure all operations are mathematically possible
            x = x.astype(float)

        x = self._demean(x)
        x = self._detrend(x)
        x = self._amp_norm(x)

        state_dict[self.key] = x

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
        :param key: The key to extract from the state_dict for filtering
        """
        self.forward_backward = forward_backward
        self.axis = axis
        self.key = key
        self._filt_args = (N, Wn, btype, analog)

    def __call__(self, state_dict):
        x = state_dict[self.key]
        sampling_rate = state_dict["metadata"]["trace_sampling_rate_hz"]

        sos = scipy.signal.butter(*self._filt_args, output="sos", fs=sampling_rate)
        if self.forward_backward:
            # Copy is required, because otherwise the stride of x is negative.T
            # This can break the pytorch collate function.
            x = scipy.signal.sosfiltfilt(sos, x, axis=self.axis).copy()
        else:
            x = scipy.signal.sosfilt(sos, x, axis=self.axis)

        state_dict[self.key] = x

    def __str__(self):
        N, Wn, btype, analog = self._filt_args
        return (
            f"Filter ({btype}, order={N}, frequencies={Wn}, analog={analog}, "
            f"forward_backward={self.forward_backward}, axis={self.axis})"
        )
