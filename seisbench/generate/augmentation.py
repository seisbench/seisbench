import numpy as np
import scipy.signal
import copy


class Normalize:
    """
    A normalization augmentation that allows demeaning, detrending and amplitude normalization (in this order).

    :param demean_axis: The axis (single axis or tuple) which should be jointly demeaned.
                        None indicates no demeaning.
    :type demean_axis: int, None
    :param detrend_axis: The axis along with detrending should be applied.
                         None indicates no normalization.
    :type detrend_axis: int, None
    :param amp_norm_axis: The axis (single axis or tuple) which should be jointly amplitude normalized.
                     None indicates no normalization.
    :type amp_norm_axis: int, None
    :param amp_norm_type: Type of amplitude normalization. Supported types:
        - "peak": division by the absolute peak of the trace
        - "std": division by the standard deviation of the trace
    :type amp_norm_type: str
    :param eps: Epsilon value added in amplitude normalization to avoid division by zero
    :type eps: float
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]
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
                x = x / (
                    np.max(np.abs(x), axis=self.amp_norm_axis, keepdims=True) + self.eps
                )
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


class Copy:
    """
    A copy augmentation. Maps data from a given key in the state_dict to a new key.

    :param key: The keys for reading from and writing to the state dict.
                A a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]

    """

    def __init__(self, key=("X", "Xc")):
        self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        state_dict[self.key[1]] = (x.copy(), copy.deepcopy(metadata))

    def __str__(self):
        return f"Copy (prev_key={self.key[0]}, new_key={self.key[1]})"


class Filter:
    """
    Implements a filter augmentation, closely based on scipy.signal.butter.
    Please refer to the scipy documentation for more detailed description of the parameters

    :param N: Filter order
    :type N: int
    :param Wn: The critical frequency or frequencies
    :type Wn: list/array of float
    :param btype: The filter type: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
    :type btype: str
    :param analog: When True, return an analog filter, otherwise a digital filter is returned.
    :type analog: bool
    :param forward_backward: If true, filters once forward and once backward.
                             This doubles the order of the filter and makes the filter zero-phase.
    :type forward_backward: bool
    :param axis: Axis along which the filter is applied.
    :type axis: int
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]

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
        if isinstance(sampling_rate, (list, tuple, np.ndarray)):
            if not np.allclose(sampling_rate, sampling_rate[0]):
                raise NotImplementedError(
                    "Found mixed sampling rates in filter. "
                    "Filter currently only works on consistent sampling rates."
                )
            else:
                sampling_rate = sampling_rate[0]

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
    :type include: list[str] or None
    :param exclude: All keys except these keys will be present in the output.
    :type exclude: list[str] or None

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
    :type dtype: numpy.dtype
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]
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


class OneOf:
    """
    Runs one of the augmentations provided, choosing randomly each time called.

    :param augmentations: A list of augmentations
    :type augmentations: callable
    :param probabilities: Probability for each augmentation to be used.
                          Probabilities will automatically be normed to sum to 1.
                          If None, equal probability is assigned to each augmentation.
    :type probabilities: list/array/tuple of scalar
    """

    def __init__(self, augmentations, probabilities=None):
        self.augmentations = augmentations
        self.probabilities = probabilities

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, value):
        if value is None:
            self._probabilities = np.array(
                [1 / len(self.augmentations)] * len(self.augmentations)
            )
        else:
            if len(value) != len(self.augmentations):
                raise ValueError(
                    f"Number of augmentations and probabilities need to be identical, "
                    f"but got {len(self.augmentations)} and {len(value)}."
                )
            self._probabilities = np.array(value) / np.sum(value)

    def __call__(self, state_dict):
        augmentation = np.random.choice(self.augmentations, p=self.probabilities)
        augmentation(state_dict)


class NullAugmentation:
    """
    This augmentation does not perform any modification on the state dict.
    It is primarily intended to be used as a no-op for :py:class:`OneOf`.
    """

    def __call__(self, state_dict):
        pass

    def __str__(self):
        return "NullAugmentation"


class ChannelDropout:
    """
    Similar to Dropout, zeros out between 0 and the c - 1 channels randomly.
    Outputs are multiplied by the inverse of the fraction of remaining channels.
    As for regular Dropout, this ensures that the output "energy" is unchanged.

    :param axis: Channel dimension, defaults to -2.
    :type axis: int
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]
    """

    def __init__(self, axis=-2, key="X"):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

        self.axis = axis

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]

        if self.key[0] != self.key[1]:
            # Ensure data and metadata are not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)
            x = x.copy()

        axis = self.axis
        if axis < 0:
            axis += x.ndim

        n_channels = x.shape[axis]
        n_drop = np.random.randint(n_channels)  # Number of channels to drop

        if n_drop > 0:
            drop_channels = np.arange(n_channels)
            np.random.shuffle(drop_channels)
            drop_channels = drop_channels[:n_drop]

            for i in range(x.ndim):
                if i < axis:
                    drop_channels = np.expand_dims(drop_channels, 0)
                elif i > axis:
                    drop_channels = np.expand_dims(drop_channels, -1)

            np.put_along_axis(x, drop_channels, 0, axis=axis)

        state_dict[self.key[1]] = (x, metadata)


class AddGap:
    """
    Adds a gap into the data by zeroing out entries.

    :param axis: Sample dimension, defaults to -1.
    :type axis: int
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]
    """

    def __init__(self, axis=-1, key="X"):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

        self.axis = axis

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]

        if self.key[0] != self.key[1]:
            # Ensure data and metadata are not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)
            x = x.copy()

        axis = self.axis
        if axis < 0:
            axis += x.ndim

        n_samples = x.shape[axis]

        gap_start = np.random.randint(n_samples - 1)
        gap_end = np.random.randint(gap_start, n_samples)
        gap = np.arange(gap_start, gap_end, dtype=int)

        for i in range(x.ndim):
            if i < axis:
                gap = np.expand_dims(gap, 0)
            elif i > axis:
                gap = np.expand_dims(gap, -1)

        np.put_along_axis(x, gap, 0, axis=axis)

        state_dict[self.key[1]] = (x, metadata)


class RandomArrayRotation:
    """
    Randomly rotates a set of arrays, i.e., shifts samples along an axis and puts the end samples to the start.
    The same rotation will be applied to each array.
    All arrays need to have the same length along the target axis.

    .. warning::
        This augmentation does **not** modify the metadata, as positional entries anyhow become non-unique
        after rotation. Workflows should therefore always first generate labels from metadata and then jointly
        rotate data and labels.

    :param keys: Single key specification or list of key specifications.
                 Each key specification is either a string, for identical input and output keys,
                 or as a tuple of two strings, input and output keys.
                 Defaults to "X".
    :type keys: Union[list[Union[str, tuple[str, str]]], str, tuple[str,str]]
    :param axis: Sample axis. Either a single integer or a list of integers for multiple keys.
                 If a single integer but multiple keys are provided, the same axis will be used for each key.
                 Defaults to -1.
    :type axis: Union[int, list[int]]
    """

    def __init__(self, keys="X", axis=-1):
        # Single key
        if not isinstance(keys, list):
            keys = [keys]

        # Resolve identical input and output keys
        self.keys = []
        for key in keys:
            if isinstance(key, tuple):
                self.keys.append(key)
            else:
                self.keys.append((key, key))

        if isinstance(axis, list):
            self.axis = axis
        else:
            self.axis = [axis] * len(self.keys)

    def __call__(self, state_dict):
        rotation = None
        n_samples = None

        for key, axis in zip(self.keys, self.axis):
            x, metadata = state_dict[key[0]]

            if key[0] != key[1]:
                # Ensure metadata is not modified inplace unless input and output key are anyhow identical
                metadata = copy.deepcopy(metadata)

            if n_samples is None:
                n_samples = x.shape[axis]
                rotation = np.random.randint(n_samples)
            else:
                if n_samples != x.shape[axis]:
                    raise ValueError(
                        "RandomArrayRotation requires all inputs to be of identical length along "
                        "the provided axis."
                    )

            x = np.roll(x, rotation, axis=axis)

            state_dict[key[1]] = (x, metadata)


class GaussianNoise:
    """
    Adds point-wise independent Gaussian noise to an array.

    :param scale: Tuple of minimum and maximum relative amplitude of the noise.
                  Relative amplitude is defined as the quotient of the standard deviation of the noise and
                  the absolute maximum of the input array.
    :type scale: tuple[float, float]
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]
    """

    def __init__(self, scale=(0, 0.15), key="X"):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

        self.scale = scale

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]

        if self.key[0] != self.key[1]:
            # Ensure metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)

        scale = np.random.uniform(*self.scale) * np.max(x)
        noise = np.random.randn(*x.shape).astype(x.dtype) * scale
        x = x + noise

        state_dict[self.key[1]] = (x, metadata)

    def __str__(self):
        return f"GaussianNoise (Scale (mu={self.scale[0]}, sigma={self.scale[1]}))"
