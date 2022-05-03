import seisbench

import copy
import numpy as np


class FixedWindow:
    """
    A simple windower that returns fixed windows.
    In addition, the windower rewrites all metadata ending in "_sample" to point to the correct sample after window
    selection. Window start and length can be set either at initialization or separately in each call.
    The later is primarily intended for more complicated windowers inheriting from FixedWindow.

    :param p0: Start position of the trace. If p0 is negative, this will be treated as identifying
               a sample before the start of the trace. This is in contrast to standard list indexing
               with negative indices in Python, which counts items from the end of the list. Negative
               p0 is not possible with the strategy "fail".
    :type p0: None or int
    :param windowlen: Window length
    :type windowlen: None or int
    :param strategy: Strategy to mitigate insufficient data. Options are:

                     *  "fail": Raises a ValueError

                     *  "pad": Adds zero padding to the window

                     *  "move": Moves the start to the closest possible position to achieve sufficient trace length.
                        The resulting trace will be aligned to the input trace on one of the ends,
                        depending if parts before (left aligned) or after the trace (right aligned) were requested.
                        Will fail if total trace length is shorter than requested window length.

                     *  "variable": Returns shorter length window, resulting in possibly varying window size.
                        Might return empty window if requested window is completely outside target range.

    :type strategy: str
    :param axis: Axis along which the window selection should be performed
    :type axis: int
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]
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
                    f"Requested window length ({windowlen}) is longer than available length after p0 "
                    f"({x.shape[self.axis] - p0})."
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
                        f"Total trace length ({x.shape[self.axis]}) is shorter than requested window length "
                        f"({windowlen})."
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

    :param timestep: Difference between two consecutive window starts in samples
    :type timestep: int
    :param windowlen: Length of the output window
    :type windowlen: int
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
    The generator will return a window starting at the first sample, if all relevant metadata entries are NaN.

    :param samples_before: The number of samples to include before the target sample.
    :type samples_before: int
    :param selection: Selection strategy in case multiple metadata keys are provided and have non-NaN values.
                      Options are:

        - "first": use the first available key
        - "random": use uniform random selection among the keys

    :type selection: str
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
            metadata[key]
            for key in self.metadata_keys
            if key in metadata and not np.isnan(metadata[key])
        ]

        if len(cand) == 0:
            cand = [self.samples_before]
            # raise ValueError("Found no possible candidate for window selection")

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
        return (
            f"WindowAroundSample (metadata_keys={self.metadata_keys}, samples_before={self.samples_before}, "
            f"selection={self.selection})"
        )


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
    :type low: None or int
    :param high: The highest allowed index for the end.
                 The sample at position high can *not* be included in the output
    :type high: None or int
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


class SteeredWindow(FixedWindow):
    """
    Window selection that relies on the "_control_" dict from the :py:class:`SteeredGenerator`.
    Selects a window of given length with the window defined by start and end sample in the middle.
    If no length is given, the window between start and end sample is returned.
    If there are insufficient samples on one side of the target window, the window will be moved.
    If the total number of samples is insufficient, the window will start at the earliest possible sample.
    The behavior in this case depends on the chosen strategy for :py:class:`FixedWindow`.

    :param windowlen: Length of the window to be returned.
                      If None, will be determined using the start and end samples from the "_control_" dict.
    :type windowlen: int or None
    :param start_key: Key of the start sample in the "_control_" dict
    :type start_key: str
    :param end_key: Key of the end sample in the "_control_" dict
    :type end_key: str
    :param window_output_key: The sample start and end will be written as numpy array to this key in the state_dict
    :type window_output_key: str
    :param kwargs: Parameters passed to the init method of FixedWindow.
    """

    def __init__(
        self,
        windowlen,
        start_key="start_sample",
        end_key="end_sample",
        window_output_key="window_borders",
        **kwargs,
    ):
        super().__init__(windowlen=windowlen, **kwargs)

        self.start_key = start_key
        self.end_key = end_key
        self.window_output_key = window_output_key

    def __call__(self, state_dict):
        control = state_dict["_control_"]

        start_sample = int(control[self.start_key])
        end_sample = int(control[self.end_key])

        if self.windowlen is None:
            windowlen = end_sample - start_sample
        else:
            windowlen = self.windowlen

        x, _ = state_dict[self.key[0]]
        n_samples = x.shape[self.axis]

        sample_range = end_sample - start_sample
        if sample_range > windowlen:
            raise ValueError(
                f"Requested window length is {windowlen}, but the difference between start and end sample "
                f"is already {end_sample - start_sample}. Can't return window."
            )

        elif sample_range == windowlen:
            p0 = max(0, start_sample)

        else:
            p0 = (
                start_sample - (windowlen - sample_range) // 2
            )  # Try to put the window into the center

            if p0 + windowlen > n_samples:
                p0 = (
                    n_samples - windowlen
                )  # If the window end is behind the trace end, move it backward

            p0 = max(
                0, p0
            )  # If the window start is before the trace start, move it forward

        window_borders = np.array([start_sample - p0, end_sample - p0])
        state_dict[self.window_output_key] = (
            window_borders,
            {},
        )  # No metadata associated

        super().__call__(state_dict, p0=p0, windowlen=windowlen)

    def __str__(self):
        return f"SteeredWindow"
