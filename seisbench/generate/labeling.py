import copy
import re
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.signal import stft

import seisbench
from seisbench import config


class SupervisedLabeller(ABC):
    """
    Supervised classification labels.
    Performs simple checks for standard supervised classification labels.

    :param label_type: The type of label either: 'multi_label', 'multi_class', 'binary'.
    :type label_type: str
    :param dim: Dimension over which labelling will be applied.
    :type dim: int
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    :type key: str, tuple[str, str]
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
        """
        to be overwritten in subclasses

        :param X: trace from state dict
        :type X: numpy.ndarray
        :param metadata: metadata from state dict
        :type metadata: dict
        :return: Label
        :rtype: numpy.ndarray
        """
        return

    @staticmethod
    def _swap_dimension_order(arr, current_dim, expected_dim):
        config_dim = tuple(current_dim.find(d) for d in expected_dim)
        return np.transpose(arr, config_dim)

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
                "Only computes dimension order given 3 dimensions (NCW), "
                "or 2 dimensions (CW)."
            )

        return sample_dim, channel_dim, width_dim

    def _check_labels(self, y, metadata, eps=1e-8):
        if (
            self.label_type == "multi_class"
            and self.label_method == "probabilistic"
            and getattr(self, "noise_column", True)
        ):
            if (y.sum(self.dim) > 1 + eps).any():
                raise ValueError(
                    "More than one label provided. For multi_class problems, only one label can be provided per input."
                )

        if self.label_type == "binary":
            if (y.sum(self.dim) > 1 + eps).any():
                raise ValueError("Binary labels should lie within 0-1 range.")

        for label in self.label_columns:
            if label not in metadata:
                # Ignore labels that are not present
                continue

            if (isinstance(metadata[label], (int, np.integer))) and self.ndim == 3:
                raise ValueError(
                    f"Only provided single arrival in metadata {label} column to multiple windows. "
                    f"Check augmentation workflow."
                )

    def __call__(self, state_dict):
        X, metadata = state_dict[self.key[0]]
        self.ndim = len(X.shape)

        y = self.label(X, metadata)
        self._check_labels(y, metadata)
        state_dict[self.key[1]] = (y, copy.deepcopy(metadata))


class PickLabeller(SupervisedLabeller, ABC):
    """
    Abstract class for PickLabellers implementing common functionality

    :param label_columns: Specify the columns to use for pick labelling, defaults to None and columns are inferred from
                          metadata. Columns can either be specified as list or dict.
                          For a list, each entry is treated as its own pick type.
                          The dict should contain a mapping from column name to pick label, e.g.,
                          {"trace_Pg_arrival_sample": "P"}.
                          This allows to group phases, e.g., Pg, Pn, pP all being labeled as P phase.
                          Multiple phases present within a window can lead to the labeller annotating multiple picks
                          for the same label.
    :param noise_column: If False, disables normalization of phases and noise label, default is True
    :param model_labels: Order of the labels, defaults to None.
                         If None, the labels will be in alphabetical order with Noise as the last label.
                         To get the labels from a WaveformModel, use by model.labels.
    :param kwargs: Kwargs are passed to the SupervisedLabeller superclass
    """

    def __init__(
        self,
        label_columns: Union[list[str], dict[str, str]] = None,
        noise_column: bool = True,
        model_labels: Union[str, list[str]] = None,
        **kwargs,
    ):
        self.label_columns = label_columns
        self.noise_column = noise_column
        self.model_labels = model_labels
        if label_columns is not None:
            (
                self.label_columns,
                self.labels,
                self.label_ids,
            ) = self._columns_to_dict_and_labels(
                label_columns, noise_column=noise_column, model_labels=model_labels
            )
        else:
            self.labels = None
            self.label_ids = None

        super(PickLabeller, self).__init__(**kwargs)

    @staticmethod
    def _auto_identify_picklabels(state_dict):
        """
        Identify pick columns from the generator state dict
        :return: Sorted list of pick columns
        """
        return sorted(
            list(filter(re.compile("trace_.*_arrival_sample").match, state_dict.keys()))
        )

    @staticmethod
    def _columns_to_dict_and_labels(
        label_columns,
        noise_column: bool = True,
        model_labels: Union[str, list[str]] = None,
    ):
        """
        Generate label columns dict and list of labels from label_columns list or dict.
        Always appends a noise column at the end by searching for 'n' or 'N' in model_labels.
        The indices of the labels are set automatically by the order of the given model_labels, e.g. 'PSN' results in
        label_ids {'Noise': 2, 'P': 0, 'S': 1}.

        :param label_columns: List of label columns or dict[label_columns -> labels]
        :param noise_column: If False, disables normalization of phases and noise label, default is True
        :param model_labels: Order of the labels, defaults to None.
                             If None, the labels will be in alphabetical order with Noise as the last label.
                             To get the labels from a WaveformModel, use by model.labels.
        :return: dict[label_columns -> labels], list[labels], dict[labels -> ids]
        """
        if not isinstance(label_columns, dict):
            label_columns = {label: label.split("_")[1] for label in label_columns}

        labels = sorted(list(np.unique(list(label_columns.values()))))

        if any(not x.endswith("_sample") for x in label_columns.keys()):
            seisbench.logger.warning(
                "Found key in labeler that does not end in '_sample'. "
                "If your dataset uses automatic resampling, the label position might be wrong."
            )

        if model_labels:
            label_ids = {
                label: [*model_labels].index(label) for label in labels
            }  # label ids for P and S
            if noise_column:
                if "n" in (label.lower() for label in model_labels):
                    label_ids["Noise"] = [
                        label.lower() for label in model_labels
                    ].index("n")
                else:
                    # The -1 is required if label_ids contains 0 or 1 values
                    label_ids["Noise"] = max(*label_ids.values(), -1) + 1
                labels.append("Noise")
        else:
            if noise_column:
                labels.append("Noise")
            label_ids = {label: i for i, label in enumerate(labels)}

        return label_columns, labels, label_ids


class StreamPickLabeller(PickLabeller):
    """
    Base class for stream-based pick labellers that share common probability distribution logic.
    Handles the common functionality of creating probability distributions at pick locations.
    """

    def __init__(
        self,
        shape: str = "gaussian",
        sigma: int = 10,
        labeller_type: str = "probabilistic",
        polarity_column: str = "trace_polarity",
        **kwargs,
    ):
        self.label_method = "probabilistic"
        self.sigma = sigma
        self.shape = shape
        self.labeller_type = labeller_type  # "probabilistic" or "polarity"
        self.polarity_column = polarity_column
        self._labelshape_fn_mapper = {
            "gaussian": gaussian_pick,
            "triangle": triangle_pick,
            "box": box_pick,
        }
        kwargs["dim"] = kwargs.get("dim", 1)
        super().__init__(label_type="multi_class", **kwargs)

    def _fill_y_value(self, y, label, label_val, metadata, i, j=None):
        """
        Fill y value based on labeller type.

        Args:
            y: Output array
            label: Label name
            label_val: Label value array
            metadata: Metadata dict
            i: Label index
            j: Sample index (for multi-window case)
        """
        if self.labeller_type == "polarity":
            # Polarity-specific logic
            if label == "P":
                if self.polarity_column in metadata:
                    polarity = metadata[self.polarity_column]
                    if polarity not in ["U", "D"]:
                        return
                    target_idx = self.label_ids.get(polarity)

                    if j is None:
                        y[target_idx, :] = np.maximum(y[target_idx, :], label_val)
                    else:
                        y[j, target_idx, :] = np.maximum(y[j, target_idx, :], label_val)
        else:
            # Probabilistic logic
            if j is None:
                y[i, :] = np.maximum(y[i, :], label_val)
            else:
                y[j, i, :] = np.maximum(y[j, i, :], label_val)

    def label(self, X, metadata):
        # Get label configuration
        if not self.label_columns:
            label_columns = self._auto_identify_picklabels(metadata)
            (
                self.label_columns,
                self.labels,
                self.label_ids,
            ) = self._columns_to_dict_and_labels(
                label_columns, self.noise_column, self.model_labels
            )
        if self.labeller_type == "polarity":
            self.labels = ["U", "D", "N"]
            self.label_ids = {"U": 0, "D": 1, "N": 2}

        sample_dim, channel_dim, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        # Initialize y array
        if self.ndim == 2:
            y = np.zeros(shape=(len(self.labels), X.shape[width_dim]))
        elif self.ndim == 3:
            y = np.zeros(
                shape=(
                    X.shape[sample_dim],
                    len(self.labels),
                    X.shape[width_dim],
                )
            )
        else:
            raise ValueError(f"ndim must be either 2 or 3, not {self.ndim}")

        # Construct pick labels
        for label_column, label in self.label_columns.items():
            if label_column not in metadata:
                # Unknown pick
                continue

            i = self.label_ids[label]

            if isinstance(metadata[label_column], (int, np.integer, float)):
                # Handle single window case
                onset = metadata[label_column]
                if self.shape in self._labelshape_fn_mapper.keys():
                    label_val = self._labelshape_fn_mapper[self.shape](
                        onset=onset, length=X.shape[width_dim], sigma=self.sigma
                    )
                else:
                    raise ValueError(
                        f"Labeller of shape {self.shape} is not implemented."
                    )

                label_val[np.isnan(label_val)] = (
                    0  # Set non-present pick probabilities to 0
                )
                self._fill_y_value(y, label, label_val, metadata, i)
            else:
                # Handle multi-window case
                for j in range(X.shape[sample_dim]):
                    onset = metadata[label_column][j]
                    if self.shape in self._labelshape_fn_mapper.keys():
                        label_val = self._labelshape_fn_mapper[self.shape](
                            onset=onset, length=X.shape[width_dim], sigma=self.sigma
                        )
                    else:
                        raise ValueError(
                            f"Labeller of shape {self.shape} is not implemented."
                        )

                    label_val[np.isnan(label_val)] = 0
                    self._fill_y_value(y, label, label_val, metadata, i, j)

        # Handle noise normalization
        if self.noise_column or self.labeller_type == "polarity":
            y /= np.maximum(1, np.nansum(y, axis=channel_dim, keepdims=True))

            # Construct noise label
            if self.labeller_type == "polarity":
                noise_idx = self.label_ids.get("N", len(self.labels) - 1)
            else:
                noise_idx = self.label_ids.get("Noise", len(self.labels) - 1)
            if self.ndim == 2:
                y[noise_idx, :] = 1 - np.nansum(y, axis=channel_dim)
            elif self.ndim == 3:
                y[:, noise_idx, :] = 1 - np.nansum(y, axis=channel_dim)

        # Swap dimensions
        if self.ndim == 2:
            y = self._swap_dimension_order(
                y,
                current_dim="CW",
                expected_dim=config["dimension_order"].replace("N", ""),
            )
        elif self.ndim == 3:
            y = self._swap_dimension_order(
                y, current_dim="NCW", expected_dim=config["dimension_order"]
            )

        return y


class ProbabilisticLabeller(StreamPickLabeller):
    """
    Create supervised labels from picks with probabilistic representation.
    """

    def __init__(self, shape: str = "gaussian", sigma: int = 10, **kwargs):
        super().__init__(
            shape=shape, sigma=sigma, labeller_type="probabilistic", **kwargs
        )

    def __str__(self):
        return f"ProbabilisticLabeller (label_type={self.label_type}, dim={self.dim})"


class PolarityLabeller(StreamPickLabeller):
    """
    Create probabilistic supervised labels from seismic polarity information.
    """

    def __init__(
        self,
        polarity_column: str = "trace_polarity",
        shape: str = "gaussian",
        sigma: int = 10,
        **kwargs,
    ):
        self.polarity_column = polarity_column
        super().__init__(shape=shape, sigma=sigma, labeller_type="polarity", **kwargs)

    def __str__(self):
        return f"PolarityLabeller (label_type={self.label_type}, dim={self.dim})"


class StepLabeller(PickLabeller):
    """
    Create supervised labels from picks. The picks are represented by probability curves with value 0
    before the pick and 1 afterwards. The output contains one channel per pick type and no noise channel.

    All picks with NaN sample are treated as not present.
    """

    def __init__(self, **kwargs):
        self.label_method = "probabilistic"
        kwargs["dim"] = kwargs.get("dim", -2)
        super().__init__(label_type="multi_label", noise_column=False, **kwargs)

    def label(self, X, metadata):
        if not self.label_columns:
            label_columns = self._auto_identify_picklabels(metadata)
            (
                self.label_columns,
                self.labels,
                self.label_ids,
            ) = self._columns_to_dict_and_labels(
                label_columns, noise_column=False, model_labels=self.model_labels
            )

        sample_dim, channel_dim, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        if self.ndim == 2:
            y = np.zeros(shape=(len(self.labels), X.shape[width_dim]))
        elif self.ndim == 3:
            y = np.zeros(
                shape=(
                    X.shape[sample_dim],
                    len(self.labels),
                    X.shape[width_dim],
                )
            )

        # Construct pick labels
        for label_column, label in self.label_columns.items():
            i = self.label_ids[label]

            if label_column not in metadata:
                # Unknown pick
                continue

            if isinstance(metadata[label_column], (int, np.integer, float)):
                # Handle single window case
                onset = metadata[label_column]
                if not np.isnan(onset):
                    onset = max(0, onset)
                    y[i, int(onset) :] = 1
            else:
                # Handle multi-window case
                for j in range(X.shape[sample_dim]):
                    onset = metadata[label_column][j]
                    if not np.isnan(onset):
                        onset = max(0, onset)
                        y[j, i, int(onset) :] = 1

        return y

    def __str__(self):
        return f"ProbabilisticLabeller (label_type={self.label_type}, dim={self.dim})"


class ProbabilisticPointLabeller(ProbabilisticLabeller):
    """
    This labeller annotates windows true their pick probabilities at a certain point in the window.
    The output is a probability vector, i.e., [0.5, 0.2, 0.3] could indicate 50 % P wave, 20 % S wave, 30 % noise.
    This labelling scheme is more flexible than the :py:class:`StandardLabeller` and can encode for example the
    centrality of a pick within a window or multiple picks within the same window.

    This class relies on the ProbabilisticLabeller, but instead of probability curves only returns the probabilities
    at one point.

    :param position: Position to label as fraction of the total trace length. Defaults to 0.5, i.e.,
                     the center of the window.
    :type position: float
    :param kwargs: Passed to ProbabilisticLabeller
    """

    def __init__(self, position=0.5, **kwargs):
        self.label_method = "probabilistic"
        self.position = position
        if not 0 <= position <= 1:
            raise ValueError("position must be between 0 and 1.")
        kwargs["dim"] = kwargs.get("dim", 0)
        super().__init__(**kwargs)

    def label(self, X, metadata):
        y = super().label(X, metadata)

        _, _, width_dim = self._get_dimension_order_from_config(config, self.ndim)

        position_sample = int(self.position * (y.shape[width_dim] - 1))
        y = np.take(y, position_sample, axis=width_dim)

        return y

    def __str__(self):
        return (
            f"ProbabilisticPointLabeller (label_type={self.label_type}, dim={self.dim})"
        )


class DetectionLabeller(SupervisedLabeller):
    """
    Create detection labels from picks.
    The labeler can either use fixed detection length or determine the length from the P to S time as in
    Mousavi et al. (2020, Nature communications). In the latter case, detections range from P to S + factor * (S - P)
    and are only annotated if both P and S phases are present.
    All detections are represented through a boxcar time series with the same length as the input waveforms.
    For both P and S, lists of phases can be passed of which the sequentially first one will be used.
    All picks with NaN sample are treated as not present.

    :param p_phases: (List of) P phase metadata columns
    :type p_phases: str, list[str]
    :param s_phases: (List of) S phase metadata columns
    :type s_phases: str, list[str]
    :param factor: Factor for length of window after S onset
    :type factor: float
    :param fixed_window: Number of samples for fixed window detections. If none, will determine length from P to S time.
    :type fixed_window: int
    """

    def __init__(
        self, p_phases, s_phases=None, factor=1.4, fixed_window=None, **kwargs
    ):
        self.label_method = "probabilistic"
        self.label_columns = "detections"
        if isinstance(p_phases, str):
            self.p_phases = [p_phases]
        else:
            self.p_phases = p_phases

        if isinstance(s_phases, str):
            self.s_phases = [s_phases]
        elif s_phases is None:
            self.s_phases = []
        else:
            self.s_phases = s_phases

        if s_phases is not None and fixed_window is not None:
            seisbench.logger.warning(
                "Provided both S phases and fixed window length to DetectionLabeller. "
                "Will use fixed window size and ignore S phases."
            )

        self.factor = factor
        self.fixed_window = fixed_window

        kwargs["dim"] = kwargs.get("dim", -2)
        super().__init__(label_type="multi_class", **kwargs)

    def label(self, X, metadata):
        sample_dim, channel_dim, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        if self.fixed_window:
            # Only label until end of fixed window
            factor = 0
        else:
            factor = self.factor

        if self.ndim == 2:
            y = np.zeros((1, X.shape[width_dim]))
            p_arrivals = [
                metadata[phase]
                for phase in self.p_phases
                if phase in metadata and not np.isnan(metadata[phase])
            ]
            if self.fixed_window is not None:
                # Fake S arrivals for simulating fixed window
                s_arrivals = [x + self.fixed_window for x in p_arrivals]
            else:
                s_arrivals = [
                    metadata[phase]
                    for phase in self.s_phases
                    if phase in metadata and not np.isnan(metadata[phase])
                ]

            if len(p_arrivals) != 0 and len(s_arrivals) != 0:
                p_arrival = min(p_arrivals)
                s_arrival = min(s_arrivals)
                p_to_s = s_arrival - p_arrival
                if s_arrival >= p_arrival:
                    # Only annotate valid options
                    p0 = max(int(p_arrival), 0)
                    p1 = max(int(s_arrival + factor * p_to_s), 0)
                    y[0, p0:p1] = 1

        elif self.ndim == 3:
            y = np.zeros(
                shape=(
                    X.shape[sample_dim],
                    1,
                    X.shape[width_dim],
                )
            )
            p_arrivals = [
                metadata[phase] for phase in self.p_phases if phase in metadata
            ]

            if self.fixed_window is not None:
                # Fake S arrivals for simulating fixed window
                s_arrivals = [x + self.fixed_window for x in p_arrivals]
            else:
                s_arrivals = [
                    metadata[phase] for phase in self.s_phases if phase in metadata
                ]

            if len(p_arrivals) != 0 and len(s_arrivals) != 0:
                p_arrivals = np.stack(p_arrivals, axis=-1)  # Shape (samples, phases)
                s_arrivals = np.stack(s_arrivals, axis=-1)

                mask = np.logical_and(
                    np.any(~np.isnan(p_arrivals), axis=1),
                    np.any(~np.isnan(s_arrivals), axis=1),
                )
                if not mask.any():
                    return y

                p_arrivals = np.nanmin(
                    p_arrivals[mask, :], axis=1
                )  # Shape (samples (which are present),)
                s_arrivals = np.nanmin(s_arrivals[mask, :], axis=1)
                p_to_s = s_arrivals - p_arrivals

                starts = p_arrivals.astype(int)
                ends = (s_arrivals + factor * p_to_s).astype(int)

                # print(mask, starts, ends)
                for i, s, e in zip(np.arange(len(mask))[mask], starts, ends):
                    s = max(0, s)
                    e = max(0, e)
                    y[i, 0, s:e] = 1

        else:
            raise ValueError(
                f"Illegal number of input dimensions for DetectionLabeller (ndim={self.ndim})."
            )

        return y

    def __str__(self):
        return f"DetectionLabeller (label_type={self.label_type}, dim={self.dim})"


class StandardLabeller(PickLabeller):
    """
    Create supervised labels from picks. The entire example is labelled as a single class/pick.
    For cases where multiple picks overlap in the input window, a number of options can be specified:

      - 'label-first' Only use first pick as label example.
      - 'fixed-relevance' Use pick closest to centre point of window as example.
      - 'random' Use random choice as example label.

    In general, it is recommended to set low and high, as it is very difficult for models to spot if a pick is just
    inside or outside the window. This can lead to noisy predictions that strongly depend on the marginal label
    distribution in the training set.

    :param low: Only take into account picks after or at this sample. If None, uses low=0.
                If negative, counts from the end of the trace.
    :type low: int, None
    :param high: Only take into account picks before this sample. If None, uses high=num_samples.
                 If negative, counts from the end of the trace.
    :type high: int, None
    :param on_overlap: Method used to label when multiple picks present in window, defaults to "label-first"
    :type on_overlap: str, optional
    """

    def __init__(self, on_overlap="label-first", low=None, high=None, **kwargs):
        kwargs["dim"] = kwargs.get("dim", 1)
        self.label_method = "standard"
        self.on_overlap = on_overlap
        self.low = low
        self.high = high

        if self.on_overlap not in ("label-first", "fixed-relevance", "random"):
            raise ValueError(
                "Unexpected value for `on_overlap` argument. Accepted values are: "
                "`label-first`, `fixed-relevance`, `random`."
            )

        super().__init__(label_type="multi_class", **kwargs)

    def _get_pick_arrivals_as_array(self, metadata, trace_length, add_sample_dim):
        """
        Convert picked arrivals to numpy array. Set arrivals outside
        window to null (NaN) values.
        """
        arrival_array = []
        for col in self.label_columns:
            if col in metadata:
                if add_sample_dim:
                    arrival_array.append(np.array([metadata[col]]))
                else:
                    arrival_array.append(metadata[col])
            else:
                arrival_array.append(None)

        # Identify shape of entry
        nan_dummy = np.ones(1) * np.nan  # Only used for all nan arrays
        for x in arrival_array:
            if x is not None:
                nan_dummy = np.ones_like(x, dtype=float) * np.nan
                break

        # Replace all picks missing in metadata with NaN
        new_arrival_array = []
        for x in arrival_array:
            if x is None:
                new_arrival_array.append(nan_dummy)
            else:
                new_arrival_array.append(x)

        arrival_array = np.array(new_arrival_array, dtype=float).T

        if self.low is None:
            low = 0
        else:
            if self.low > 0:
                low = self.low
            else:
                low = trace_length + self.low

        if self.high is None:
            high = trace_length
        else:
            if self.high > 0:
                high = self.high
            else:
                high = trace_length + self.high

        arrival_array[arrival_array < low] = np.nan  # Mask samples before low
        arrival_array[arrival_array >= high] = np.nan  # Mask samples after high
        nan_mask = np.isnan(arrival_array)
        return arrival_array, nan_mask

    @staticmethod
    def _label_first(row_id, arrival_array, nan_mask):
        """
        Label based on first arrival in time in window.
        """
        arrival_array[nan_mask] = np.inf
        first_arrival = np.nanargmin(arrival_array[row_id])
        return first_arrival

    @staticmethod
    def _label_random(row_id, nan_mask):
        """
        Label by randomly choosing pick inside window.
        If no arrivals present, label as noise (class 0)
        """
        non_null_columns = np.argwhere(~nan_mask[row_id])
        return np.random.choice(non_null_columns.reshape(-1))

    @staticmethod
    def _label_fixed_relevance(row_id, arrival_array, midpoint):
        """
        Label using closest pick to centre of window.
        If no arrivals present, label as noise (class 0)
        """
        return np.nanargmin(abs(arrival_array[row_id] - midpoint))

    def label(self, X, metadata):
        if not self.label_columns:
            label_columns = self._auto_identify_picklabels(metadata)
            (
                self.label_columns,
                self.labels,
                self.label_ids,
            ) = self._columns_to_dict_and_labels(
                label_columns, model_labels=self.model_labels
            )

        sample_dim, _, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        # Construct pick labels
        if sample_dim is None:
            y = np.zeros(shape=(1, 1))
        else:
            y = np.zeros(shape=(X.shape[sample_dim], 1))

        arrival_array, nan_mask = self._get_pick_arrivals_as_array(
            metadata, X.shape[width_dim], sample_dim is None
        )
        n_arrivals = nan_mask.shape[1] - nan_mask.sum(axis=1)

        for row_id, arrivals in enumerate(arrival_array):
            # Label
            if n_arrivals[row_id] == 0:
                y[row_id] = self.label_ids["Noise"]
            else:
                if n_arrivals[row_id] == 1:
                    label_column_id = np.nanargmin(arrivals)
                # On potentially overlapping labels
                else:
                    if self.on_overlap == "label-first":
                        label_column_id = self._label_first(
                            row_id, arrival_array, nan_mask
                        )
                    elif self.on_overlap == "random":
                        label_column_id = self._label_random(row_id, nan_mask)
                    elif self.on_overlap == "fixed-relevance":
                        midpoint = X.shape[width_dim] // 2
                        label_column_id = self._label_fixed_relevance(
                            row_id, arrival_array, midpoint
                        )

                label_column = list(self.label_columns.keys())[label_column_id]
                y[row_id] = self.label_ids[self.label_columns[label_column]]

        if sample_dim is None:
            y = y.reshape(y.shape[1:])  # Remove fake sample_dim

        return y.astype(int)

    def __str__(self):
        return f"StandardLabeller (label_type={self.label_type}, dim={self.dim})"


class STFTDenoiserLabeller:
    """
    Create supervised labels for DeepDenoiser and SeisDAE with short-time Fourier transform (STFT).
    Noise samples from noise_dataset are randomly selected and earthquakes waveforms are deteriorated
    by adding randomly scaled earthquake and noise waveforms.

    :param noise_dataset: Seisbench WaveformDataset that only contains noise waveforms for training.
    :param scale: Tuple of minimum and maximum relative amplitude of the earthquake and noise waveform.
                  Both earthquake and noise waveforms are scaled independently by using np.random.uniform.
                  Relative amplitude is defined either by the absolute maximum or the root-mean-square of input array
                  (see scaling_type). Default value is (0, 1).
    :param scaling_type: Method how to find the relative amplitude of the input array. Either from the absolute maximum
                         (peak) or from the standard deviation (std). Default is peak.
    :param component: Component to take into account for training. Default are three components, ZNE.
                      If DeepDenoiser/SeisDAE is only trained with a single component, use e.g., component="Z".
                      Otherwise, DeepDenoiser/SeisDAE is trained with all components, however, noise samples are only
                      added on same component of earthquake waveforms, i.e., on Z-component are only added noise
                      waveforms of Z-component and so on.
    :param input_key: Key for reading data from the state_dict. Default is "X".
    :param noisy_output_key: Key to write noisy STFT to state_dict. Default is "X", i.e. the input data
                             will be overwritten.
    :param mask_key: Key for writing computed mask function to state_dict. Default is "y".
    :param nfft: Length of the FFT used, if a zero padded FFT is desired for scipy STFT.
                 If None, the FFT length is nperseg.
    :param nperseg: Length of each segment for scipy STFT. Default is 60
    :param eps: Avoiding division by zero, default is 1e-12.
    """

    def __init__(
        self,
        noise_dataset: seisbench.data.WaveformDataset,
        scale: tuple[float, float] = (0, 1),
        scaling_type: str = "peak",
        component: str = "ZNE",
        nfft: int = 60,
        nperseg: int = 30,
        eps: float = 1e-12,
        input_key: str = "X",
        noisy_output_key: str = "X",
        mask_key: str = "y",
        **kwargs,
    ):
        self.noise_dataset = noise_dataset
        self.scale = scale
        self.scaling_type = scaling_type.lower()

        self.input_key = input_key
        self.noisy_output_key = noisy_output_key
        self.mask_key = mask_key
        self.component = component
        self.noise_generator = seisbench.generate.GenericGenerator(noise_dataset)
        self.noise_samples = len(noise_dataset)
        self.nfft = nfft
        self.nperseg = nperseg
        self.eps = eps
        self.kwargs = kwargs

        if self.scaling_type.lower() not in ["std", "peak"]:
            msg = "Argument scaling_type must be either 'std' or 'peak'."
            raise ValueError(msg)

    def __call__(self, state_dict):
        x, metadata = state_dict[self.input_key]

        if self.input_key != self.noisy_output_key:
            # Ensure data and metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)
            x = x.copy()

        # Select component if only one is given. If more than one component is given, choose random component
        # Note, if more than one component is given, components of signal and noise are not mixed
        if len(self.component) == 1:
            component_idx = metadata["trace_component_order"].index(self.component)
            metadata["trace_component_order"] = self.component
        else:
            component_idx = np.random.randint(0, len(self.component) - 1)
            metadata["trace_component_order"] = self.component[component_idx]

        # Select component from x and modify metadata
        x = x[component_idx, :]

        # Defining scale for earthquake and noise amplitude
        # I.e., scale for earthquake and noise can become, for example, zero and thus either no noise or
        # earthquake waveform exists. Using this approach, the denoiser learns better to distinguish between noise
        # and earthquake.
        scale_earthquake = np.random.uniform(*self.scale)
        scale_noise = np.random.uniform(*self.scale)

        # Draw random noise sample from the dataset and cut to same length as x
        n = self.noise_generator[np.random.randint(low=0, high=self.noise_samples)]["X"]

        # Select noise component
        n = n[component_idx, :]

        # Removing mean from earthquake and noise samples
        x = x - np.mean(x, axis=-1, keepdims=True)
        n = n - np.mean(n, axis=-1, keepdims=True)

        # Normalize earthquake and noise samples
        if self.scaling_type == "peak":
            x = x / (np.max(np.abs(x)) + self.eps)
            n = n / (np.max(np.abs(n)) + self.eps)
        elif self.scaling_type == "std":
            x = x / (np.std(x) + self.eps)
            n = n / (np.std(n) + self.eps)

        # Cutting noise to same length as x
        if n.shape[-1] < x.shape[-1]:
            msg = (
                f"The length of the data ({len(x)} samples) and the noise ({len(n)} samples) must either be the same "
                f"or the shape of the noise must be larger than the shape of the data."
            )
            raise ValueError(msg)

        if n.shape[-1] < x.shape[-1]:
            spoint = np.random.randint(low=0, high=len(n) - len(x))
        else:
            spoint = 0
        n = n[spoint : spoint + len(x)]

        # Scale earthquake (x) and noise (n) waveforms
        x *= scale_earthquake
        n *= scale_noise

        # Add noise and signal to create noisy signal
        noisy = x + n

        # Normalize noisy, x and n by normalization factor of noisy to range [-1, 1]
        if self.scaling_type == "peak":
            norm_noisy = np.max(np.abs(noisy)) + self.eps
        elif self.scaling_type == "std":
            norm_noisy = np.std(noisy) + self.eps

        # Scale earthquake, noise and noisy waveform with same scaling factor, that noisy is in range [-1, 1]
        x = x / norm_noisy
        n = n / norm_noisy
        noisy = noisy / norm_noisy

        # Perform STFT of x, n, and noisy
        _, _, stft_x = stft(
            x=x,
            fs=metadata["trace_sampling_rate_hz"],
            nperseg=self.nperseg,
            nfft=self.nfft,
            **self.kwargs,
        )
        _, _, stft_n = stft(
            x=n,
            fs=metadata["trace_sampling_rate_hz"],
            nperseg=self.nperseg,
            nfft=self.nfft,
            **self.kwargs,
        )
        _, _, stft_noisy = stft(
            x=noisy,
            fs=metadata["trace_sampling_rate_hz"],
            nperseg=self.nperseg,
            nfft=self.nfft,
            **self.kwargs,
        )

        # Determine output masks
        X = np.empty(
            shape=(2, *stft_n.shape)
        )  # Input, i.e. real and imag part of noisy

        y = np.empty(
            shape=(2, *stft_n.shape)
        )  # Target, i.e. masks for signal and noise

        # Normalize real and imaginary parts of STFT to range [-1, 1]
        # Normalizing real and imaginary part might distort amplitude and phase, however, from my experiences the
        # denoising result is more accurate. If you train a SeisDAE model without normalization, don't forget to
        # remove the normalization in the prediction (seisdae.annotate_batch_pre)
        X[0, :, :] = stft_noisy.real / (np.max(np.abs(stft_noisy.real)) + self.eps)
        X[1, :, :] = stft_noisy.imag / (np.max(np.abs(stft_noisy.imag)) + self.eps)

        # Compute target masking function
        y[0, :, :] = 1 / (1 + np.abs(stft_n) / (np.abs(stft_x) + self.eps))
        y[1, :, :] = 1 - y[0, :, :]

        # Replace nan values in X and y
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

        # Update state_dicts for input and target
        state_dict[self.noisy_output_key] = (X, metadata)
        state_dict[self.mask_key] = (y, metadata)

    def __str__(self):
        return (
            f"Labeller for STFT and masking functions for Denoiser "
            f"(nfft={self.nfft}, nperseg={self.nperseg})"
        )


def gaussian_pick(onset, length, sigma):
    r"""
    Create probabilistic representation of pick in time series.
    PDF function given by:

    .. math::
        \mathcal{N}(\mu,\,\sigma^{2})

    :param onset: The nearest sample to pick onset
    :type onset: float
    :param length: The length of the trace time series in samples
    :type length: int
    :param sigma: The variance of the Gaussian distribution in samples
    :type sigma: float
    :return prob_pick: 1D time series with probabilistic representation of pick
    :rtype: np.ndarray
    """
    x = np.arange(length)
    return np.exp(-np.power(x - onset, 2.0) / (2 * np.power(sigma, 2.0)))


def triangle_pick(onset, length, sigma):
    r"""
    Create triangle representation of pick in time series.

    :param onset: The nearest sample to pick onset
    :type onset: float
    :param length: The length of the trace time series in samples
    :type length: int
    :param sigma: The half width of the triangle distribution in samples
    :type sigma: float
    :return y: 1D time series with triangle representation of pick
    :rtype: np.ndarray
    """
    x = np.arange(length)
    y1 = -(x - onset) / sigma + 1
    y1 *= (y1 >= 0) & (y1 <= 1)
    y2 = (x - onset) / sigma + 1
    y2 *= (y2 >= 0) & (y2 < 1)
    y = y1 + y2
    return y


def box_pick(onset, length, sigma):
    r"""
    Create box representation of pick in time series.

    :param onset: The nearest sample to pick onset
    :type onset: float
    :param length: The length of the trace time series in samples
    :type length: int
    :param sigma: The half width of the box distribution in samples
    :type sigma: float
    :return y: 1D time series with box representation of pick
    :rtype: np.ndarray
    """
    x = np.arange(length)
    return ((x - onset) <= sigma) & (-(x - onset + 1) < sigma)
