import numpy

import seisbench
import seisbench.util as util

from abc import abstractmethod, ABC
from pathlib import Path
import os
import torch
import torch.nn as nn
from collections import defaultdict
from queue import PriorityQueue
import json
import numpy as np
import obspy
import warnings


class SeisBenchModel(nn.Module):
    """
    Base SeisBench model interface for processing waveforms.

    :param citation: Citation reference, defaults to None.
    :type citation: str, optional
    """

    def __init__(self, citation=None):
        super().__init__()
        self._citation = citation
        self._weights_docstring = None
        self._weights_metadata = None

    def __str__(self):
        return f"SeisBench model\t\t{self.name}\n\n{super().__str__()}"

    @property
    def name(self):
        return self._name_internal()

    @classmethod
    def _name_internal(cls):
        return cls.__name__

    @property
    def citation(self):
        return self._citation

    @property
    def weights_docstring(self):
        return self._weights_docstring

    @classmethod
    def _model_path(cls):
        return Path(seisbench.cache_root, "models", cls._name_internal().lower())

    @classmethod
    def _remote_path(cls):
        return os.path.join(
            seisbench.remote_root, "models", cls._name_internal().lower()
        )

    @classmethod
    def _pretrained_path(cls, name):
        weight_path = cls._model_path() / f"{name}.pt"
        metadata_path = cls._model_path() / f"{name}.json"

        return weight_path, metadata_path

    @classmethod
    def from_pretrained(cls, name, force=False, wait_for_file=False):
        """
        Load pretrained model with weights.

        :param name: Model name prefix.
        :type name: str
        :param force: Force execution of download callback, defaults to False
        :type force: bool, optional
        :param wait_for_file: Whether to wait on partially downloaded files, defaults to False
        :type wait_for_file: bool, optional
        :return: PyTorch model instance
        :rtype: torch.nn.Module
        """
        weight_path, metadata_path = cls._pretrained_path(name)

        def download_weights_callback(weight_path):
            seisbench.logger.info(f"Weight file {name} not in cache. Downloading...")
            weight_path.parent.mkdir(exist_ok=True, parents=True)

            remote_weight_path = os.path.join(cls._remote_path(), f"{name}.pt")
            util.download_http(remote_weight_path, weight_path)

        def download_metadata_callback(metadata_path):
            remote_metadata_path = os.path.join(cls._remote_path(), f"{name}.json")
            try:
                util.download_http(
                    remote_metadata_path, metadata_path, progress_bar=False
                )
            except ValueError:
                # A missing metadata file does not lead to a crash
                pass

        seisbench.util.callback_if_uncached(
            weight_path,
            download_weights_callback,
            force=force,
            wait_for_file=wait_for_file,
        )
        seisbench.util.callback_if_uncached(
            metadata_path,
            download_metadata_callback,
            force=force,
            wait_for_file=wait_for_file,
        )

        if metadata_path.is_file():
            with open(metadata_path, "r") as f:
                weights_metadata = json.load(f)
        else:
            weights_metadata = {}

        model_args = weights_metadata.get("model_args", {})
        model = cls(**model_args)

        model._weights_metadata = weights_metadata
        model._parse_metadata()

        model.load_state_dict(torch.load(weight_path))

        return model

    def _parse_metadata(self):
        self._weights_docstring = self._weights_metadata.get("docstring", "")


# TODO: Add classify function using aggregation of annotate results
class WaveformModel(SeisBenchModel, ABC):
    """
    Abstract interface for models processing waveforms.
    Based on the properties specified by inheriting models, WaveformModel automatically provides the respective
    :py:func:`annotate`/:py:func:`classify` functions. For details see the documentation of these functions.

    :param component_order: Specify component order (e.g. ['ZNE']), defaults to None.
    :type component_order: list, optional
    :param sampling_rate: Sampling rate of the model, defaults to None.
                          If sampling rate is not None, the annotate and classify functions will automatically resample
                          incoming traces and validate correct sampling rate if the model overwrites
                          :py:func:`annotate_stream_pre`.
    :type sampling_rate: float
    :param output_type: The type of output from the model. Current options are:

                        - "point" for a point prediction, i.e., the probability of containing a pick in the window
                          or of a pick at a certain location. This will provide an :py:func:`annotate` function.
                          If an :py:func:`classify_aggregate` function is provided by the inheriting model,
                          this will also provide a :py:func:`classify` function.
                        - "array" for prediction curves, i.e., probabilities over time for the arrival of certain wave types.
                          This will provide an :py:func:`annotate` function.
                          If an :py:func:`classify_aggregate` function is provided by the inheriting model,
                          this will also provide a :py:func:`classify` function.
                        - "regression" for a regression value, i.e., the sample of the arrival within a window.
                          This will only provide a :py:func:`classify` function.

    :type output_type: str
    :param default_args: Default arguments to use in annotate and classify functions
    :type default_args: dict[str, any]
    :param in_samples: Number of input samples in time
    :type in_samples: int
    :param pred_sample: For a "point" prediction: sample number of the sample in a window for which the prediction is valid.
                        For an "array" prediction: a tuple of first and last sample defining the prediction range.
                        Note that the number of output samples and input samples within the given range are not required
                        to agree.
    :type pred_sample: int, tuple
    :param labels: Labels for the different predictions in the output, e.g., Noise, P, S
    :type labels: list or string
    :param kwargs: Kwargs are passed to the superclass
    """

    def __init__(
        self,
        component_order=None,
        sampling_rate=None,
        output_type=None,
        default_args=None,
        in_samples=None,
        pred_sample=0,
        labels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if component_order is None:
            self._component_order = seisbench.config["component_order"]
        else:
            self._component_order = component_order

        self.sampling_rate = sampling_rate
        self.output_type = output_type
        if default_args is None:
            self.default_args = {}
        else:
            self.default_args = default_args
        self.in_samples = in_samples
        self.pred_sample = pred_sample

        # Validate pred sample
        if output_type == "point" and not isinstance(pred_sample, (int, float)):
            raise TypeError(
                "For output type 'point', pred_sample needs to be a scalar."
            )
        if output_type == "array":
            if not isinstance(pred_sample, (list, tuple)) or not len(pred_sample) == 2:
                raise TypeError(
                    "For output type 'array', pred_sample needs to be a tuple of length 2."
                )
            if pred_sample[0] < 0 or pred_sample[1] < 0:
                raise ValueError(
                    "For output type 'array', both entries of pred_sample need to be non-negative."
                )

        self.labels = labels

        self._annotate_function_mapping = {
            "point": self._annotate_point,
            "array": self._annotate_array,
        }
        self._classify_function_mapping = {"regression": None}

        self._annotate_function = self._annotate_function_mapping.get(output_type, None)
        self._classify_function = self._classify_function_mapping.get(output_type, None)

    def __str__(self):
        return f"Component order:\t{self.component_order}\n{super().__str__()}"

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def component_order(self):
        return self._component_order

    def annotate(self, stream, strict=True, **kwargs):
        """
        Annotates an obspy stream using the model based on the configuration of the WaveformModel superclass.
        For example, for a picking model, annotate will give a characteristic function/probability function for picks
        over time.
        The annotate function contains multiple subfunction, which can be overwritten individually by inheriting
        models to accomodate their requirements. These functions are:

        - :py:func:`annotate_stream_pre`
        - :py:func:`annotate_stream_validate`
        - :py:func:`annotate_window_pre`
        - :py:func:`annotate_window_post`

        Please see the respective documentation for details on their functionality, inputs and outputs.

        :param stream: Obspy stream to annotate
        :type stream: obspy.core.Stream
        :param strict: If true, only annotate if recordings for all components are available,
                       otherwise impute missing data with zeros.
        :type strict: bool
        :param args:
        :param kwargs:
        :return: Obspy stream of annotations
        """
        if self._annotate_function is None:
            raise NotImplementedError(
                "This model has no annotate function implemented."
            )

        # Kwargs overwrite default args
        argdict = self.default_args.copy()
        argdict.update(kwargs)

        stream = stream.copy()
        stream.merge(-1)

        output = obspy.Stream()
        if len(stream) == 0:
            return output

        # Preprocess stream, e.g., filter/resample
        self.annotate_stream_pre(stream, argdict)

        # Validate stream
        self.annotate_stream_validate(stream, argdict)

        # Group stream
        groups = self.groups_stream_by_instrument(stream)

        # Stream to arrays to windows
        for group in groups:
            trace = group[0]
            # Sampling rate of the data. Equal to self.sampling_rate is this is not None
            argdict["sampling_rate"] = trace.stats.sampling_rate

            times, data = self.stream_to_arrays(group, strict=strict)

            pred_times, pred_rates, preds = self._annotate_function(
                times, data, argdict
            )

            # Write to output stream
            output += self._predictions_to_stream(pred_rates, pred_times, preds, trace)

        return output

    def _predictions_to_stream(self, pred_rates, pred_times, preds, trace):
        """
        Converts a set of predictions to obspy streams

        :param pred_rates: Sampling rates of the prediction arrays
        :param pred_times: Start time of each prediction array
        :param preds: The prediction arrays, each with shape (samples, channels)
        :param trace: A source trace to extract trace naming from
        :return: Obspy stream of predictions
        """
        output = obspy.Stream()
        for (pred_time, pred_rate, pred) in zip(pred_times, pred_rates, preds):
            for i in range(pred.shape[1]):
                if self.labels is None:
                    label = i
                else:
                    label = self.labels[i]

                trimmed_pred, f, _ = self._trim_nan(pred[:, i])
                trimmed_start = pred_time + f / pred_rate
                output.append(
                    obspy.Trace(
                        trimmed_pred,
                        {
                            "starttime": trimmed_start,
                            "sampling_rate": pred_rate,
                            "network": trace.stats.network,
                            "station": trace.stats.station,
                            "location": trace.stats.location,
                            "channel": f"{self.__class__.__name__}_{label}",
                        },
                    )
                )

        return output

    def annotate_stream_pre(self, stream, argdict):
        """
        Runs preprocessing on stream level for the annotate function, e.g., filtering or resampling.
        By default, this function will resample all traces if a sampling rate for the model is provided.
        As annotate create a copy of the input stream, this function can safely modify the stream inplace.
        Inheriting classes should overwrite this function if necessary.
        To keep the default functionality, a call to the overwritten method can be included.

        :param stream: Input stream
        :type stream: obspy.Stream
        :param argdict: Dictionary of arguments
        :return: Preprocessed stream
        """
        if self.sampling_rate is not None:
            self.resample(stream, self.sampling_rate)
        return stream

    def annotate_stream_validate(self, stream, argdict):
        """
        Validates stream for the annotate function.
        This function should raise an exception if the stream is invalid.
        By default, this function will check if the sampling rate fits the provided one, unless it is None,
        and check for mismatching traces, i.e., traces covering the same time range on the same instrument with
        different values.
        Inheriting classes should overwrite this function if necessary.
        To keep the default functionality, a call to the overwritten method can be included.

        :param stream: Input stream
        :type stream: obspy.Stream
        :param argdict: Dictionary of arguments
        :return: None
        """
        if self.sampling_rate is not None:
            if any(trace.stats.sampling_rate != self.sampling_rate for trace in stream):
                raise ValueError(
                    f"Detected traces with mismatching sampling rate. "
                    f"Expected {self.sampling_rate} Hz for all traces."
                )

        if self.has_mismatching_records(stream):
            raise ValueError(
                "Detected multiple records for the same time and component that did not agree."
            )

    def annotate_window_pre(self, window, argdict):
        """
        Runs preprocessing on window level for the annotate function, e.g., normalization.
        By default returns the input window.
        Inheriting classes should overwrite this function if necessary.

        :param window: Input window
        :type window: numpy.array
        :param argdict: Dictionary of arguments
        :return: Preprocessed window
        """
        return window

    def annotate_window_post(self, pred, argdict):
        """
        Runs postprocessing on the predictions of a window for the annotate function, e.g., reformatting them.
        By default returns the original prediction.
        Inheriting classes should overwrite this function if necessary.

        :param pred: Predictions for one window. The data type depends on the model.
        :param argdict: Dictionary of arguments
        :return: Postprocessed predictions
        """
        return pred

    def _annotate_point(self, times, data, argdict):
        """
        Annotation function for a point prediction model using a sliding window approach.
        Will use the key `stride` from the `argdict` to determine the shift (in samples) between two windows.
        Default `stride` is 1.
        This function expects model outputs after postprocessing for each window to be scalar or 1D arrays.
        """
        stride = argdict.get("stride", 1)

        pred_times = []
        pred_rates = []
        full_preds = []

        # Iterate over all blocks of waveforms
        for t0, block in zip(times, data):
            starts = np.arange(0, block.shape[1] - self.in_samples + 1, stride)
            if len(starts) == 0:
                seisbench.logger.warning(
                    "Parts of the input stream consist of fragments shorter than the number "
                    "of input samples. Output might be empty."
                )
                continue

            # Generate windows and preprocess
            fragments = [
                self.annotate_window_pre(block[:, s : s + self.in_samples], argdict)
                for s in starts
            ]
            fragments = np.stack(fragments, axis=0)
            fragments = torch.tensor(fragments, device=self.device, dtype=torch.float32)

            with torch.no_grad():
                preds = self._predict_and_postprocess_windows(argdict, fragments)
                preds = np.stack(preds, axis=0)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)

                pred_times.append(t0 + self.pred_sample / argdict["sampling_rate"])
                pred_rates.append(argdict["sampling_rate"] / stride)
                full_preds.append(preds)

        return pred_times, pred_rates, full_preds

    def _annotate_array(self, times, data, argdict):
        """
        Annotation function for an array prediction model using a sliding window approach.
        Will use the key `overlap` from the `argdict` to determine the overlap (in samples) between two neighboring windows.
        Overlapping predictions will be averaged. NaN predictions will be ignored in the averaging.
        This function expects model outputs after postprocessing for each window to be 1D arrays (only sample dimension)
        or 2D arrays (sample and channel dimension in this order) and that each prediction has the same number of output samples.
        """
        overlap = argdict.get("overlap", 0)

        pred_times = []
        pred_rates = []
        full_preds = []

        # Iterate over all blocks of waveforms
        for t0, block in zip(times, data):
            starts = np.arange(
                0, block.shape[1] - self.in_samples + 1, self.in_samples - overlap
            )
            if len(starts) == 0:
                seisbench.logger.warning(
                    "Parts of the input stream consist of fragments shorter than the number "
                    "of input samples. Output might be empty."
                )
                continue

            # Generate windows and preprocess
            fragments = [
                self.annotate_window_pre(block[:, s : s + self.in_samples], argdict)
                for s in starts
            ]
            fragments = np.stack(fragments, axis=0)
            fragments = torch.tensor(fragments, device=self.device, dtype=torch.float32)

            with torch.no_grad():
                preds = self._predict_and_postprocess_windows(argdict, fragments)

                # Number of prediction samples per input sample
                prediction_sample_factor = preds[0].shape[0] / (
                    self.pred_sample[1] - self.pred_sample[0]
                )

                # Maximum number of predictions covering a point
                coverage = int(
                    np.ceil(self.in_samples / (self.in_samples - overlap) + 1)
                )

                pred_length = int(np.ceil(block.shape[1] * prediction_sample_factor))
                pred_merge = (
                    np.zeros_like(
                        preds[0], shape=(pred_length, preds[0].shape[1], coverage)
                    )
                    * np.nan
                )
                for i, (pred, start) in enumerate(zip(preds, starts)):
                    pred_start = int(start * prediction_sample_factor)
                    pred_merge[
                        pred_start : pred_start + pred.shape[0], :, i % coverage
                    ] = pred

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        action="ignore", message="Mean of empty slice"
                    )
                    preds = np.nanmean(pred_merge, axis=-1)

                pred_times.append(t0 + self.pred_sample[0] / argdict["sampling_rate"])
                pred_rates.append(argdict["sampling_rate"] * prediction_sample_factor)
                full_preds.append(preds)

        return pred_times, pred_rates, full_preds

    def _predict_and_postprocess_windows(self, argdict, fragments):
        train_mode = self.training
        preds = []
        try:
            self.eval()
            batch_size = argdict.get("batch_size", 64)
            p0 = 0
            # Iterate over batches
            while p0 < fragments.shape[0]:
                preds.append(self(fragments[p0 : p0 + batch_size]))
                p0 += batch_size
        finally:
            if train_mode:
                self.train()
        preds = self._recursive_torch_to_numpy(preds)
        # Separate and postprocess window predictions
        reshaped_preds = []
        for pred_batch in preds:
            reshaped_preds += [
                self.annotate_window_post(pred, argdict)
                for pred in self._recursive_slice_pred(pred_batch)
            ]
        return reshaped_preds

    @staticmethod
    def _trim_nan(x):
        """
        Removes all starting and trailing nan values from a 1D array and returns the new array and the number of NaNs removed per side.
        """
        mask_forward = np.cumprod(np.isnan(x)).astype(
            bool
        )  # cumprod will be one until the first non-Nan value
        x = x[~mask_forward]
        mask_backward = np.cumprod(np.isnan(x)[::-1])[::-1].astype(
            bool
        )  # Double reverse for a backwards cumprod
        x = x[~mask_backward]

        return x, np.sum(mask_forward.astype(int)), np.sum(mask_backward.astype(int))

    def _recursive_torch_to_numpy(self, x):
        """
        Recursively converts torch.Tensor objects to numpy arrays while preserving any overarching tuple or list structure.
        :param x:
        :return:
        """
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, list):
            return [self._recursive_torch_to_numpy(y) for y in x]
        elif isinstance(x, tuple):
            return tuple([self._recursive_torch_to_numpy(y) for y in x])
        else:
            raise ValueError(f"Can't unpack object of type {type(x)}.")

    def _recursive_slice_pred(self, x):
        """
        Converts batched predictions into a list of single predictions, assuming batch axis is first in all cases.
        Preserves overarching tuple and list structures
        :param x:
        :return:
        """
        if isinstance(x, numpy.ndarray):
            return list(y for y in x)
        elif isinstance(x, list):
            return [
                list(entry)
                for entry in zip(*[self._recursive_slice_pred(y) for y in x])
            ]
        elif isinstance(x, tuple):
            return [entry for entry in zip(*[self._recursive_slice_pred(y) for y in x])]
        else:
            raise ValueError(f"Can't unpack object of type {type(x)}.")

    @abstractmethod
    def classify(self, stream, *args, **kwargs):
        """
        Classifies the stream.

        :param stream: Obspy stream to classify
        :type stream: obspy.core.Stream
        :param args:
        :param kwargs:
        :return: A classification for the full stream, e.g., signal/noise or source magnitude.
        """
        pass

    def _parse_metadata(self):
        super()._parse_metadata()
        self._component_order = self._weights_metadata.get(
            "component_order", seisbench.config["component_order"]
        )

    @staticmethod
    def resample(stream, sampling_rate):
        """
        Perform inplace resampling of stream to a given sampling rate.

        :param stream: Input stream
        :type stream: obspy.core.Stream
        :param sampling_rate: Sampling rate (sps) to resample to
        :type sampling_rate: float
        """
        for trace in stream:
            if trace.stats.sampling_rate == sampling_rate:
                return
            if trace.stats.sampling_rate % sampling_rate == 0:
                trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
            else:
                trace.resample(sampling_rate)

    @staticmethod
    def groups_stream_by_instrument(stream):
        """
        Perform instrument-based grouping of input stream.

        :param stream: Input stream
        :type stream: obspy.core.Stream
        :return: List of traces grouped by instrument.
        :rtype: list
        """
        groups = defaultdict(list)
        for trace in stream:
            groups[trace.id[:-1]].append(trace)

        return list(groups.values())

    @staticmethod
    def has_mismatching_records(stream):
        """
        Detects if for any id the stream contains overlapping traces that do not match.

        :param stream: Input stream
        :type stream: obspy.core.Stream
        :return: Flag whether there are mismatching records
        :rtype: bool
        """
        stream.merge(-1)  # Ensures overlapping matching traces are merged

        ids = defaultdict(list)
        for trace in stream:
            ids[trace.id].append(trace)

        for traces in ids.values():
            starttimes = sorted(
                [(trace.stats.starttime, i) for i, trace in enumerate(traces)],
                key=lambda x: x[0],
            )
            endtimes = sorted(
                [(trace.stats.endtime, i) for i, trace in enumerate(traces)],
                key=lambda x: x[0],
            )

            for i, _ in enumerate(starttimes):
                if starttimes[i][1] != endtimes[i][1]:
                    return True
                if i > 0 and starttimes[i] < endtimes[i - 1]:
                    return True

        return False

    def stream_to_arrays(self, stream, strict=True):
        """
        Converts streams into a list of start times and numpy arrays.
        Assumes:

        - All traces in the stream are from the same instrument and only differ in the components
        - No overlapping traces of the same component exist
        - All traces have the same sampling rate

        :param stream: Input stream
        :type stream: obspy.core.Stream
        :param strict: If true, only annotate if recordings for all components are available, otherwise impute missing data with zeros.
        :type strict: bool, default True
        :return: output_times: Start times for each array
        :return: output_data: Arrays with waveforms
        """
        seqnum = 0  # Obspy raises an error when trying to compare traces. The seqnum hack guarantees that no two tuples reach comparison of the traces.
        if len(stream) == 0:
            return [], []

        sampling_rate = stream[0].stats.sampling_rate

        component_order = self._component_order
        comp_dict = {c: i for i, c in enumerate(component_order)}

        start_sorted = PriorityQueue()
        for trace in stream:
            if trace.id[-1] in component_order and len(trace.data) > 0:
                start_sorted.put((trace.stats.starttime, seqnum, trace))
                seqnum += 1

        active = (
            PriorityQueue()
        )  # Traces with starttime before the current time, but endtime after
        to_write = (
            []
        )  # Traces that are not active any more, but need to be written in the next array. Irrelevant for strict mode

        output_times = []
        output_data = []
        while True:
            if not start_sorted.empty():
                start_element = start_sorted.get()
            else:
                start_element = None
                if strict:
                    # In the strict case, all data would already have been written
                    break

            if not active.empty():
                end_element = active.get()
            else:
                end_element = None

            if start_element is None and end_element is None:
                # Processed all data
                break
            elif start_element is not None and end_element is None:
                active.put(
                    (start_element[2].stats.endtime, start_element[1], start_element[2])
                )
            elif start_element is None and end_element is not None:
                to_write.append(end_element[2])
            else:
                # both start_element and end_element are active
                # TODO: If end_element == start_element, it depends on strict what we want to do
                if end_element[0] < start_element[0] or (
                    strict and end_element[0] == start_element[0]
                ):
                    to_write.append(end_element[2])
                    start_sorted.put(start_element)
                else:
                    active.put(
                        (
                            start_element[2].stats.endtime,
                            start_element[1],
                            start_element[2],
                        )
                    )
                    active.put(end_element)

            if not strict and active.qsize() == 0 and len(to_write) != 0:
                t0 = min(trace.stats.starttime for trace in to_write)
                t1 = max(trace.stats.endtime for trace in to_write)

                data = np.zeros(
                    (len(component_order), int((t1 - t0) * sampling_rate + 2))
                )  # +2 avoids fractional errors

                for trace in to_write:
                    p = int((trace.stats.starttime - t0) * sampling_rate)
                    cidx = comp_dict[trace.id[-1]]
                    data[cidx, p : p + len(trace.data)] = trace.data

                data = data[:, :-1]  # Remove fractional error +1

                output_times.append(t0)
                output_data.append(data)

                to_write = []

            if strict and active.qsize() == len(component_order):
                traces = []
                while not active.empty():
                    traces.append(active.get()[2])

                t0 = max(trace.stats.starttime for trace in traces)
                t1 = min(trace.stats.endtime for trace in traces)

                short_traces = [trace.slice(t0, t1) for trace in traces]
                data = np.zeros((len(component_order), len(short_traces[0].data)))
                for trace in short_traces:
                    cidx = comp_dict[trace.id[-1]]
                    data[cidx] = trace.data

                output_times.append(t0)
                output_data.append(data)

                for trace in traces:
                    if t1 < trace.stats.endtime:
                        start_sorted.put((t1, seqnum, trace.slice(starttime=t1)))
                        seqnum += 1

        return output_times, output_data
