import seisbench
import seisbench.util as util

from abc import abstractmethod, ABC
from pathlib import Path
import os
import torch
import torch.nn as nn
from collections import defaultdict, deque
from queue import PriorityQueue
import json
import numpy as np
import time


class SeisBenchModel(nn.Module):
    def __init__(self, name, citation=None):
        super().__init__()
        self._name = name
        self._citation = citation
        self._weights_docstring = None
        self._weights_metadata = None

    def __str__(self):
        return f"SeisBench model\t\t{self.name}\n\n{super().__str__()}"

    @property
    def name(self):
        return self._name

    @property
    def citation(self):
        return self._citation

    @property
    def weights_docstring(self):
        return self._weights_docstring

    def _model_path(self):
        return Path(seisbench.cache_root, "models", self.name.lower())

    def _remote_path(self):
        return os.path.join(seisbench.remote_root, "models", self.name.lower())

    def _pretrained_path(self, name):
        weight_path = self._model_path() / f"{name}.pt"
        metadata_path = self._model_path() / f"{name}.json"

        return weight_path, metadata_path

    @classmethod
    def from_pretrained(cls, name, force=False, wait_for_file=False):
        dummy_instance = cls()  # Required to query the paths
        weight_path, metadata_path = dummy_instance._pretrained_path(name)

        def download_weights_callback(weight_path):
            seisbench.logger.info(f"Weight file {name} not in cache. Downloading...")
            weight_path.parent.mkdir(exist_ok=True, parents=True)

            remote_weight_path = os.path.join(
                dummy_instance._remote_path(), f"{name}.pt"
            )
            util.download_http(remote_weight_path, weight_path)

        def download_metadata_callback(metadata_path):
            remote_metadata_path = os.path.join(
                dummy_instance._remote_path(), f"{name}.json"
            )
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


class WaveformModel(SeisBenchModel, ABC):
    """
    Abstract interface for models processing waveforms.
    Additionally implements functions commonly required to apply models to waveforms.
    Class does not inherit from SeisBenchModel to allow implementation of non-DL models with this interface.
    """

    def __init__(self, name, component_order=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        if component_order is None:
            self._component_order = seisbench.config["component_order"]
        else:
            self._component_order = component_order

    def __str__(self):
        return f"Component order:\t{self.component_order}\n{super().__str__()}"

    @property
    def component_order(self):
        return self._component_order

    @abstractmethod
    def annotate(self, stream, *args, **kwargs):
        """
        Annotates a stream
        :param stream: Obspy stream to annotate
        :param args:
        :param kwargs:
        :return: Obspy stream of annotations
        """
        pass

    @abstractmethod
    def classify(self, stream, *args, **kwargs):
        """
        Classifies the stream. As there are no
        :param stream: Obspy stream to classify
        :param args:
        :param kwargs:
        :return: A classification for the full stream, e.g., signal/noise or source magnitude
        """
        pass

    def _parse_metadata(self):
        super()._parse_metadata()
        self._component_order = self._weights_metadata.get(
            "component_order", seisbench.config["component_order"]
        )

    @staticmethod
    def resample(stream, sampling_rate):
        for trace in stream:
            if trace.stats.sampling_rate == sampling_rate:
                return
            if trace.stats.sampling_rate % sampling_rate == 0:
                trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
            else:
                trace.resample(sampling_rate)

    @staticmethod
    def groups_stream_by_instrument(stream):
        groups = defaultdict(list)
        for trace in stream:
            groups[trace.id[:-1]].append(trace)

        return list(groups.values())

    @staticmethod
    def has_mismatching_records(stream):
        """
        Detects if for any id the stream contains overlapping traces that do not match
        :param stream:
        :return:
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
        Converts streams into a list of start times and numpy arrays
        Assumes:
        - all traces in the stream are from the same instrument and only differ in the components
        - no overlapping traces of the same component exist
        - all traces have the same sampling rate
        :param stream:
        :param component_order:
        :param strict: If true, only annotate if recordings for all components are available, otherwise impute missing
        data with zeros.
        :return: times: Start times for each array
        :return: data: Arrays with waveforms
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