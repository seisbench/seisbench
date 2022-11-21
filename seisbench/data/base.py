import seisbench
import seisbench.util

from abc import abstractmethod, ABC
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import inspect
import scipy.signal
import copy
from collections import defaultdict
from collections.abc import Iterable
import warnings


class WaveformDataset:
    """
    This class is the base class for waveform datasets.

    A key consideration should be how the cache is used. If sufficient memory is available to keep the full data set
    in memory, activating the cache will yield strong performance gains. For details on the cache strategies, see the
    documentation of the ``cache`` parameter.

    :param path: Path to dataset.
    :type path: pathlib.Path, str
    :param name: Dataset name, default is None.
    :type name: str, optional
    :param dimension_order: Dimension order e.g. 'CHW', if not specified will be assumed from config file,
                            defaults to None.
    :type dimension_order: str, optional
    :param component_order: Component order e.g. 'ZNE', if not specified will be assumed from config file,
                            defaults to None.
    :type component_order: str, optional
    :param sampling_rate: Common sampling rate of waveforms in dataset, sampling rate can also be specified
                          as a metadata column if not common across dataset.
    :type sampling_rate: int, optional
    :param cache: Defines the behaviour of the waveform cache. Provides three options:

                  *  "full": When a trace is queried, the full block containing the trace is loaded into the cache
                     and stored in memory. This causes the highest memory consumption, but also best
                     performance when using large parts of the dataset.

                  *  "trace": When a trace is queried, only the trace itself is loaded and stored in memory.
                     This is particularly useful when only a subset of traces is queried,
                     but these are queried multiple times. In this case the performance of
                     this strategy might outperform "full".

                  *  None: When a trace is queried, it is always loaded from disk.
                     This mode leads to low memory consumption but high IO load.
                     It is most likely not usable for model training.

                  Note that for datasets without blocks, i.e., each trace in a single array in the hdf5 file,
                  the strategies "full" and "trace" are identical.
                  The default cache strategy is None.

                  Use :py:func:`preload_waveforms` to populate the cache.
                  Preloading the waveforms is often much faster than loading them during later application,
                  as preloading can use sequential access.
                  Note that it is recommended to always first filter a dataset and then preload to reduce
                  unnecessary reads and memory consumption.
    :type cache: str, optional
    :param chunks: Specify particular chunks to load. If None, loads all chunks. Defaults to None.
    :type chunks: list, optional
    :param missing_components: Strategy to deal with missing components. Options are:

                               *  "pad": Fill with zeros.

                               *  "copy": Fill with values from first existing traces.

                               *  "ignore": Order all existing components in the requested order,
                                  but ignore missing ones. This will raise an error if traces with different
                                  numbers of components are requested together.
    :type missing_components: str
    :param kwargs:
    """

    def __init__(
        self,
        path,
        name=None,
        dimension_order=None,
        component_order=None,
        sampling_rate=None,
        cache=None,
        chunks=None,
        missing_components="pad",
        **kwargs,
    ):
        if name is None:
            self._name = "Unnamed dataset"
        else:
            self._name = name

        self.cache = cache
        self._path = path
        self._chunks = chunks
        if chunks is not None:
            self._chunks = sorted(chunks)

            available_chunks = self.available_chunks(path)
            for chunk in self._chunks:
                if chunk not in available_chunks:
                    raise ValueError(f"Dataset does not contain the chunk '{chunk}'.")

        self._missing_components = None

        self._trace_identification_warning_issued = (
            False  # Traced whether warning for trace name was issued already
        )

        self._dimension_order = None  # Target dimension order
        self._dimension_mapping = None  # List for reordering input to target dimensions
        self._component_order = None  # Target component order
        # Dict [source_component_order -> list for reordering source to target components]
        self._component_mapping = None
        self.sampling_rate = sampling_rate

        self._verify_dataset()

        metadatas = []
        for chunk, metadata_path, _ in zip(*self._chunks_with_paths()):
            with warnings.catch_warnings():
                # Catch warning for mixed dtype
                warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

                tmp_metadata = pd.read_csv(
                    metadata_path,
                    dtype={
                        "trace_sampling_rate_hz": float,
                        "trace_dt_s": float,
                        "trace_component_order": str,
                    },
                )
            tmp_metadata["trace_chunk"] = chunk
            metadatas.append(tmp_metadata)
        self._metadata = pd.concat(metadatas)

        self._data_format = self._read_data_format()

        self._unify_sampling_rate()
        self._unify_component_order()
        self._build_trace_name_to_idx_dict()

        self.dimension_order = dimension_order
        self.component_order = component_order
        self.missing_components = missing_components

        self._waveform_cache = defaultdict(dict)

    def __str__(self):
        return f"{self._name} - {len(self)} traces"

    def __add__(self, other):
        if isinstance(other, WaveformDataset):
            return MultiWaveformDataset([self, other])
        elif isinstance(other, MultiWaveformDataset):
            return MultiWaveformDataset([self] + other.datasets)
        else:
            raise TypeError(
                "Can only add WaveformDataset and MultiWaveformDataset to WaveformDataset."
            )

    def copy(self):
        """
        Create a copy of the data set. All attributes are copied by value, except waveform cache entries.
        The cache entries are copied by reference, as the waveforms will take up most of the memory.
        This should be fine for most use cases, because the cache entries should anyhow never be modified.
        Note that the cache dict itself is not shared, such that cache evictions and
        inserts in one of the data sets do not affect the other one.

        :return: Copy of the dataset
        """

        other = copy.copy(self)
        other._metadata = self._metadata.copy()
        other._waveform_cache = defaultdict(dict)
        for key in self._waveform_cache.keys():
            other._waveform_cache[key] = copy.copy(self._waveform_cache[key])

        return other

    @property
    def metadata(self):
        """
        Metadata of the dataset as pandas DataFrame.
        """
        return self._metadata

    @property
    def name(self):
        """
        Name of the dataset (immutable)
        """
        return self._name

    @property
    def cache(self):
        """
        Get or set the cache strategy of the dataset. For possible strategies see the constructor.
        Note that changing cache strategies will not cause a cache eviction.
        """
        return self._cache

    @cache.setter
    def cache(self, cache):
        if cache not in ["full", "trace", None]:
            raise ValueError(
                f"Unknown cache strategy '{cache}'. Allowed values are 'full', 'trace' and None."
            )

        self._cache = cache

    @property
    def path(self):
        """
        Path of the dataset (immutable)
        """
        if self._path is None:
            raise ValueError("Path is None. Can't create data set without a path.")
        return Path(self._path)

    @property
    def data_format(self):
        """
        Data format dictionary, describing the data format of the stored dataset.
        Note that this does not necessarily equals the output data format of get waveforms.
        To query these, use the relevant class properties.
        """
        # Returns a copy of the data format, to ensure internal data format is not modified.
        return dict(self._data_format)

    @property
    def dimension_order(self):
        """
        Get or set the order of the dimension in the output.
        """
        return self._dimension_order

    @dimension_order.setter
    def dimension_order(self, value):
        if value is None:
            value = seisbench.config["dimension_order"]

        self._dimension_mapping = self._get_dimension_mapping(
            "N" + self._data_format["dimension_order"], value
        )
        self._dimension_order = value

    @property
    def missing_components(self):
        """
        Get or set strategy to handle missing components. For options, see the constructor.
        """
        return self._missing_components

    @missing_components.setter
    def missing_components(self, value):
        if value not in ["pad", "copy", "ignore"]:
            raise ValueError(
                f"Unknown missing components strategy '{value}'. "
                f"Allowed values are 'pad', 'copy' and 'ignore'."
            )

        self._missing_components = value
        self.component_order = (
            self.component_order
        )  # Recalculate component order transfer arrays

    @property
    def component_order(self):
        """
        Get or set order of components in the output.
        """
        return self._component_order

    @component_order.setter
    def component_order(self, value):
        if value is None:
            value = seisbench.config["component_order"]

        if self.missing_components is not None:
            # In init, missing_components will be None in the first call,
            # but the component_mapping will be calculated when setting missing_components.
            self._component_mapping = {}
            for source_order in self.metadata["trace_component_order"].unique():
                self._component_mapping[source_order] = self._get_component_mapping(
                    source_order, value
                )

        self._component_order = value

    @property
    def chunks(self):
        """
        Returns a list of chunks. If dataset is not chunked, returns an empty list.
        """
        if self._chunks is None:
            self._chunks = self.available_chunks(self.path)
        return self._chunks

    @staticmethod
    def available_chunks(path):
        """
        Determines the chunks of the dataset in the given path.

        :param path: Dataset path
        :return: List of chunks
        """
        path = Path(path)
        chunks_path = path / "chunks"
        if chunks_path.is_file():
            with open(chunks_path, "r") as f:
                chunks = [x for x in f.read().split("\n") if x.strip()]

            if len(chunks) == 0:
                seisbench.logger.warning(
                    "Found empty chunks file. Using chunk detection from file names."
                )

        else:
            chunks = []

        # This control flow ensures that chunks are detected from file names also in case of an empty chunks file
        if len(chunks) == 0:
            if (path / "waveforms.hdf5").is_file():
                chunks = [""]
            else:
                metadata_files = set(
                    [
                        x.name[8:-4]
                        for x in path.iterdir()
                        if x.name.startswith("metadata") and x.name.endswith(".csv")
                    ]
                )
                waveform_files = set(
                    [
                        x.name[9:-5]
                        for x in path.iterdir()
                        if x.name.startswith("waveforms") and x.name.endswith(".hdf5")
                    ]
                )

                chunks = metadata_files & waveform_files

                if len(metadata_files - chunks) > 0:
                    seisbench.logger.warning(
                        f"Found metadata but no waveforms for chunks {metadata_files - chunks}"
                    )
                if len(waveform_files - chunks) > 0:
                    seisbench.logger.warning(
                        f"Found waveforms but no metadata for chunks {waveform_files - chunks}"
                    )

                if len(chunks) == 0:
                    raise FileNotFoundError(f"Did not find any chunks for the dataset")

                chunks = list(chunks)

        return sorted(chunks)

    def _unify_sampling_rate(self, eps=1e-4):
        """
        Unify sampling rate to common base. Operation is performed inplace. The sampling rate can be
        specified in three ways:

        - as trace_sampling_rate_hz in the metadata
        - as trace_dt_s in the metadata
        - as sampling_rate in the data format group, indicating identical sampling rate for all traces

        This function writes the sampling rate into trace_sampling_rate_hz in the metadata for unified
        access in the later processing. If the sampling rate is specified in multiple ways and inconsistencies
        are detected, a warning is logged.

        :param eps: floating precision, defaults to 1e-4.
        :type eps: float, optional
        :return: None
        """
        if "trace_sampling_rate_hz" in self.metadata.columns:
            if "trace_dt_s" in self.metadata.columns:
                if np.any(np.isnan(self.metadata["trace_sampling_rate_hz"].values)):
                    # Implace NaN values. Useful if for parts of data set sampling rate is specified, and for others dt
                    mask = np.isnan(self.metadata["trace_sampling_rate_hz"].values)
                    self._metadata["trace_sampling_rate_hz"].values[mask] = (
                        1 / self.metadata["trace_dt_s"].values[mask]
                    )

                q = (
                    self.metadata["trace_sampling_rate_hz"].values
                    * self.metadata["trace_dt_s"].values
                )
                if np.any(q < 1 - eps) or np.any(1 + eps < q):
                    seisbench.logger.warning(
                        "Inconsistent sampling rates in metadata. Using values from 'trace_sampling_rate_hz'."
                    )

            if "sampling_rate" in self.data_format:
                q = (
                    self.metadata["trace_sampling_rate_hz"].values
                    / self.data_format["sampling_rate"]
                )
                if np.any(q < 1 - eps) or np.any(1 + eps < q):
                    seisbench.logger.warning(
                        "Inconsistent sampling rates between metadata and data_format. Using values from metadata."
                    )

        elif "trace_dt_s" in self.metadata.columns:
            self.metadata["trace_sampling_rate_hz"] = 1 / self.metadata["trace_dt_s"]
            if "sampling_rate" in self.data_format:
                q = (
                    self.metadata["trace_sampling_rate_hz"].values
                    / self.data_format["sampling_rate"]
                )
                if np.any(q < 1 - eps) or np.any(1 + eps < q):
                    seisbench.logger.warning(
                        "Inconsistent sampling rates between metadata and data_format. Using values from metadata."
                    )

        elif "sampling_rate" in self.data_format:
            self._metadata["trace_sampling_rate_hz"] = self.data_format["sampling_rate"]

        else:
            seisbench.logger.warning("Sampling rate not specified in data set.")
            self._metadata["trace_sampling_rate_hz"] = np.nan

        if (
            "trace_sampling_rate_hz" in self.metadata.columns
            and np.any(np.isnan(self._metadata["trace_sampling_rate_hz"]))
            and not np.all(np.isnan(self._metadata["trace_sampling_rate_hz"]))
        ):
            seisbench.logger.warning("Found some traces with undefined sampling rates.")

        elif self.sampling_rate is None:
            sr = self.metadata["trace_sampling_rate_hz"].values
            q = sr / sr[0]
            if np.any(q < 1 - eps) or np.any(1 + eps < q):
                seisbench.logger.warning(
                    "Data set contains mixed sampling rate, but no sampling rate was specified for the dataset."
                    "get_waveforms will return mixed sampling rate waveforms."
                )

    def _get_component_mapping(self, source, target):
        """
        Calculates the mapping from source to target components while taking into account
        the missing_components setting.

        :param source:
        :param target:
        :return:
        """

        if (
            isinstance(source, float)
            and np.isnan(source)
            or (
                (isinstance(source, list) or isinstance(source, str))
                and not len(source)
            )
        ):
            raise ValueError(f"Component order not set for (parts of) the dataset.")

        source = list(source)
        target = list(target)

        mapping = []
        for t in target:
            if t in source:
                mapping.append(source.index(t))
            else:
                if self.missing_components == "pad":
                    mapping.append(len(source))  # Will be padded with zero later
                elif self.missing_components == "copy":
                    mapping.append(0)  # Use first component
                else:  # missing_components == "ignore"
                    pass

        return mapping

    @staticmethod
    def _get_dimension_mapping(source, target):
        """
        Calculates the mapping from source to target dimension orders.

        :param source:
        :param target:
        :return:
        """
        source = list(source)
        target = list(target)

        if len(target) != len(source):
            raise ValueError(
                f"Number of source and target components needs to be identical. Got {len(source)}!={len(target)}."
            )
        if len(set(source)) != len(source):
            raise ValueError(
                f"Source components/channels need to have unique names. Got {source}."
            )
        if len(set(target)) != len(target):
            raise ValueError(
                f"Target components/channels need to have unique names. Got {target}."
            )

        try:
            mapping = [source.index(t) for t in target]
        except ValueError:
            raise ValueError(
                f"Could not determine mapping {source} -> {target}. Please provide valid target components/channels."
            )

        return mapping

    def _chunks_with_paths(self):
        """
        See return value

        :return: List of chunks, list of metadata paths, list of waveform paths
        """
        metadata_paths = [self.path / f"metadata{chunk}.csv" for chunk in self.chunks]
        waveform_paths = [self.path / f"waveforms{chunk}.hdf5" for chunk in self.chunks]

        return self.chunks, metadata_paths, waveform_paths

    def _verify_dataset(self):
        """
        Checks that metadata and waveforms of all chunks are available and raises an exception otherwise.
        """
        for chunk, metadata_path, waveform_path in zip(*self._chunks_with_paths()):
            chunks_str = f" for chunk '{chunk}'" if chunk != "" else ""

            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing metadata file{chunks_str}")
            if not waveform_path.is_file():
                raise FileNotFoundError(f"Missing waveforms file{chunks_str}")

    def _read_data_format(self):
        """
        Reads the data format group from the hdf5 file(s).
        Checks consistency in case of chunked datasets.

        :return: Data format dict
        """

        data_format = None
        for waveform_file in self._chunks_with_paths()[2]:
            with h5py.File(waveform_file, "r") as f_wave:
                try:
                    g_data_format = f_wave["data_format"]
                    tmp_data_format = {
                        key: g_data_format[key][()] for key in g_data_format.keys()
                    }
                except KeyError:
                    seisbench.logger.warning(
                        "No data_format group found in .hdf5 File."
                    )
                    tmp_data_format = {}

            if data_format is None:
                data_format = tmp_data_format

            if not tmp_data_format == data_format:
                raise ValueError(f"Found inconsistent data format groups.")

        for key in data_format.keys():
            if isinstance(data_format[key], bytes):
                data_format[key] = data_format[key].decode()

        if "dimension_order" not in data_format:
            seisbench.logger.warning(
                "Dimension order not specified in data set. Assuming CW."
            )
            data_format["dimension_order"] = "CW"

        return data_format

    def _unify_component_order(self):
        """
        Unify different ways to pass component order, i.e., through data_format or metadata column
        trace_component_order. This function writes the component order into trace_component_order in the metadata
        for unified access in the later processing. If the component order is specified in multiple ways and
        inconsistencies are detected, a warning is logged.

        :return: None
        """
        if "component_order" in self.data_format:
            if "trace_component_order" in self.metadata.columns:
                if (
                    self.metadata["trace_component_order"]
                    != self.data_format["component_order"]
                ).any():
                    seisbench.logger.warning(
                        "Found inconsistent component orders between data format and metadata. "
                        "Using values from metadata."
                    )
            else:
                self._metadata["trace_component_order"] = self.data_format[
                    "component_order"
                ]
        else:
            if "trace_component_order" not in self.metadata.columns:
                seisbench.logger.warning(
                    "Component order not specified in data set. "
                    "Keeping original components."
                )

    def get_idx_from_trace_name(self, trace_name, chunk=None, dataset=None):
        """
        Returns the index of a trace with given trace_name, chunk and dataset.
        Chunk and dataset parameters are optional, but might be necessary to uniquely identify traces for
        chunked datasets or for :py:class:`MultiWaveformDataset`.
        The method will issue a warning *the first time* a non-uniquely identifiable trace is requested.
        If no matching key is found, a `KeyError` is raised.

        :param trace_name: Trace name as in metadata["trace_name"]
        :type trace_name: str
        :param chunk: Trace chunk as in metadata["trace_chunk"]. If None this key will be ignored.
        :type chunk: None
        :param dataset: Trace dataset as in metadata["trace_dataset"]. Only for :py:class:`MultiWaveformDataset`.
                        If None this key will be ignored.
        :type dataset: None
        :return: Index of the sample
        """
        dict_key = "name"
        search_key = [trace_name]
        if chunk is not None:
            dict_key += "_chunk"
            search_key.append(chunk)
        if dataset is not None:
            dict_key += "_dataset"
            search_key.append(dataset)

        search_key = tuple(search_key)

        if not self._trace_identification_warning_issued and len(
            self._trace_name_to_idx[dict_key]
        ) != len(self.metadata):
            seisbench.logger.warning(
                f'Traces can not uniformly be identified using {dict_key.replace("_", ", ")}. '
                '"get_idx_from_trace_name" will return only one possible matching trace.'
            )
            self._trace_identification_warning_issued = True

        if search_key in self._trace_name_to_idx[dict_key]:
            return self._trace_name_to_idx[dict_key][search_key]
        else:
            raise KeyError("The dataset does not contain the requested trace.")

    def _build_trace_name_to_idx_dict(self):
        """
        Builds mapping of trace_names to idx.
        """
        self._trace_name_to_idx = {}
        self._trace_name_to_idx["name"] = {
            (trace_name,): i for i, trace_name in enumerate(self.metadata["trace_name"])
        }
        self._trace_name_to_idx["name_chunk"] = {
            trace_info: i
            for i, trace_info in enumerate(
                zip(self.metadata["trace_name"], self.metadata["trace_chunk"])
            )
        }
        if "trace_dataset" in self.metadata.columns:
            self._trace_name_to_idx["name_dataset"] = {
                trace_info: i
                for i, trace_info in enumerate(
                    zip(self.metadata["trace_name"], self.metadata["trace_dataset"])
                )
            }
            self._trace_name_to_idx["name_chunk_dataset"] = {
                trace_info: i
                for i, trace_info in enumerate(
                    zip(
                        self.metadata["trace_name"],
                        self.metadata["trace_chunk"],
                        self.metadata["trace_dataset"],
                    )
                )
            }
        else:
            self._trace_name_to_idx["name_dataset"] = {}
            self._trace_name_to_idx["name_chunk_dataset"] = {}

        self._trace_identification_warning_issued = False

    def preload_waveforms(self, pbar=False):
        """
        Loads waveform data from hdf5 file into cache. Fails if caching strategy is None.

        :param pbar: If true, shows progress bar. Defaults to False.
        """
        if self.cache is None:
            seisbench.logger.warning("Skipping preload, as cache is disabled.")
            return

        chunks, metadata_paths, waveforms_path = self._chunks_with_paths()
        with LoadingContext(chunks, waveforms_path) as context:
            iterator = zip(self._metadata["trace_name"], self._metadata["trace_chunk"])
            if pbar:
                iterator = tqdm(
                    iterator, total=len(self._metadata), desc="Preloading waveforms"
                )

            for trace_name, chunk in iterator:
                self._get_single_waveform(trace_name, chunk, context=context)

    def filter(self, mask, inplace=True):
        """
        Filters dataset, e.g. by distance/magnitude/..., using a binary mask.
        Default behaviour is to perform inplace filtering, directly changing the
        metadata and waveforms to only keep the results of the masking query.
        Setting inplace equal to false will return a filtered copy of the data set.
        For details on the copy operation see :py:func:`~WaveformDataset.copy`.

        :param mask: Boolean mask to apply to metadata.
        :type mask: boolean array
        :param inplace: If true, filter inplace.
        :type inplace: bool

        Example usage:

        .. code:: python

            dataset.filter(dataset["p_status"] == "manual")

        :return: None if inplace=True, otherwise the filtered dataset.
        """
        if inplace:
            self._metadata = self._metadata[mask]
            self._evict_cache()
            self._build_trace_name_to_idx_dict()
        else:
            other = self.copy()
            other.filter(mask, inplace=True)
            return other

    # NOTE: lat/lon columns are specified to enhance generalisability as naming convention may
    # change between datasets and users may also want to filter as a function of  receivers/sources
    def region_filter(self, domain, lat_col, lon_col, inplace=True):
        """
        Filtering of dataset based on predefined region or geometry.
        See also convenience functions region_filter_[source|receiver].

        :param domain: The domain filter
        :type domain: obspy.core.fdsn.mass_downloader.domain:
        :param lat_col: Name of latitude coordinate column
        :type lat_col: str
        :param lon_col: Name of longitude coordinate column
        :type lon_col: str
        :param inplace: Inplace filtering, default to true. See also :py:func:`~WaveformDataset.filter`.
        :type inplace: bool
        :return: None if inplace=True, otherwise the filtered dataset.
        """

        def check_domain(metadata):
            return domain.is_in_domain(metadata[lat_col], metadata[lon_col])

        mask = self.metadata.apply(check_domain, axis=1)
        self.filter(mask, inplace=inplace)

    def region_filter_source(self, domain, inplace=True):
        """
        Convenience method for region filtering by source location.
        """
        self.region_filter(
            domain,
            lat_col="source_latitude_deg",
            lon_col="source_longitude_deg",
            inplace=inplace,
        )

    def region_filter_receiver(self, domain, inplace=True):
        """
        Convenience method for region filtering by receiver location.
        """
        self.region_filter(
            domain,
            lat_col="station_latitude_deg",
            lon_col="station_longitude_deg",
            inplace=inplace,
        )

    def get_split(self, split):
        """
        Returns a dataset with the requested split.

        :param split: Split name to return. Usually one of "train", "dev", "test"
        :return: Dataset filtered to the requested split.
        """
        if "split" not in self.metadata.columns:
            raise ValueError("Split requested but no split defined in metadata")

        mask = (self.metadata["split"] == split).values

        return self.filter(mask, inplace=False)

    def train(self):
        """
        Convenience method for get_split("train").

        :return: Training dataset
        """
        return self.get_split("train")

    def dev(self):
        """
        Convenience method for get_split("dev").

        :return: Development dataset
        """
        return self.get_split("dev")

    def test(self):
        """
        Convenience method for get_split("test").

        :return: Test dataset
        """
        return self.get_split("test")

    def train_dev_test(self):
        """
        Convenience method for returning training, development and test set. Equal to:

        >>> self.train(), self.dev(), self.test()

        :return: Training dataset, development dataset, test dataset
        """
        return self.train(), self.dev(), self.test()

    def _evict_cache(self):
        """
        Remove all traces from cache that do not have any reference in metadata anymore

        :return: None
        """
        existing_keys = defaultdict(set)
        if self.cache == "full":
            # Extract block names
            block_names = self._metadata["trace_name"].apply(lambda x: x.split("$")[0])
            chunks = self._metadata["trace_chunk"]
            for chunk, block in zip(chunks, block_names):
                existing_keys[chunk].add(block)
        elif self.cache == "trace":
            trace_names = self._metadata["trace_name"]
            chunks = self._metadata["trace_chunk"]
            for chunk, trace in zip(chunks, trace_names):
                existing_keys[chunk].add(trace)

        delete_count = 0
        for chunk in self._waveform_cache.keys():
            delete_keys = []
            for key in self._waveform_cache[chunk].keys():
                if key not in existing_keys[chunk]:
                    delete_keys.append(key)

            for key in delete_keys:
                del self._waveform_cache[chunk][key]

            delete_count += len(delete_keys)

        seisbench.logger.debug(f"Deleted {delete_count} entries in cache eviction")

    def __getitem__(self, item):
        """
        Only accepts string inputs. Returns respective column from metadata
        """
        if not isinstance(item, str):
            raise TypeError("Can only use strings to access metadata parameters")
        return self._metadata[item]

    def __len__(self):
        """
        Number of samples in the dataset.
        """
        return len(self._metadata)

    def get_sample(self, idx, sampling_rate=None):
        """
        Returns both waveforms and metadata of a traces.
        Adjusts all metadata traces with sampling rate dependent values to the correct sampling rate,
        e.g., p_pick_samples will still point to the right sample after this operation, even if the trace was resampled.

        :param idx: Idx of sample to return
        :param sampling_rate: Target sampling rate, overwrites sampling rate for dataset.
        :return: Tuple with the waveforms and the metadata of the sample.
        """
        metadata = self.metadata.iloc[idx].to_dict()

        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if sampling_rate is not None:
            source_sampling_rate = metadata["trace_sampling_rate_hz"]
            if np.isnan(source_sampling_rate):
                raise ValueError("Tried resampling trace with unknown sampling rate.")
            else:
                resampling_factor = sampling_rate / source_sampling_rate
                # Rewrite all metadata keys in sample units to the new sampling rate
                for key in metadata.keys():
                    if key.endswith("_sample"):
                        try:
                            metadata[key] = metadata[key] * resampling_factor
                        except TypeError:
                            seisbench.logger.info(
                                f"Failed to do sampling rate adjustment for column {key} "
                                f"due to type error. "
                            )

                metadata["trace_sampling_rate_hz"] = sampling_rate
                metadata["trace_dt_s"] = 1.0 / sampling_rate

        waveforms = self.get_waveforms(idx, sampling_rate=sampling_rate)

        # Find correct dimension, but ignore batch dimension as this will be squeezed in get_waveforms
        dimension_order = list(self.dimension_order)
        del dimension_order[dimension_order.index("N")]
        sample_dimension = dimension_order.index("W")
        metadata["trace_npts"] = waveforms.shape[sample_dimension]

        return waveforms, metadata

    def get_waveforms(self, idx=None, mask=None, sampling_rate=None):
        """
        Collects waveforms and returns them as an array.

        :param idx: Idx or list of idx to obtain waveforms for
        :type idx: int, list[int]
        :param mask: Binary mask on the metadata, indicating which traces should be returned.
                     Can not be used jointly with idx.
        :type mask: np.ndarray[bool]
        :param sampling_rate: Target sampling rate, overwrites sampling rate for dataset
        :type sampling_rate: float
        :return: Waveform array with dimensions ordered according to dimension_order e.g. default 'NCW'
                 (number of traces, number of components, record samples). If the number of record samples
                 varies between different entries, all entries are padded to the maximum length.
        :rtype: np.ndarray
        """
        squeeze = False
        if idx is not None:
            if mask is not None:
                raise ValueError("Mask can not be used jointly with idx.")
            if not isinstance(idx, Iterable):
                idx = [idx]
                squeeze = True

            load_metadata = self._metadata.iloc[idx]
        else:
            if mask is not None:
                load_metadata = self._metadata[mask]
            else:
                load_metadata = self._metadata

        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        waveforms = []
        chunks, metadata_paths, waveforms_path = self._chunks_with_paths()
        with LoadingContext(chunks, waveforms_path) as context:
            for trace_name, chunk, trace_sampling_rate, trace_component_order in zip(
                load_metadata["trace_name"],
                load_metadata["trace_chunk"],
                load_metadata["trace_sampling_rate_hz"],
                load_metadata["trace_component_order"],
            ):
                waveforms.append(
                    self._get_single_waveform(
                        trace_name,
                        chunk,
                        context=context,
                        target_sampling_rate=sampling_rate,
                        source_sampling_rate=trace_sampling_rate,
                        source_component_order=trace_component_order,
                    )
                )

        if self.missing_components == "ignore":
            # Check consistent number of components
            component_dimension = list(self._data_format["dimension_order"]).index("C")
            n_components = np.array([x.shape[component_dimension] for x in waveforms])
            if (n_components[0] != n_components).any():
                raise ValueError(
                    "Requested traces with mixed number of components. "
                    "Change missing_components or request traces separately."
                )

        waveforms = self._pad_packed_sequence(waveforms)

        # Impose correct dimension order
        waveforms = waveforms.transpose(*self._dimension_mapping)

        if squeeze:
            batch_dimension = list(self.dimension_order).index("N")
            waveforms = np.squeeze(waveforms, axis=batch_dimension)

        return waveforms

    def _get_single_waveform(
        self,
        trace_name,
        chunk,
        context,
        target_sampling_rate=None,
        source_sampling_rate=None,
        source_component_order=None,
    ):
        """
        Returns waveforms for single traces while handling caching and resampling.

        :param trace_name: Name of trace to load
        :param chunk: Chunk containing the trace
        :param context: A LoadingContext instance holding the pointers to the hdf5 files
        :param target_sampling_rate: Sampling rate for the output. If None keeps input sampling rate.
        :param source_sampling_rate: Sampling rate of the original trace
        :param source_component_order: Component order of the original trace
        :return: Array with trace waveforms at target_sampling_rate
        """
        trace_name = str(trace_name)

        if trace_name in self._waveform_cache[chunk]:
            # Cache hit on trace level
            waveform = self._waveform_cache[chunk][trace_name]

        else:
            # Cache miss on trace level
            if trace_name.find("$") != -1:
                # Trace is part of a block
                block_name, location = trace_name.split("$")
            else:
                # Trace is not part of a block
                block_name, location = trace_name, ":"

            location = self._parse_location(location)

            if block_name in self._waveform_cache[chunk]:
                # Cache hit on block level
                waveform = self._waveform_cache[chunk][block_name][location]

            else:
                # Cache miss on block level - Load from hdf5 file required
                g_data = context[chunk]["data"]
                block = g_data[block_name]
                if self.cache == "full":
                    block = block[()]  # Explicit load from hdf5 file
                    self._waveform_cache[chunk][block_name] = block
                    waveform = block[location]
                else:
                    waveform = block[location]  # Implies the load from hdf5 file
                    if self.cache == "trace":
                        self._waveform_cache[chunk][trace_name] = waveform

        if target_sampling_rate is not None:
            if np.isnan(source_sampling_rate):
                raise ValueError("Tried resampling trace with unknown sampling rate.")
            else:
                waveform = self._resample(
                    waveform, target_sampling_rate, source_sampling_rate
                )

        if source_component_order is not None:
            # Impose correct component order
            component_dimension = list(self._data_format["dimension_order"]).index("C")
            component_mapping = self._component_mapping[source_component_order]

            if waveform.shape[component_dimension] == max(component_mapping):
                # Add zero dimension used for padding
                pad = []
                for i in range(waveform.ndim):
                    if i == component_dimension:
                        pad += [(0, 1)]
                    else:
                        pad += [(0, 0)]
                waveform = np.pad(waveform, pad, "constant", constant_values=0)

            waveform = waveform.take(component_mapping, axis=component_dimension)

        return waveform

    @staticmethod
    def _parse_location(location):
        """
        Parses the location string and returns the associated tuple of slice objects
        :param location: Location identifier in numpy format as string, e.g., "5,1:7,:".
                         Allows omission of numbers and negative indexing, e.g., ":-5".
        :return: tuple of slice objects
        """
        location = location.replace(" ", "")  # Remove whitespace

        slices = []
        dim_slices = location.split(",")

        def int_or_none(s):
            if s == "":
                return None
            else:
                return int(s)

        for dim_slice in dim_slices:
            parts = dim_slice.split(":")
            if len(parts) == 1:
                idx = int_or_none(parts[0])
                slices.append(idx)
            elif len(parts) == 2:
                start = int_or_none(parts[0])
                stop = int_or_none(parts[1])
                slices.append(slice(start, stop))
            elif len(parts) == 3:
                start = int_or_none(parts[0])
                stop = int_or_none(parts[1])
                step = int_or_none(parts[2])
                slices.append(slice(start, stop, step))
            else:
                raise ValueError(f"Invalid location string {location}")

        return tuple(slices)

    def _resample(self, waveform, target_sampling_rate, source_sampling_rate, eps=1e-4):
        """
        Resamples waveform from source to target sampling rate.
        Automatically chooses between scipy.signal.decimate and scipy.signal.resample
        based on source and target sampling rate.

        :param waveform:
        :param target_sampling_rate:
        :param source_sampling_rate:
        :param eps: Tolerance for equality of source an target sampling rate
        :return:
        """
        try:
            sample_axis = list(self._data_format["dimension_order"]).index("W")
        except KeyError:
            # Dimension order not specified
            sample_axis = None
        except ValueError:
            # W not in dimension order
            sample_axis = None

        if 1 - eps < target_sampling_rate / source_sampling_rate < 1 + eps:
            return waveform
        else:
            if sample_axis is None:
                raise ValueError(
                    "Trace can not be resampled because of missing or incorrect dimension order."
                )

            if waveform.shape[sample_axis] == 0:
                seisbench.logger.info(
                    "Trying to resample empty trace, skipping resampling."
                )
                return waveform

            if (source_sampling_rate % target_sampling_rate) < eps:
                q = int(source_sampling_rate // target_sampling_rate)
                return scipy.signal.decimate(waveform, q, axis=sample_axis)
            else:
                num = int(
                    waveform.shape[sample_axis]
                    * target_sampling_rate
                    / source_sampling_rate
                )
                return scipy.signal.resample(waveform, num, axis=sample_axis)

    @staticmethod
    def _pad_packed_sequence(seq):
        """
        Packs a list of arrays into one array by adding a new first dimension and padding where necessary.

        :param seq:
        :return:
        """
        max_size = np.array(
            [max([x.shape[i] for x in seq]) for i in range(seq[0].ndim)]
        )

        new_seq = []
        for i, elem in enumerate(seq):
            d = max_size - np.array(elem.shape)
            if (d != 0).any():
                pad = [(0, d_dim) for d_dim in d]
                new_seq.append(np.pad(elem, pad, "constant", constant_values=0))
            else:
                new_seq.append(elem)

        return np.stack(new_seq, axis=0)

    def plot_map(self, res="110m", connections=False, **kwargs):
        """
        Plots the dataset onto a map using the Mercator projection. Requires a cartopy installation.

        :param res: Resolution for cartopy features, defaults to 110m.
        :type res: str, optional
        :param connections: If true, plots lines connecting sources and stations. Defaults to false.
        :type connections: bool, optional
        :param kwargs: Plotting kwargs that will be passed to matplotlib plot. Args need to be prefixed with
                       `sta_`, `ev_` and `conn_` to address stations, events or connections.
        :return: A figure handle for the created figure.
        """
        fig = plt.figure(figsize=(15, 10))
        try:
            import cartopy.crs as ccrs
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            import cartopy.feature as cfeature
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Plotting the data set requires cartopy. "
                "Please install cartopy, e.g., using conda."
            )

        ax = fig.add_subplot(111, projection=ccrs.Mercator())

        ax.coastlines(res)
        land_50m = cfeature.NaturalEarthFeature(
            "physical", "land", res, edgecolor="face", facecolor=cfeature.COLORS["land"]
        )
        ax.add_feature(land_50m)

        def prefix_dict(kws, prefix):
            return {
                k[len(prefix) :]: v
                for k, v in kws.items()
                if k[: len(prefix)] == prefix
            }

        lines_kws = {
            "marker": "",
            "linestyle": "-",
            "color": "grey",
            "alpha": 0.5,
            "linewidth": 0.5,
        }
        lines_kws.update(prefix_dict(kwargs, "conn_"))

        station_kws = {"marker": "^", "color": "k", "linestyle": "", "ms": 10}
        station_kws.update(prefix_dict(kwargs, "sta_"))

        event_kws = {"marker": ".", "color": "r", "linestyle": ""}
        event_kws.update(prefix_dict(kwargs, "ev_"))

        # Plot connecting lines
        if connections:
            station_source_pairs = self.metadata[
                [
                    "station_longitude_deg",
                    "station_latitude_deg",
                    "source_longitude_deg",
                    "source_latitude_deg",
                ]
            ].values
            for row in station_source_pairs:
                ax.plot(
                    [row[0], row[2]],
                    [row[1], row[3]],
                    transform=ccrs.Geodetic(),
                    **lines_kws,
                )

        # Plot stations
        station_locations = np.unique(
            self.metadata[["station_longitude_deg", "station_latitude_deg"]].values,
            axis=0,
        )
        ax.plot(
            station_locations[:, 0],
            station_locations[:, 1],
            transform=ccrs.PlateCarree(),
            **station_kws,
        )

        # Plot events
        source_locations = np.unique(
            self.metadata[["source_longitude_deg", "source_latitude_deg"]].values,
            axis=0,
        )
        ax.plot(
            source_locations[:, 0],
            source_locations[:, 1],
            transform=ccrs.PlateCarree(),
            **event_kws,
        )

        # Gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.top_labels = False
        gl.left_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        fig.suptitle(self.name)

        return fig


class MultiWaveformDataset:
    """
    A :py:class:`MultiWaveformDataset` is an ordered collection of :py:class:`WaveformDataset`.
    It exposes mostly the same API as a single :py:class:`WaveformDataset`.

    The constructor checks for compatibility of `dimension_order`, `component_order` and `sampling_rate`.
    The caching strategy of each contained dataset is left unmodified,
    but a warning is issued if different caching schemes are found.

    :param datasets: List of :py:class:`WaveformDataset`.
                     The constructor will create a copy of each dataset using the
                     :py:func:`WaveformDataset.copy` method.
    """

    def __init__(self, datasets):
        if not isinstance(datasets, list) or not all(
            isinstance(x, WaveformDataset) for x in datasets
        ):
            raise TypeError(
                "MultiWaveformDataset expects a list of WaveformDataset as input."
            )

        if len(datasets) == 0:
            raise ValueError("MultiWaveformDatasets need to have at least one member.")

        self._datasets = [dataset.copy() for dataset in datasets]
        self._metadata = pd.concat(x.metadata for x in datasets)

        # Identify dataset
        self._metadata["trace_dataset"] = sum(
            ([dataset.name] * len(dataset) for dataset in self.datasets), []
        )

        self._trace_identification_warning_issued = (
            False  # Traced whether warning for trace name was issued already
        )

        self._homogenize_dataformat(datasets)
        self._build_trace_name_to_idx_dict()

    def __add__(self, other):
        if isinstance(other, WaveformDataset):
            return MultiWaveformDataset(self.datasets + [other])
        elif isinstance(other, MultiWaveformDataset):
            return MultiWaveformDataset(self.datasets + other.datasets)
        else:
            raise TypeError(
                "Can only add WaveformDataset and MultiWaveformDataset to MultiWaveformDataset."
            )

    def _homogenize_dataformat(self, datasets):
        """
        Checks if the output data format options agree.
        In case of mismatches, warnings are issued and the format is reset.
        """
        has_split = ["split" in dataset.metadata.columns for dataset in datasets]
        if (
            np.sum(has_split) % len(datasets) != 0
        ):  # Check if all or no dataset has a split
            seisbench.logger.warning(
                "Combining datasets with and without split. "
                "get_split and all derived methods will never return any samples from "
                "the datasets without split."
            )
        if not self._test_attribute_equal(datasets, "cache"):
            seisbench.logger.warning(
                "Found inconsistent caching strategies. "
                "This does not cause an error, but is usually unintended."
            )
        if not self._test_attribute_equal(datasets, "sampling_rate"):
            seisbench.logger.warning(
                "Found mismatching sampling rates between datasets. "
                "Setting sampling rate to None, i.e., deactivating automatic resampling. "
                "You can change the sampling rate for all datasets through "
                "the sampling_rate property."
            )
            self.sampling_rate = None

        if self.sampling_rate is None and len(self) > 0:
            sr = self.metadata["trace_sampling_rate_hz"].values
            q = sr / sr[0]
            if not np.allclose(q, 1):
                seisbench.logger.warning(
                    "Data set contains mixed sampling rate, but no sampling rate was specified for the dataset."
                    "get_waveforms will return mixed sampling rate waveforms."
                )

        if not self._test_attribute_equal(datasets, "component_order"):
            seisbench.logger.warning(
                "Found inconsistent component orders. "
                f"Using component order from first dataset ({self.datasets[0].component_order})."
            )
            self.component_order = self.datasets[0].component_order

        if not self._test_attribute_equal(datasets, "dimension_order"):
            seisbench.logger.warning(
                "Found inconsistent dimension orders. "
                f"Using dimension order from first dataset ({self.datasets[0].dimension_order})."
            )
            self.dimension_order = self.datasets[0].dimension_order

        if not self._test_attribute_equal(datasets, "missing_components"):
            seisbench.logger.warning(
                "Found inconsistent missing_components. "
                f"Using missing_components from first dataset ({self.datasets[0].missing_components})."
            )
            self.missing_components = self.datasets[0].missing_components

    @property
    def datasets(self):
        """
        Datasets contained in MultiWaveformDataset.
        """
        return list(self._datasets)

    @property
    def metadata(self):
        """
        Metadata of the dataset as pandas DataFrame.
        """
        return self._metadata

    @property
    def sampling_rate(self):
        """
        Get or set sampling rate for output
        """
        return self.datasets[0].sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        for dataset in self.datasets:
            dataset.sampling_rate = sampling_rate

    @property
    def dimension_order(self):
        """
        Get or set dimension order for output
        """
        return self.datasets[0].dimension_order

    @dimension_order.setter
    def dimension_order(self, value):
        for dataset in self.datasets:
            dataset.dimension_order = value

    @property
    def missing_components(self):
        """
        Get or set strategy for missing components
        """
        return self.datasets[0].missing_components

    @missing_components.setter
    def missing_components(self, value):
        for dataset in self.datasets:
            dataset.missing_components = value

    @property
    def component_order(self):
        """
        Get or set component order
        """
        return self.datasets[0].component_order

    @component_order.setter
    def component_order(self, value):
        for dataset in self.datasets:
            dataset.component_order = value

    @property
    def cache(self):
        """
        Get or set cache strategy
        """
        if self._test_attribute_equal(self.datasets, "cache"):
            return self.datasets[0].cache
        else:
            return "Inconsistent"

    @cache.setter
    def cache(self, value):
        for dataset in self.datasets:
            dataset.cache = value

    @staticmethod
    def _test_attribute_equal(datasets, attribute):
        """
        Checks whether the given attribute is equal in all datasets.

        :param datasets: List of WaveformDatasets
        :param attribute: attribute as string
        :return: True if attribute is identical in all datasets, false otherwise
        """
        attribute_list = [dataset.__getattribute__(attribute) for dataset in datasets]
        return all(x == attribute_list[0] for x in attribute_list)

    def __getitem__(self, item):
        """
        Only accepts string inputs. Returns respective column from metadata
        """
        if not isinstance(item, str):
            raise TypeError("Can only use strings to access metadata parameters")
        return self.metadata[item]

    def __len__(self):
        return len(self.metadata)

    def get_sample(self, idx, *args, **kwargs):
        """
        Wraps :py:func:`WaveformDataset.get_sample`

        :param idx: Index of the sample
        :param args: passed to parent function
        :param kwargs: passed to parent function
        :return: Return value of parent function
        """
        dataset_idx, local_idx = self._resolve_idx(idx)
        return self.datasets[dataset_idx].get_sample(local_idx, *args, **kwargs)

    def get_waveforms(self, idx=None, mask=None, **kwargs):
        """
        Collects waveforms and returns them as an array.

        :param idx: Idx or list of idx to obtain waveforms for
        :type idx: int, list[int]
        :param mask: Binary mask on the metadata, indicating which traces should be returned.
                     Can not be used jointly with idx.
        :type mask: np.ndarray[bool]
        :param kwargs: Passed to :py:func:`WaveformDataset.get_waveforms`
        :return: Waveform array with dimensions ordered according to dimension_order e.g. default 'NCW'
                 (number of traces, number of components, record samples). If the number record samples
                 varies between different entries, all entries are padded to the maximum length.
        :rtype: np.ndarray
        """
        squeeze = False
        waveforms = []
        if idx is not None:
            if mask is not None:
                raise ValueError("Mask can not be used jointly with idx.")
            if not isinstance(idx, Iterable):
                idx = [idx]
                squeeze = True

            for i in idx:
                dataset_idx, local_idx = self._resolve_idx(i)
                waveforms.append(
                    self.datasets[dataset_idx].get_waveforms(idx=[local_idx], **kwargs)
                )
        else:
            if mask is None:
                mask = np.ones(len(self), dtype=bool)

            submasks = self._split_mask(mask)

            for submask, dataset in zip(submasks, self.datasets):
                if submask.any():
                    waveforms.append(dataset.get_waveforms(mask=submask, **kwargs))

        if self.missing_components == "ignore":
            # Check consistent number of components
            component_dimension = list(self.dimension_order).index("C")
            n_components = np.array([x.shape[component_dimension] for x in waveforms])
            if (n_components[0] != n_components).any():
                raise ValueError(
                    "Requested traces with mixed number of components. "
                    "Change missing_components or request traces separately."
                )

        batch_dimension = list(self.dimension_order).index("N")
        waveforms = self._pad_pack_along_axis(waveforms, axis=batch_dimension)

        if squeeze:
            waveforms = np.squeeze(waveforms, axis=batch_dimension)

        return waveforms

    @staticmethod
    def _pad_pack_along_axis(seq, axis):
        """
        Concatenate arrays along axis. In each but the given axis, all input arrays are padded with zeros
        to the maximum size of any input array.

        :param seq: List of arrays
        :param axis: Axis along which to concatenate
        :return:
        """
        max_size = np.array(
            [max([x.shape[i] for x in seq]) for i in range(seq[0].ndim)]
        )

        new_seq = []
        for i, elem in enumerate(seq):
            d = max_size - np.array(elem.shape)
            if (d != 0).any():
                pad = [(0, d_dim) for d_dim in d]
                pad[axis] = (0, 0)
                new_seq.append(np.pad(elem, pad, "constant", constant_values=0))
            else:
                new_seq.append(elem)

        return np.concatenate(new_seq, axis=axis)

    def filter(self, mask, inplace=True):
        """
        Filters dataset, similar to :py:func:`WaveformDataset.filter`.

        :param mask: Boolean mask to apple to metadata.
        :type mask: masked-array
        :param inplace: If true, filter inplace.
        :type inplace: bool
        :return: None if filter=true, otherwise the filtered dataset.
        """
        submasks = self._split_mask(mask)
        if inplace:
            for submask, dataset in zip(submasks, self.datasets):
                dataset.filter(submask, inplace=True)
            # Calculate new metadata
            self._metadata = pd.concat(x.metadata for x in self.datasets)
            self._build_trace_name_to_idx_dict()

        else:
            return MultiWaveformDataset(
                [
                    dataset.filter(submask, inplace=False)
                    for submask, dataset in zip(submasks, self.datasets)
                ]
            )

    def _resolve_idx(self, idx):
        """
        Translates an index into the dataset index and the index within the dataset

        :param idx: Index of the sample
        :return: Dataset index, index within the dataset
        """
        borders = np.cumsum([len(x) for x in self.datasets])
        if idx < 0:
            idx += len(self)

        if idx >= len(self) or idx < 0:
            raise IndexError("Sample index out out range.")

        dataset_idx = np.argmax(idx < borders)
        local_idx = (
            idx - borders[dataset_idx] + len(self.datasets[dataset_idx])
        )  # Resolve the negative indexing

        return dataset_idx, local_idx

    def _split_mask(self, mask):
        """
        Split one mask for the full dataset into several masks for each subset

        :param mask: Mask for the full dataset
        :return: List of masks, one for each dataset
        """
        if not len(mask) == len(self):
            raise ValueError("Mask does not match dataset.")

        masks = []
        p = 0
        for dataset in self.datasets:
            masks.append(mask[p : p + len(dataset)])
            p += len(dataset)

        return masks

    def preload_waveforms(self, *args, **kwargs):
        """
        Calls :py:func:`WaveformDataset.preload_waveforms` for all member datasets with the provided arguments.
        """
        for dataset in self.datasets:
            dataset.preload_waveforms(*args, **kwargs)

    # Copy compatible parts from WaveformDataset
    region_filter = WaveformDataset.region_filter
    region_filter_source = WaveformDataset.region_filter_source
    region_filter_receiver = WaveformDataset.region_filter_receiver
    plot_map = WaveformDataset.plot_map
    get_split = WaveformDataset.get_split
    train = WaveformDataset.train
    dev = WaveformDataset.dev
    test = WaveformDataset.test
    train_dev_test = WaveformDataset.train_dev_test
    _build_trace_name_to_idx_dict = WaveformDataset._build_trace_name_to_idx_dict
    get_idx_from_trace_name = WaveformDataset.get_idx_from_trace_name


class LoadingContext:
    """
    The LoadingContext is a dict of pointers to the hdf5 files for the chunks.
    It is an easy way to manage opening and closing of file pointers when required.
    """

    def __init__(self, chunks, waveform_paths):
        self.chunk_dict = {
            chunk: waveform_path for chunk, waveform_path in zip(chunks, waveform_paths)
        }
        self.file_pointers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for file in self.file_pointers.values():
            file.close()
        self.file_pointers = {}

    def __getitem__(self, chunk):
        if chunk not in self.chunk_dict:
            raise KeyError(f'Unknown chunk "{chunk}"')

        if chunk not in self.file_pointers:
            self.file_pointers[chunk] = h5py.File(self.chunk_dict[chunk], "r")
        return self.file_pointers[chunk]


class BenchmarkDataset(WaveformDataset, ABC):
    """
    This class is the base class for benchmark waveform datasets.
    It adds functionality to automatically download the dataset to the SeisBench cache.
    Downloads can either be from the SeisBench repository, if the dataset is available there and in the right format,
    or from another source, which will usually require some form of conversion.
    Furthermore, it adds annotations for citation and license.

    :param chunks: List of chunks to download
    :param citation: Citation for the dataset. Should be set in the inheriting class.
    :param license: License associated with the dataset. Should be set in the inheriting class.
    :param force: Passed to :py:func:`~seisbench.util.callback_if_uncached`
    :param wait_for_file: Passed to :py:func:`~seisbench.util.callback_if_uncached`
    :param repository_lookup: Whether the data set should be search in the remote repository or directly use
                              the download function. Should be set in the inheriting class. Only needs to be
                              set to true if the dataset is available in a repository, e.g., the SeisBench
                              repository, for direct download.
    :param download_kwargs: Dict of arguments passed to the download_dataset function,
                            in case the dataset is loaded from scratch.
    :param kwargs: Keyword arguments passed to WaveformDataset
    """

    def __init__(
        self,
        chunks=None,
        citation=None,
        license=None,
        force=False,
        wait_for_file=False,
        repository_lookup=False,
        download_kwargs=None,
        **kwargs,
    ):
        self._name = self._name_internal()
        self._citation = citation
        self._license = license
        self.path.mkdir(exist_ok=True, parents=True)

        if download_kwargs is None:
            download_kwargs = {}

        if chunks is None:
            chunks = self.available_chunks(force=force, wait_for_file=wait_for_file)

        for chunk in chunks:

            def download_callback(files):
                chunk_str = f'Chunk "{chunk}" of ' if chunk != "" else ""
                seisbench.logger.warning(
                    f"{chunk_str}Dataset {self.name} not in cache."
                )
                successful_repository_download = False
                if repository_lookup:
                    seisbench.logger.warning(
                        "Trying to download preprocessed version from SeisBench repository."
                    )
                    try:
                        self._download_preprocessed(*files, chunk=chunk)
                        successful_repository_download = True
                    except ValueError:
                        pass

                if not successful_repository_download:
                    seisbench.logger.warning(
                        f"{chunk_str}Dataset {self.name} not in SeisBench repository. "
                        f"Starting download and conversion from source."
                    )

                    download_dataset_parameters = inspect.signature(
                        self._download_dataset
                    ).parameters
                    if "chunk" not in download_dataset_parameters and chunk != "":
                        raise ValueError(
                            "Data set seems not to support chunking, but chunk provided."
                        )

                    # Pass chunk parameter to _download_dataset through download_kwargs
                    tmp_download_args = copy.copy(download_kwargs)
                    if "chunk" in download_dataset_parameters:
                        tmp_download_args["chunk"] = chunk

                    with WaveformDataWriter(*files) as writer:
                        self._download_dataset(writer, **tmp_download_args)

            files = [
                self.path / f"metadata{chunk}.csv",
                self.path / f"waveforms{chunk}.hdf5",
            ]
            seisbench.util.callback_if_uncached(
                files, download_callback, force=force, wait_for_file=wait_for_file
            )

        super().__init__(path=None, name=self._name_internal(), chunks=chunks, **kwargs)

    @property
    def citation(self):
        """
        The suggested citation for this dataset
        """
        return self._citation

    @property
    def license(self):
        """
        The licence attached to this dataset
        """
        return self._license

    @classmethod
    def _path_internal(cls):
        """
        Path to the dataset location in the SeisBench cache. This class method is required for technical reasons.
        """
        return Path(seisbench.cache_root, "datasets", cls._name_internal().lower())

    @property
    def path(self):
        """
        Path to the dataset location in the SeisBench cache
        """
        return self._path_internal()

    @classmethod
    def _name_internal(cls):
        """
        Name of the dataset. This class method is required for technical reasons.
        """
        return cls.__name__

    @property
    def name(self):
        """
        Name of the dataset. For BenchmarkDatasets, always matches the class name.
        """
        return self._name_internal()

    @classmethod
    def _remote_path(cls):
        """
        Path within the remote repository. Does only generate the pass without checking actual availability.
        Can be overwritten for datasets stored in the correct format but at a different location.
        """
        return os.path.join(
            seisbench.remote_root, "datasets", cls._name_internal().lower()
        )

    @classmethod
    def available_chunks(cls, force=False, wait_for_file=False):
        """
        Returns a list of available chunks. Queries both the local cache and the remote root.
        """
        if (cls._path_internal() / "metadata.csv").is_file() and (
            cls._path_internal() / "waveforms.hdf5"
        ).is_file():
            # If the data set is not chunked, do not search for a chunk file.
            chunks = [""]
        else:
            # Search for chunk file in cache or remote repository.
            # This is necessary, because otherwise it is unclear if datasets have been downloaded completely.
            def chunks_callback(file):
                remote_chunks_path = os.path.join(cls._remote_path(), "chunks")
                try:
                    seisbench.util.download_http(
                        remote_chunks_path, file, progress_bar=False, precheck_timeout=0
                    )
                except ValueError:
                    seisbench.logger.info("Found no remote chunk file. Progressing.")

            chunks_path = cls._path_internal() / "chunks"
            seisbench.util.callback_if_uncached(
                chunks_path,
                chunks_callback,
                force=force,
                wait_for_file=wait_for_file,
            )

            if chunks_path.is_file():
                with open(chunks_path, "r") as f:
                    chunks = [x for x in f.read().split("\n") if x.strip()]
            else:
                # Assume file is not chunked.
                # To write the conversion for a chunked file, simply write the chunks file before calling
                # the super constructor.
                chunks = [""]

        return chunks

    @staticmethod
    def _sample_without_replacement(indexes, n_samples):
        """
        Sample indexes without replacement.

        :param indexes: indexes to sample.
        :type indexes: array-like
        :param n_samples: The number of samples to choose.
        :type n_samples: int.
        :return (chosen_idxs, complement_idxs): The chosen indexes and remaining indexes.
        :rtype: (list, list)

        """
        chosen_idxs = np.random.choice(indexes, n_samples, replace=False)
        complement_idxs = list(set(indexes).symmetric_difference(chosen_idxs))
        return chosen_idxs, complement_idxs

    def _set_splits_random_sampling(self, ratios=(0.6, 0.1, 0.3), random_seed=42):
        """
        Set train/dev/test split randomly, labelling examples from dataset.
        The number of random choices for each split is pre-defined with parameter ratios,
        choosing the proportion of train/dev/test samples to select respectively.

        :param ratios: train/dev/test ratio respectively, defaults to (0.6, 0.1, 0.3)
        :type ratios: (float, float, float)
        :param random_seed: Set the random seed, defaults to 42.
        :type random_seed: int
        :return:

        """
        np.random.seed(seed=random_seed)

        assert (
            len(ratios) == 3
        ), f"Only train/dev/test ratios should be specified. Got {len(ratios)} ratios."

        train_ratio, dev_ratio, test_ratio = ratios

        n_train_samples = int(self.__len__() * train_ratio)
        n_dev_samples = int(self.__len__() * dev_ratio)
        n_test_samples = self.__len__() - n_dev_samples - n_train_samples

        assert (
            self.__len__() == n_train_samples + n_dev_samples + n_test_samples
        ), "`ratios` must sum to 1"

        train_idxs, dev_and_test_idxs = self._sample_without_replacement(
            indexes=np.arange(self.__len__()), n_samples=n_train_samples
        )
        dev_idxs, test_idxs = self._sample_without_replacement(
            indexes=dev_and_test_idxs, n_samples=n_dev_samples
        )

        self.metadata.loc[train_idxs, "split"] = "train"
        self.metadata.loc[dev_idxs, "split"] = "dev"
        self.metadata.loc[test_idxs, "split"] = "test"

    def _download_preprocessed(self, metadata_path, waveforms_path, chunk):
        """
        Downloads the dataset in the correct format, usually from the remote root.
        """
        self.path.mkdir(parents=True, exist_ok=True)

        remote_path = self._remote_path()
        remote_metadata_path = os.path.join(remote_path, f"metadata{chunk}.csv")
        remote_waveforms_path = os.path.join(remote_path, f"waveforms{chunk}.hdf5")

        seisbench.util.download_http(
            remote_metadata_path, metadata_path, desc="Downloading metadata"
        )
        seisbench.util.download_http(
            remote_waveforms_path, waveforms_path, desc="Downloading waveforms"
        )

    @abstractmethod
    def _download_dataset(self, writer, chunk, **kwargs):
        """
        Download and convert the dataset to the standard SeisBench format.
        The metadata must contain at least the column 'trace_name'.
        Please see the SeisBench documentation for more details on the data format.

        :param writer: A WaveformDataWriter instance
        :param chunk: The chunk to be downloaded. Can be ignored if unchunked data set is created.
        :param kwargs:
        :return: None
        """
        pass


class Bucketer(ABC):
    """
    This is the abstract bucketer class that needs to be provided to the WaveformDataWriter.
    It offers one public function, :py:func:`get_bucket`, to assign a bucket to each trace.
    """

    @abstractmethod
    def get_bucket(self, metadata, waveform):
        """
        Calculates the bucket for the trace given its metadata and waveforms

        :param metadata: Metadata as given to the WaveformDataWriter.
        :param waveform: Waveforms as given to the WaveformDataWriter.
        :return: A hashable object denoting the bucket this sample belongs to.
        """
        return ""


class GeometricBucketer(Bucketer):
    """
    A simple bucketer that uses the length of the traces and optionally the assigned split to determine buckets.
    Only takes into account the length along one fixed axis.
    Bucket edges are create with a geometric spacing above a minimum bucket.
    The first bucket is [0, minbucket), the second one [minbucket, minbucket * factor) and so on.
    There is no maximum bucket.
    This bucketer ensures that the overhead from padding is at most factor - 1, as long as only few traces with
    length < minbucket exist.
    Note that this can even be significantly reduced by passing the input traces ordered by their length.

    :param minbucket: Upper limit of the lowest bucket and start of the geometric spacing.
    :type minbucket: int
    :param factor: Factor for the geometric spacing.
    :type factor: float
    :param splits: If true, returns separate buckets for each split. Defaults to true.
                   If no split is defined in the metadata, this parameter is ignored.
    :type splits: bool
    :param track_channels: If true, uses the shape of the input waveform along all axis except the one defined in axis,
                           to determine the bucket. Only traces agreeing in all dimensions except the given axis will be
                           assigned to the same bucket.
    :type track_channels: bool
    :param axis: Axis to take into account for determining the length of the trace.
    :type axis: int
    """

    def __init__(
        self, minbucket=100, factor=1.2, splits=True, track_channels=True, axis=-1
    ):
        self.minbucket = minbucket
        self.factor = factor
        self.split = splits
        self.track_channels = track_channels
        self.axis = axis

    def get_bucket(self, metadata, waveform):
        length = waveform.shape[self.axis]
        if self.track_channels:
            shape = list(waveform.shape)
            del shape[self.axis]  # Ignore sample axis
            shape = [str(x) for x in shape]
            channel_str = f"({','.join(shape)})_"
        else:
            channel_str = ""

        if self.split and "split" in metadata:
            split_str = str(metadata["split"])
        else:
            split_str = ""

        if length < self.minbucket:
            bucket_id = 0
        else:
            bucket_id = int(np.log(length / self.minbucket) / np.log(self.factor) + 1)

        return split_str + channel_str + str(bucket_id)


class WaveformDataWriter:
    """
    The WaveformDataWriter for writing datasets in SeisBench format.

    To improve reading performance when using the datasets, the writer groups traces into blocks and writes them into
    joint arrays in the hdf5 file. The exact behaviour is controlled by the :py:attr:`bucketer` and
    the :py:attr:`bucket_size`. For details see their documentation. This packing is necessary, due to limitations
    in the hdf5 performance. Reading many small datasets from a hdf5 file causes the overhead of the hdf5 structure
    to define the read times.

    :param metadata_path: Path to write the metadata file to
    :type metadata_path: str or Path
    :param waveforms_path: Path to write the waveforms file to
    :type waveforms_path: str or Path
    :return: None
    """

    def __init__(self, metadata_path, waveforms_path):
        self.metadata_path = Path(metadata_path)
        self.waveforms_path = Path(waveforms_path)
        self.metadata_dict = {}
        self.data_format = {}

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.waveforms_path.parent.mkdir(parents=True, exist_ok=True)

        self._metadata = []
        self._waveform_file = None
        self._pbar = None

        self._bucketer = GeometricBucketer()
        self._bucket_size = 1024
        self._cache = defaultdict(list)  # Cache used for the bucketing
        self._bucket_counter = 0  # Bucket counter to generate bucket names

    @property
    def bucketer(self):
        """
        The currently used bucketer, which sorts traces into buckets.
        If the bucketer is None, no buckets are used and all traces are written separately.
        By default uses the :py:class:`GeometricBucketer` with default parameters.
        Please check that this suits your needs.
        In particular, make sure that the default axis matches your sample axis.

        :return: Returns the current bucketer.
        """
        return self._bucketer

    @bucketer.setter
    def bucketer(self, value):
        if not isinstance(value, Bucketer) and value is not None:
            raise TypeError("The bucketer needs to be an instance of Bucketer or None.")

        self._bucketer = value

    @property
    def bucket_size(self):
        """
        The maximum size of a bucket.
        Once adding another trace would overload the bucket, the bucket is written to disk.
        Defaults to 1024.

        :return: Bucket size
        """
        return self._bucket_size

    @bucket_size.setter
    def bucket_size(self, value):
        if value < 1:
            raise ValueError("Bucket size needs to be at least one")

        self._bucket_size = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize()
        if self._waveform_file is not None:
            self._waveform_file.close()
        if self._pbar is not None:
            self._pbar.close()

        if exc_type is None:
            return True
        else:
            seisbench.logger.error(
                f"Error in downloading dataset. "
                f"Saved current progress to {self.metadata_path} and {self.waveforms_path}. Error message:\n"
            )

    def add_trace(self, metadata, waveform):
        """
        Adds a trace to the writer. This does not imply that the trace is immediately written to disk, as the writer
        might wait to fill a bucket. The writer ensures that the order of traces in the metadata is identical to the
        order of calls to add_trace.

        :param metadata: Metadata of the trace
        :type metadata: dict[str, any]
        :param waveform: Waveform of the trace
        :type waveform: np.ndarray
        :return: None
        """
        self._metadata.append(
            metadata
        )  # Note that this is only a reference to the metadata. This is later used to modify the trace_name attribute.

        if self.bucketer is None:
            self._write_bucket([metadata, waveform])
        else:
            bucket = self.bucketer.get_bucket(metadata, waveform)
            self._cache[bucket].append((metadata, waveform))

            if len(self._cache[bucket]) + 1 > self.bucket_size:
                self._write_bucket(self._cache[bucket])  # Write out bucket
                self._cache[bucket] = []  # Remove bucket from cache

        if self._pbar is None:
            self._pbar = tqdm(desc="Traces converted")

        self._pbar.update()

    def _write_bucket(self, bucket):
        """
        Writes a bucket to the waveforms file
        """
        if len(bucket) == 0:
            # Empty buckets don't need to be written out
            return

        if self._waveform_file is None:
            self._waveform_file = h5py.File(self.waveforms_path, "w")
            self._waveform_file.create_group("data")

        if len(bucket) == 1:
            metadata, waveform = bucket[0]
            if "trace_name" in metadata:
                # Use trace name as bucket name
                trace_name = str(metadata["trace_name"])
                trace_name = trace_name.replace(
                    "$", "_"
                )  # As $ will be interpreted as a control sequence to remove padding
            else:
                # Use the next bucket name
                trace_name = self._get_bucket_name()

            metadata["trace_name"] = trace_name

            self._waveform_file["data"].create_dataset(trace_name, data=waveform)

        else:
            bucket_name = self._get_bucket_name()

            bucket_waveforms, locations = self._pack_arrays([x[1] for x in bucket])
            self._waveform_file["data"].create_dataset(
                bucket_name, data=bucket_waveforms
            )  # Write data to hdf5 file

            # Set correct trace_names
            for i, location in enumerate(locations):
                metadata = bucket[i][0]
                if "trace_name" in metadata:
                    metadata["trace_name_original"] = metadata[
                        "trace_name"
                    ]  # Keep original trace_name in the metadata

                metadata["trace_name"] = f"{bucket_name}${location}"

    def _get_bucket_name(self):
        """
        Get the next available bucket name and increment bucket counter

        :return: bucket name
        :rtype: str
        """
        bucket_id = self._bucket_counter
        self._bucket_counter += 1
        return f"bucket{bucket_id}"

    @staticmethod
    def _pack_arrays(arrays):
        """
        Packs a list of arrays into one large array.
        Requires all arrays to have the same ndim and dtype.

        :param arrays: List of arrays to pack into
        :return: Packed array, location identifiers for each input trace.
        """

        assert all(x.ndim == arrays[0].ndim for x in arrays)
        assert all(x.dtype == arrays[0].dtype for x in arrays)

        output_shape = tuple(
            [len(arrays)]
            + [max(x.shape[i] for x in arrays) for i in range(arrays[0].ndim)]
        )
        output = np.zeros(output_shape, dtype=arrays[0].dtype)
        locations = []

        for i, x in enumerate(arrays):
            location_str = ",".join([str(i)] + [f":{s}" for s in x.shape])
            locations.append(location_str)

            pad_width = [
                (0, output_shape[i + 1] - x.shape[i])
                for i in range(len(output_shape) - 1)
            ]
            x = np.pad(x, pad_width, mode="constant", constant_values=0)
            output[i] = x

        return output, locations

    def set_total(self, n):
        """
        Set the total number of traces to write. Only used for correct progress calculation

        :param n: Number of traces
        :type n: int
        :return: None
        """
        if self._pbar is None:
            self._pbar = tqdm(desc="Traces converted")
        self._pbar.total = n

    def _finalize(self):
        """
        Finalizes the dataset, by flushing the remaining traces to hdf5 and writing metadata and data format.
        """
        self.flush_hdf5()

        if len(self._metadata) == 0:
            return

        if len(self.data_format) > 0:
            g_data_format = self._waveform_file.create_group("data_format")
            for key in self.data_format.keys():
                g_data_format.create_dataset(key, data=self.data_format[key])
        else:
            seisbench.logger.warning("No data format options specified.")

        metadata = pd.DataFrame(self._metadata)
        if self.metadata_dict is not None:
            metadata.rename(columns=self.metadata_dict, inplace=True)

        metadata.to_csv(self.metadata_path, index=False)

    def flush_hdf5(self):
        """
        Writes out all traces currently in the cache to the hdf5 file.
        Should be called if no more traces for the existing buckets will be added, e.g., after finishing a split.
        Does not write the metadata to csv.
        """
        buckets = list(self._cache.keys())

        for bucket in buckets:
            self._write_bucket(self._cache[bucket])
            del self._cache[bucket]
