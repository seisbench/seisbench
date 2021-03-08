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


class WaveformDataset:
    """
    This class is the base class for waveform datasets.

    Description for chunked data sets:
    As data sets can become too large to handle, one can chunk datasets.
    Instead of one metadata.csv and one waveforms.hdf5 file, a chunked data set consists of multiple chunks.
    Each chunk has one metadata{chunk}.csv and waveforms{chunk}.hdf5 file.
    Optionally, but recommended, a chunks file can be added to the dataset, which lists all chunk identifiers seperated by linebreaks.
    If no chunks file is available, chunks are derived from the file names.
    The data_format needs to be specified in each chunk and needs to be consistent across chunks.
    """

    def __init__(
        self,
        path,
        name=None,
        lazyload=True,
        dimension_order=None,
        component_order=None,
        sampling_rate=None,
        cache=False,
        chunks=None,  # Which chunks should be loaded. If None, load all chunks.
        **kwargs,
    ):
        if name is None:
            self._name = "Unnamed dataset"
        else:
            self._name = name
        self.lazyload = lazyload
        self.cache = cache
        self._path = path
        self._chunks = chunks
        if chunks is not None:
            self._chunks = sorted(chunks)

        self._dimension_order = None
        self._dimension_mapping = None
        self._component_order = None
        self._component_mapping = None
        self.sampling_rate = sampling_rate

        self._verify_dataset()

        metadatas = []
        for chunk, metadata_path, _ in zip(*self._chunks_with_paths()):
            tmp_metadata = pd.read_csv(
                metadata_path,
                dtype={"trace_sampling_rate_hz": float, "trace_dt_s": float},
            )
            tmp_metadata["trace_chunk"] = chunk
            metadatas.append(tmp_metadata)
        self._metadata = pd.concat(metadatas)

        self._data_format = self._read_data_format()

        self._unify_sampling_rate()

        self.dimension_order = dimension_order
        self.component_order = component_order

        self._waveform_cache = {}

        if not self.lazyload:
            if self.cache:
                self._load_waveform_data()
            else:
                seisbench.logger.warning(
                    "Skipping preloading of waveforms as cache is set to inactive."
                )

    def __str__(self):
        return f"{self._name} - {len(self)} traces"

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
        other._waveform_cache = copy.copy(self._waveform_cache)

        return other

    @property
    def metadata(self):
        return self._metadata

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        if self._path is None:
            raise ValueError("Path is None. Can't create data set without a path.")
        return Path(self._path)

    @property
    def data_format(self):
        # Returns a copy of the data format, to ensure internal data format is not modified.
        return dict(self._data_format)

    @property
    def dimension_order(self):
        return self._dimension_order

    @dimension_order.setter
    def dimension_order(self, value):
        if value is None:
            value = seisbench.config["dimension_order"]

        self._dimension_mapping = self._get_order_mapping(
            "N" + self._data_format["dimension_order"], value
        )
        self._dimension_order = value

    @property
    def component_order(self):
        return self._component_order

    @component_order.setter
    def component_order(self, value):
        if value is None:
            value = seisbench.config["component_order"]

        self._component_mapping = self._get_order_mapping(
            self._data_format["component_order"], value
        )
        self._component_order = value

    @property
    def chunks(self):
        if self._chunks is None:
            self._chunks = self.available_chunks(self.path)
        return self._chunks

    @staticmethod
    def available_chunks(path):
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
        The sampling rate can be specified in three ways:
        - as trace_sampling_rate_hz in the metadata
        - as trace_dt_s in the metadata
        - as sampling_rate in the data format group, indicating identical sampling rate for all traces
        This function writes the sampling rate into trace_sampling_rate_hz in the metadata for unified access in the later processing.
        If the sampling rate is specified in multiple ways and inconsistencies are detected, a warning is logged.
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

    @staticmethod
    def _get_order_mapping(source, target):
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
        metadata_paths = [self.path / f"metadata{chunk}.csv" for chunk in self.chunks]
        waveform_paths = [self.path / f"waveforms{chunk}.hdf5" for chunk in self.chunks]

        return self.chunks, metadata_paths, waveform_paths

    def _verify_dataset(self):
        for chunk, metadata_path, waveform_path in zip(*self._chunks_with_paths()):
            chunks_str = f" for chunk '{chunk}'" if chunk != "" else ""

            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing metadata file{chunks_str}")
            if not waveform_path.is_file():
                raise FileNotFoundError(f"Missing waveforms file{chunks_str}")

    def _read_data_format(self):
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
        if "component_order" not in data_format:
            seisbench.logger.warning(
                "Component order not specified in data set. Assuming ZNE."
            )
            data_format["component_order"] = "ZNE"

        return data_format

    def _load_waveform_data(self):
        """
        Loads waveform data from hdf5 file into cache
        :return:
        """
        chunks, metadata_paths, waveforms_path = self._chunks_with_paths()
        with LoadingContext(chunks, waveforms_path) as context:
            for trace_name, chunk in zip(
                self._metadata["trace_name"], self._metadata["trace_chunk"]
            ):
                self._get_single_waveform(trace_name, chunk, context=context)

    def filter(self, mask, inplace=True):
        """
        Filters dataset, e.g. by distance/magnitude/..., using a binary mask.
        Default behaviour is to perform inplace filtering, directly changing the
        metadata and waveforms to only keep the results of the masking query.
        Setting inplace equal to false will return a filtered copy of the data set.
        For details on the copy operation see :py:func:`~WaveformDataset.copy`.
        Example usage dataset.filter(dataset["p_status"] == "manual")
        :return:
        """
        if inplace:
            self._metadata = self._metadata[mask]
            self._evict_cache()
        else:
            other = self.copy()
            other.filter(mask, inplace=True)
            return other

    # NOTE: lat/lon columns are specified to enhance generalisability as naming convention may
    # change between datasets and users may also want to filter as a function of  recievers/sources
    def region_filter(self, domain, lat_col, lon_col):
        """
        In place filtering of dataset based on predefined region or geometry.
        See also convenience functions region_filter_[source|receiver]
        :param domain: The domain filter
        :type domain: obspy.core.fdsn.mass_downloader.domain:
        :param lat_col: Name of latitude coordinate column
        :type lat_col: str
        :param lon_col: Name of longitude coordinate column
        :type lon_col: str
        :return:
        """
        check_domain = lambda metadata: domain.is_in_domain(
            metadata[lat_col], metadata[lon_col]
        )
        mask = self._metadata.apply(check_domain, axis=1)
        self.filter(mask)

    def region_filter_source(self, domain):
        self.region_filter(
            domain, lat_col="source_latitude_deg", lon_col="source_longitude_deg"
        )

    def region_filter_receiver(self, domain):
        self.region_filter(
            domain, lat_col="station_latitude_deg", lon_col="station_longitude_deg"
        )

    def get_split(self, split):
        """
        Returns a dataset with the
        :param split: Split name to return. Usually one of "train", "dev", "test"
        :return: Dataset filtered to the requested split.
        """
        if "split" not in self.metadata.columns:
            raise ValueError("Split requested but no split defined in metadata")

        mask = (self.metadata["split"] == split).values

        return self.filter(mask, inplace=False)

    def train(self):
        """
        Convenience method for get_split("train")
        :return: Training dataset
        """
        return self.get_split("train")

    def dev(self):
        """
        Convenience method for get_split("dev")
        :return: Development dataset
        """
        return self.get_split("dev")

    def test(self):
        """
        Convenience method for get_split("test")
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
        :return:
        """
        existing_keys = set(self._metadata["trace_name"])
        delete_keys = []
        for key in self._waveform_cache.keys():
            if key not in existing_keys:
                delete_keys.append(key)

        for key in delete_keys:
            del self._waveform_cache[key]

        seisbench.logger.debug(f"Deleted {len(delete_keys)} entries in cache eviction")

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Can only use strings to access metadata parameters")
        return self._metadata[item]

    def __len__(self):
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
                        metadata[key] *= resampling_factor

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
        :param mask: Binary mask on the metadata, indicating which traces should be returned. Can not be used jointly with split.
        :param sampling_rate: Target sampling rate, overwrites sampling rate for dataset
        :return: Waveform array with dimensions ordered according to dimension_order e.g. default 'NCW' (number of traces, number of components, record samples). If the number of components or record samples varies between different entries, all entries are padded to the maximum length.
        """
        squeeze = False
        if idx is not None:
            if mask is not None:
                raise ValueError("Mask can not be used jointly with idx.")
            if isinstance(idx, int):
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
            for trace_name, chunk, trace_sampling_rate in zip(
                load_metadata["trace_name"],
                load_metadata["trace_chunk"],
                load_metadata["trace_sampling_rate_hz"],
            ):
                waveforms.append(
                    self._get_single_waveform(
                        trace_name,
                        chunk,
                        context=context,
                        target_sampling_rate=sampling_rate,
                        source_sampling_rate=trace_sampling_rate,
                    )
                )

        waveforms = self._pad_packed_sequence(waveforms)
        # Impose correct component order
        component_dimension = list(self._data_format["dimension_order"]).index("C") + 1
        waveforms = waveforms.take(self._component_mapping, axis=component_dimension)

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
    ):
        if not self.cache or trace_name not in self._waveform_cache:
            g_data = context[chunk]["data"]
            waveform = g_data[str(trace_name)][()]
            if self.cache:
                self._waveform_cache[trace_name] = waveform
        else:
            waveform = self._waveform_cache[trace_name]

        if target_sampling_rate is not None:
            if np.isnan(source_sampling_rate):
                raise ValueError("Tried resampling trace with unknown sampling rate.")
            else:
                waveform = self._resample(
                    waveform, target_sampling_rate, source_sampling_rate
                )

        return waveform

    def _resample(self, waveform, target_sampling_rate, source_sampling_rate, eps=1e-4):
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
        :param res: Resolution for cartopy features
        :param connections: If true, plots lines connecting sources and stations
        :param kwargs: Plotting kwargs that will be passed to matplotlib plot. Args need to be prefixed with 'sta_', 'ev_' and 'conn_' to address stations, events or connections.
        :return: A figure handle for the created figure
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
    It adds functionality to download the dataset to cache and to annotate it with a citation.
    """

    def __init__(
        self,
        chunks=None,
        citation=None,
        force=False,
        wait_for_file=False,
        **kwargs,
    ):
        self._name = self._name_internal()
        self._citation = citation
        self.path.mkdir(exist_ok=True, parents=True)

        if chunks is None:
            chunks = self.available_chunks(force=force, wait_for_file=wait_for_file)

        # TODO: Validate if cached dataset was downloaded with the same parameters
        for chunk in chunks:

            def download_callback(files):
                chunk_str = f'Chunk "{chunk}" of ' if chunk != "" else ""
                seisbench.logger.info(
                    f"{chunk_str}Dataset {self.name} not in cache. Trying to download preprocessed version from SeisBench repository."
                )
                try:
                    self._download_preprocessed(*files, chunk=chunk)
                except ValueError:
                    seisbench.logger.info(
                        f"{chunk_str}Dataset {self.name} not SeisBench repository. Starting download and conversion from source."
                    )
                    if (
                        "chunk"
                        not in inspect.signature(self._download_dataset).parameters
                        and chunk != ""
                    ):
                        raise ValueError(
                            "Data set seems not to support chunking, but chunk provided."
                        )
                    with WaveformDataWriter(*files) as writer:
                        self._download_dataset(writer, chunk=chunk, **kwargs)

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
        return self._citation

    @classmethod
    def _path_internal(cls):
        return Path(seisbench.cache_root, "datasets", cls._name_internal().lower())

    @property
    def path(self):
        return self._path_internal()

    @classmethod
    def _name_internal(cls):
        return cls.__name__

    @property
    def name(self):
        return self._name_internal()

    @classmethod
    def _remote_path(cls):
        return os.path.join(
            seisbench.remote_root, "datasets", cls._name_internal().lower()
        )

    @classmethod
    def available_chunks(cls, force=False, wait_for_file=False):
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
                        remote_chunks_path, file, progress_bar=False
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
                # To write the conversion for a chunked file, simply write the chunks file before calling the super constructor.
                chunks = [""]

        return chunks

    def _download_preprocessed(self, metadata_path, waveforms_path, chunk):
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
        Download and convert the dataset to the standard seisbench format.
        The metadata must contain at least the columns 'trace_name' and 'split'.
        :param writer: A WaveformDataWriter instance
        :param chunk: The chunk to be downloaded. Can be ignored if unchunked data set is created.
        :param kwargs:
        :return:
        """
        pass


class WaveformDataWriter:
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
        self._metadata.append(metadata)
        trace_name = str(metadata["trace_name"])

        if self._waveform_file is None:
            self._waveform_file = h5py.File(self.waveforms_path, "w")
            self._waveform_file.create_group("data")
        if self._pbar is None:
            self._pbar = tqdm(desc="Traces converted")

        self._waveform_file["data"].create_dataset(trace_name, data=waveform)
        self._pbar.update()

    def set_total(self, n):
        if self._pbar is None:
            self._pbar = tqdm(desc="Traces converted")
        self._pbar.total = n

    def _finalize(self):
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
