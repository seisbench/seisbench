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

        self._verify_dataset()

        metadatas = []
        for chunk, metadata_path, _ in zip(*self._chunks_with_paths()):
            tmp_metadata = pd.read_csv(metadata_path)
            tmp_metadata["trace_chunk"] = chunk
            metadatas.append(tmp_metadata)
        self._metadata = pd.concat(metadatas)

        self._data_format = self._read_data_format()
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

        if "sampling_rate" not in data_format:
            seisbench.logger.warning("Sampling rate not specified in data set")
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

    def filter(self, mask):
        """
        Filters dataset inplace, e.g. by distance/magnitude/..., using a binary mask
        Example usage dataset.filter(dataset["p_status"] == "manual")
        :return:
        """
        self._metadata = self._metadata[mask]
        self._evict_cache()

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

    def get_waveforms(self, idx=None, split=None, mask=None):
        """
        Collects waveforms and returns them as an array.
        :param idx: Idx or list of idx to obtain waveforms for
        :param split: Split (train/dev/test) to obtain waveforms for
        :param mask: Binary mask on the metadata, indicating which traces should be returned. Can not be used jointly with split.
        :return: Waveform array with dimensions ordered according to dimension_order e.g. default 'NCW' (number of traces, number of components, record samples). If the number of components or record samples varies between different entries, all entries are padded to the maximum length.
        """
        squeeze = False
        if idx is not None:
            if split is not None or mask is not None:
                raise ValueError("Split and mask can not be used jointly with idx.")
            if isinstance(idx, int):
                idx = [idx]
                squeeze = True

            load_metadata = self._metadata.iloc[idx]
        else:
            if split is not None and mask is not None:
                raise ValueError("Split and mask can not be used jointly.")

            if split is not None and split not in ["train", "dev", "test"]:
                raise ValueError("Split must be one of 'train', 'dev', 'test' or None.")

            if split is not None:
                load_metadata = self._metadata["split"] == split
            elif mask is not None:
                load_metadata = self._metadata[mask]
            else:
                load_metadata = self._metadata

        waveforms = []
        chunks, metadata_paths, waveforms_path = self._chunks_with_paths()
        with LoadingContext(chunks, waveforms_path) as context:
            for trace_name, chunk in zip(
                load_metadata["trace_name"], load_metadata["trace_chunk"]
            ):
                waveforms.append(
                    self._get_single_waveform(trace_name, chunk, context=context)
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

    def _get_single_waveform(self, trace_name, chunk, context):
        if not self.cache or trace_name not in self._waveform_cache:
            g_data = context[chunk]["data"]
            waveform = g_data[str(trace_name)][()]
            if self.cache:
                self._waveform_cache[trace_name] = waveform
        else:
            waveform = self._waveform_cache[trace_name]

        return waveform

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


# TODO: Overwrite available_chunks function to download chunks file if necessary
# TODO: Implement correct download
class BenchmarkDataset(WaveformDataset, ABC):
    """
    This class is the base class for benchmark waveform datasets.
    It adds functionality to download the dataset to cache and to annotate it with a citation.
    """

    def __init__(self, name, citation=None, force=False, wait_for_file=False, **kwargs):
        self._name = name
        self._citation = citation

        # TODO: Validate if cached dataset was downloaded with the same parameters
        def download_callback(files):
            seisbench.logger.info(
                f"Dataset {name} not in cache. Trying to download preprocessed corpus from SeisBench repository."
            )
            try:
                self._download_preprocessed(*files)
            except ValueError:
                seisbench.logger.info(
                    f"Dataset {name} not SeisBench repository. Starting download and conversion from source."
                )
                with WaveformDataWriter(*files) as writer:
                    self._download_dataset(writer, **kwargs)

        files = [self.path / "metadata.csv", self.path / "waveforms.hdf5"]
        seisbench.util.callback_if_uncached(
            files, download_callback, force=force, wait_for_file=wait_for_file
        )

        super().__init__(path=None, name=name, citation=citation, **kwargs)

    @property
    def citation(self):
        return self._citation

    @property
    def path(self):
        return Path(seisbench.cache_root, "datasets", self.name.lower())

    def _remote_path(self):
        return os.path.join(seisbench.remote_root, "datasets", self.name.lower())

    def _download_preprocessed(self, metadata_path, waveforms_path):
        self.path.mkdir(parents=True, exist_ok=True)

        remote_path = self._remote_path()
        remote_metadata_path = os.path.join(remote_path, "metadata.csv")
        remote_waveforms_path = os.path.join(remote_path, "waveforms.hdf5")

        seisbench.util.download_http(
            remote_metadata_path, metadata_path, desc="Downloading metadata"
        )
        seisbench.util.download_http(
            remote_waveforms_path, waveforms_path, desc="Downloading waveforms"
        )

    @abstractmethod
    def _download_dataset(self, writer, **kwargs):
        """
        Download and convert the dataset to the standard seisbench format.
        The metadata must contain at least the columns 'trace_name' and 'split'.
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
