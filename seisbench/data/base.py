import seisbench
from abc import abstractmethod, ABC
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
import ftplib
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from tqdm import tqdm
import shutil


class WaveformDataset(ABC):
    """
    This class is the abstract base class for waveform datasets.
    """

    def __init__(
        self,
        name,
        citation=None,
        lazyload=True,
        dimension_order=None,
        component_order=None,
        cache=False,
        **kwargs,
    ):
        self._name = name
        self.lazyload = lazyload
        self._citation = citation
        self.cache = cache

        self._dimension_order = None
        self._dimension_mapping = None
        self._component_order = None
        self._component_mapping = None

        # Check if dataset is cached
        # TODO: Validate if cached dataset was downloaded with the same parameters
        metadata_path = self._dataset_path() / "metadata.csv"
        if not metadata_path.is_file():
            seisbench.logger.info(
                f"Dataset {name} not in cache. Downloading and preprocessing corpus..."
            )
            with WaveformDataWriter(self._dataset_path()) as writer:
                self._download_dataset(writer, **kwargs)

        self._metadata = pd.read_csv(metadata_path)

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
    def citation(self):
        return self._citation

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

    def _dataset_path(self):
        return Path(seisbench.cache_root, self.name.lower())

    def _read_data_format(self):
        with h5py.File(self._dataset_path() / "waveforms.hdf5", "r") as f_wave:
            try:
                g_data_format = f_wave["data_format"]
                data_format = {
                    key: g_data_format[key][()] for key in g_data_format.keys()
                }
            except KeyError:
                seisbench.logger.warning("No data_format group found in .hdf5 File.")
                data_format = {}

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

    @abstractmethod
    def _download_dataset(self, writer, **kwargs):
        """
        Download and convert the dataset to the standard seisbench format. The metadata must contain at least the columns 'trace_name' and 'split'.
        :param kwargs:
        :return:
        """
        pass

    def _load_waveform_data(self):
        """
        Loads waveform data from hdf5 file into cache
        :return:
        """
        with h5py.File(self._dataset_path() / "waveforms.hdf5", "r") as f_wave:
            for trace_name in self._metadata["trace_name"]:
                self._get_single_waveform(trace_name, f_wave=f_wave)

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
            domain, lat_col="source_latitude", lon_col="source_longitude"
        )

    def region_filter_receiver(self, domain):
        self.region_filter(
            domain, lat_col="receiver_latitude", lon_col="receiver_longitude"
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

    def get_waveforms(self, split=None, mask=None):
        """
        Collects waveforms and returns them as an array.
        :param split: Split (train/dev/test) to obtain waveforms for
        :param mask: Binary mask on the metadata, indicating which traces should be returned. Can not be used jointly with split.
        :return: Waveform array with dimensions ordered according to dimension_order e.g. default 'NCW' (number of traces, number of components, record samples). If the number of components or record samples varies between different entries, all entries are padded to the maximum length.
        """
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
        with h5py.File(self._dataset_path() / "waveforms.hdf5", "r") as f_wave:
            for trace_name in load_metadata["trace_name"]:
                waveforms.append(self._get_single_waveform(trace_name, f_wave=f_wave))

        waveforms = self._pad_packed_sequence(waveforms)
        component_dimension = list(self._data_format["dimension_order"]).index("C") + 1
        waveforms = waveforms.take(self._component_mapping, axis=component_dimension)

        return waveforms.transpose(*self._dimension_mapping)

    def _get_single_waveform(self, trace_name, f_wave=None):
        if not self.cache or trace_name not in self._waveform_cache:
            if f_wave is None:
                f_wave = h5py.File(self._dataset_path() / "waveforms.hdf5", "r")
            g_data = f_wave["data"]
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


class WaveformDataWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata_dict = {}
        self.data_format = {}

        self.path.mkdir(parents=True, exist_ok=True)

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
            if (self.path / "metadata.csv").is_file():
                shutil.move(
                    self.path / "metadata.csv", self.path / "metadata.csv.partial"
                )
            if (self.path / "waveforms.hdf5").is_file():
                shutil.move(
                    self.path / "waveforms.hdf5", self.path / "waveforms.hdf5.partial"
                )
            seisbench.logger.error(
                f"Error in downloading dataset. "
                f"Saved current progress to {self.path}/*.partial. Error message:\n"
            )

    def add_trace(self, metadata, waveform):
        self._metadata.append(metadata)
        trace_name = str(metadata["trace_name"])

        if self._waveform_file is None:
            self._waveform_file = h5py.File(self.path / "waveforms.hdf5", "w")
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

        metadata.to_csv(self.path / "metadata.csv", index=False)


class DummyDataset(WaveformDataset):
    """
    A dummy dataset visualizing the implementation of custom datasets
    """

    def __init__(self, **kwargs):
        citation = (
            "Münchmeyer, Jannes; Bindi, Dino; Sippl, Christian; Leser, Ulf; Tilmann, Frederik (2019): "
            "Magnitude scales, attenuation models and feature matrices for the IPOC catalog. "
            "V. 1.0. GFZ Data Services. https://doi.org/10.5880/GFZ.2.4.2019.004"
        )
        super().__init__(
            name=self.__class__.__name__.lower(), citation=citation, **kwargs
        )

    def _download_dataset(self, writer, trace_length=60, **kwargs):
        sampling_rate = 20

        writer.metadata_dict = {
            "time": "trace_start_time",
            "latitude": "source_latitude",
            "longitude": "source_longitude",
            "depth": "source_depth_km",
            "cls": "source_event_category",
            "MA": "source_magnitude",
            "ML": "source_magnitude2",
            "std_MA": "source_magnitude_uncertainty",
            "std_ML": "source_magnitude_uncertainty2",
        }
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": sampling_rate,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        path = self._dataset_path()
        path.mkdir(parents=True, exist_ok=True)
        with ftplib.FTP("datapub.gfz-potsdam.de", "anonymous", "") as ftp:
            with open(path / "raw_catalog.csv", "wb") as fout:
                ftp.retrbinary(
                    "RETR download/10.5880.GFZ.2.4.2019.004/IPOC_catalog_magnitudes.csv",
                    fout.write,
                )

        metadata = pd.read_csv(path / "raw_catalog.csv")
        metadata = metadata.iloc[:100].copy()

        def to_tracename(x):
            for c in "/:.":
                x = x.replace(c, "_")
            return x

        client = Client("GFZ")
        inv = client.get_stations(
            net="CX",
            sta="PB01",
            starttime=UTCDateTime.strptime(
                "2007/01/01 00:00:00.00", "%Y/%m/%d %H:%M:%S.%f"
            ),
        )

        metadata["trace_name"] = metadata["time"].apply(to_tracename)
        metadata["network_code"] = "CX"
        metadata["receiver_code"] = "PB01"
        metadata["receiver_type"] = "BH"
        metadata["receiver_latitude"] = inv[0][0].latitude
        metadata["receiver_longitude"] = inv[0][0].longitude
        metadata["receiver_elevation_m"] = inv[0][0].elevation
        metadata["source_magnitude_type"] = "MA"
        metadata["source_magnitude_type2"] = "ML"

        writer.set_total(len(metadata))
        for _, row in metadata.iterrows():
            time = row["time"]
            waveform = np.zeros((3, sampling_rate * trace_length))
            time = UTCDateTime.strptime(time, "%Y/%m/%d %H:%M:%S.%f")
            stream = client.get_waveforms(
                "CX", "PB01", "*", "BH?", time, time + trace_length
            )
            for cid, component in enumerate("ZNE"):
                ctrace = stream.select(channel=f"??{component}")[0]
                waveform[cid] = ctrace.data[: sampling_rate * trace_length].astype(
                    float
                )
            writer.add_trace(row, waveform)
