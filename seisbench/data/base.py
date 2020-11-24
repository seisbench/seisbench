import seisbench
from abc import abstractmethod, ABC
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
import ftplib
import logging
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from tqdm import tqdm


class WaveformDataset(ABC):
    """
    This class is the abstract base class for waveform datasets.
    """

    # TODO: Define/implement a convention for channel naming/order
    def __init__(
        self, name, citation=None, lazyload=True, dimension_order="NCW", **kwargs
    ):
        self._name = name
        self.lazyload = lazyload
        self._citation = citation
        self.dimension_order = dimension_order

        # Check if dataset is cached
        # TODO: Validate if cached dataset was downloaded with the same parameters
        metadata_path = self._dataset_path() / "metadata.csv"
        if not metadata_path.is_file():
            logging.info(
                f"Dataset {name} not in cache. Downloading and preprocessing corpus..."
            )
            self._download_dataset(**kwargs)
        self._metadata = pd.read_csv(metadata_path)
        self._waveform_cache = {}

        if not self.lazyload:
            self._load_waveform_data()

    @property
    def metadata(self):
        return self._metadata

    @property
    def name(self):
        return self._name

    @property
    def citation(self):
        return self._citation

    def _dataset_path(self):
        return Path(seisbench.cache_root, self.name.lower())

    # NOTE: Obtains the dimension ordering which can be specified by the user
    def _get_dim_order(self):
        return [("NCW").find(s) for s in self.dimension_order]

    @abstractmethod
    def _download_dataset(self, **kwargs):
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

    def _write_data(self, metadata, waveforms, metadata_dict=None):
        """
        Writes data from memory to disk using the standard file format. Should only be used in _download_dataset.
        :param metadata:
        :param waveforms:
        :param metadata_dict: String mapping from the column names used in the provided metadata to the standard column names
        :return:
        """
        if metadata_dict is not None:
            metadata.rename(columns=metadata_dict, inplace=True)

        metadata.to_csv(self._dataset_path() / "metadata.csv", index=False)

        with h5py.File(self._dataset_path() / "waveforms.hdf5", "w") as fout:
            gdata = fout.create_group("data")
            for trace_name in tqdm(
                metadata["trace_name"],
                total=len(metadata),
                desc="Writing waveforms to disk",
            ):
                gdata.create_dataset(str(trace_name), data=waveforms[trace_name])

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

        logging.debug(f"Deleted {len(delete_keys)} entries in cache eviction")

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

        max_components = max([x.shape[0] for x in waveforms])
        max_records = max([x.shape[1] for x in waveforms])

        for i, waveform in enumerate(waveforms):
            if waveform.shape != (max_components, max_records):
                d_components = max_components - waveform.shape[0]
                d_records = max_records - waveform.shape[1]
                waveforms[i] = np.pad(
                    waveform,
                    ((0, d_components), (0, d_records)),
                    "constant",
                    constant_value=0,
                )

        dim_0, dim_1, dim_2 = self._get_dim_order()
        return np.stack(waveforms, axis=0).transpose(dim_0, dim_1, dim_2)

    def _get_single_waveform(self, trace_name, f_wave=None):
        if trace_name not in self._waveform_cache:
            if f_wave is None:
                f_wave = h5py.File(self._dataset_path() / "waveforms.hdf5", "r")
            g_data = f_wave["data"]
            self._waveform_cache[trace_name] = g_data[str(trace_name)][()]

        return self._waveform_cache[trace_name]


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

    def _download_dataset(self, trace_length=60, **kwargs):
        sampling_rate = 20
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

        metadata["trace_name"] = metadata["time"].apply(to_tracename)
        waveforms = {}
        client = Client("GFZ")
        for trace_name, time in tqdm(
            zip(metadata["trace_name"], metadata["time"]),
            total=len(metadata),
            desc="Downloading waveforms",
        ):
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
            waveforms[trace_name] = waveform

        inv = client.get_stations(
            net="CX",
            sta="PB01",
            starttime=UTCDateTime.strptime(
                "2007/01/01 00:00:00.00", "%Y/%m/%d %H:%M:%S.%f"
            ),
        )

        metadata["network_code"] = "CX"
        metadata["receiver_code"] = "PB01"
        metadata["receiver_type"] = "BH"
        metadata["receiver_latitude"] = inv[0][0].latitude
        metadata["receiver_longitude"] = inv[0][0].longitude
        metadata["receiver_elevation_m"] = inv[0][0].elevation
        metadata["source_magnitude_type"] = "MA"
        metadata["source_magnitude_type2"] = "ML"

        metadata_dict = {
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
        self._write_data(metadata, waveforms, metadata_dict)
