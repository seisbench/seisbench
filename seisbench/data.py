import seisbench
from abc import abstractmethod
from pathlib import Path
import pandas as pd


class WaveformDataset:
    """
    This class is the abstract base class for waveform datasets.
    """

    # TODO: Design interface for accessing train/dev/test splits
    # TODO: Design interface for actual data access
    def __init__(self, name, lazyload=True, **kwargs):
        self.name = name
        self.lazyload = lazyload

        # Check if dataset is cached
        # TODO: Validate if cached dataset was downloaded with the same parameters
        metadata_path = self._dataset_path() / "metadata.csv"
        if not metadata_path.is_file():
            self._download_dataset(**kwargs)

        self._metadata = pd.read_csv(metadata_path)
        self._waveform_cache = {}

        if not self.lazyload:
            self._load_waveform_data()

    @property
    def metadata(self):
        return self._metadata

    def _dataset_path(self):
        return Path(seisbench.cache_root, self.name)

    @abstractmethod
    def _download_dataset(self, **kwargs):
        pass

    # TODO: Implement
    def _load_waveform_data(self):
        """
        Loads waveform data from hdf5 file into cache
        :return:
        """
        pass

    # TODO: Implement
    def _write_data(self, metadata, waveforms, metadata_dict=None):
        """
        Writes data from memory to disk using the standard file format. Should only be used in _download_dataset.
        :param metadata:
        :param waveforms:
        :param metadata_dict: String mapping from the column names used in the provided metadata to the standard column names
        :return:
        """
        pass

    # TODO: Design actual interface and implement
    def filter(self):
        """
        Filters data set inplace, e.g. by distance/magnitude/...
        :return:
        """
        pass

    # TODO: Implement
    def _evict_cache(self):
        """
        Remove all traces from cache that do not have any reference in metadata anymore
        :return:
        """
        pass


# TODO: Implement
class DummyDataset(WaveformDataset):
    """
    A dummy dataset visualizing the implementation of custom datasets
    """

    def __init__(self, **kwargs):
        super().__init__(name=self.__class__.__name__.lower(), **kwargs)
