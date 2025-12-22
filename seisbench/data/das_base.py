from abc import ABC
from pathlib import Path
from typing import Any
import datetime

import pandas as pd
import numpy as np
import h5py
import copy

import seisbench
from .base import AbstractBenchmarkDataset, MultiWaveformDataset

NPArrayOrHDF5Dataset = np.ndarray | h5py.Dataset


class DASDataset:
    # File path format for metadata and records
    METADATA_FILE = "metadata_$CHUNK.parquet"
    DATA_FILE = "records_$CHUNK.hdf5"

    def __init__(self, path: Path | str = None, chunks: list[str] | None = None):
        self._path = path
        if chunks is None:
            chunks = self.available_chunks(path)

        self._chunks = sorted(chunks)
        self._validate_chunks(path, self._chunks)
        self._validate_dataset()

        self._metadata = self._load_metadata()
        self._data_pointers = {}

    @property
    def chunks(self) -> list[str]:
        return self._chunks

    @property
    def path(self) -> Path:
        """
        Path of the dataset
        """
        # This path is overwritten by DASBenchmarkDataset
        return Path(self._path)

    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata

    def __len__(self) -> int:
        return len(self._metadata)

    def _validate_chunks(self, path: Path, chunks: list[str]) -> None:
        available_chunks = self.available_chunks(path)
        if any(chunk not in available_chunks for chunk in chunks):
            raise ValueError(
                f"The dataset does not contain the following chunks: "
                f"{[chunk for chunk in chunks if chunk not in self.available_chunks(path)]}"
            )

    def _validate_dataset(self) -> None:
        for chunk in self.chunks:
            metadata_path = self.METADATA_FILE.replace("$CHUNK", chunk)
            data_path = self.DATA_FILE.replace("$CHUNK", chunk)
            if not (self.path / metadata_path).is_file():
                raise ValueError(f"Metadata file {metadata_path} does not exist.")
            if not (self.path / data_path).is_file():
                raise ValueError(f"Data file {data_path} does not exist.")

    def _load_metadata(self) -> pd.DataFrame:
        metadata = []
        for chunk in self.chunks:
            metadata_path = self.METADATA_FILE.replace("$CHUNK", chunk)
            chunk_metadata = pd.read_parquet(self.path / metadata_path)
            chunk_metadata["chunk"] = chunk
            metadata.append(chunk_metadata)

        metadata = pd.concat(metadata)
        metadata.reset_index(drop=True, inplace=True)
        return metadata

    @staticmethod
    def available_chunks(path: Path) -> list[str]:
        """
        Determines the chunks of the dataset in the given path. If available, parses the chunks file. Otherwise,
        scans the dataset for metadata and records files.

        :param path: Dataset path
        :return: List of chunks
        """
        chunks_path = path / "chunks"
        if chunks_path.is_file():
            with open(chunks_path, "r") as f:
                chunks = [x for x in f.read().split("\n") if x.strip()]

            if len(chunks) == 0:
                chunks = [""]
        else:
            metadata_files = set(
                [
                    x.name[9:-8]
                    for x in path.iterdir()
                    if x.name.startswith("metadata_") and x.name.endswith(".parquet")
                ]
            )
            waveform_files = set(
                [
                    x.name[8:-5]
                    for x in path.iterdir()
                    if x.name.startswith("records_") and x.name.endswith(".hdf5")
                ]
            )

            chunks = list(metadata_files & waveform_files)

        return sorted(chunks)

    def get_sample(
        self, idx: int, record_virtual: bool = True, annotations_virtual: bool = False
    ) -> tuple[dict[str, Any], NPArrayOrHDF5Dataset, dict[str, NPArrayOrHDF5Dataset]]:
        """
        Load the sample with the given index. Use the `record_virtual` and `annotations_virtual` arguments to control
        whether the record and annotations are loaded into memory or only pointers are returned. By default, the record
        will not be loaded into memory, while the annotations will be loaded into memory.

        :param idx: Index of the sample to load
        :param record_virtual: If true, the record is returned as a virtual array.
                               Otherwise, the record is loaded into memory.
        :param annotations_virtual: If true, the annotations are returned as virtual arrays.
                                    Otherwise, the annotations are loaded into memory.
        """
        metadata = self._metadata.iloc[idx].to_dict()
        chunk = metadata["chunk"]
        if chunk not in self._data_pointers:
            self._data_pointers[chunk] = h5py.File(
                self.path / self.DATA_FILE.replace("$CHUNK", chunk), "r"
            )["records"]
        record_group = self._data_pointers[chunk][metadata["record_name"]]

        record = record_group["record"]
        annotations = {}
        for key in record_group["annotations"].keys():
            annotations[key] = record_group["annotations"][key]
            if not annotations_virtual:
                annotations[key] = annotations[key][()]

        if not record_virtual:
            record = record[()]

        return metadata, record, annotations

    def filter(self, mask: np.ndarray, inplace: bool = True) -> "DASDataset | None":
        """
        Filters dataset, e.g. by distance/magnitude/..., using a binary mask.
        Default behaviour is to perform inplace filtering.
        Setting inplace equal to false will return a filtered copy of the data set.

        :param mask: Boolean mask to apply to metadata.
        :param inplace: If true, filter inplace.

        Example usage:

        .. code:: python

            dataset.filter(dataset.metadata["record_sampling_rate_hz"] > 100)
        """
        if inplace:
            self._metadata = self._metadata[mask]
            # Drop pointers to chunks that are no longer in the metadata
            remaining_chunks = self._metadata["chunk"].unique()
            self._data_pointers = {
                k: v for k, v in self._data_pointers.values() if k in remaining_chunks
            }
            return None
        else:
            other = self.copy()
            other.filter(mask, inplace=True)
            return other

    def copy(self) -> "DASDataset":
        """
        Create a copy of the data set. All attributes are copied by value.
        """
        other = copy.copy(self)
        other._metadata = self._metadata.copy()
        other._data_pointers = {}

        return other

    def get_split(self, split: str, inplace: bool = False) -> "DASDataset | None":
        """
        Returns a dataset with the requested split.

        :param split: Split name to return. Usually one of "train", "dev", "test"
        :return: Dataset filtered to the requested split.
        """
        if "split" not in self.metadata.columns:
            raise ValueError("Split requested but no split defined in metadata")

        mask = (self.metadata["split"] == split).values

        return self.filter(mask, inplace=inplace)

    def train(self, inplace: bool = False) -> "DASDataset | None":
        """
        Convenience method for get_split("train").

        :return: Training dataset
        """
        return self.get_split("train", inplace=inplace)

    def dev(self, inplace: bool = False) -> "DASDataset | None":
        """
        Convenience method for get_split("dev").

        :return: Development dataset
        """
        return self.get_split("dev", inplace=inplace)

    def test(self, inplace: bool = False) -> "DASDataset | None":
        """
        Convenience method for get_split("test").

        :return: Test dataset
        """
        return self.get_split("test", inplace=inplace)

    def train_dev_test(self) -> tuple["DASDataset", "DASDataset", "DASDataset"]:
        """
        Convenience method for returning training, development and test set. Equal to:

        >>> self.train(), self.dev(), self.test()

        :return: Training dataset, development dataset, test dataset
        """
        return self.train(), self.dev(), self.test()

    def __add__(self, other) -> "MultiDASDataset":
        if isinstance(other, DASDataset):
            return MultiDASDataset([self, other])
        elif isinstance(other, MultiDASDataset):
            return MultiDASDataset([self] + other.datasets)
        else:
            raise TypeError(
                "Can only add DASDataset and MultiDASDataset to DASDataset."
            )


class MultiDASDataset:
    """
    This class is a wrapper for multiple DAS datasets. It allows combining multiple datasets into a single dataset.
    It has mostly the same API as :py:class:`DASDataset`.
    """

    def __init__(self, datasets: list[DASDataset]):
        if not isinstance(datasets, list) or not all(
            isinstance(x, DASDataset) for x in datasets
        ):
            raise TypeError("MultiDASDataset expects a list of DASDataset as input.")

        if len(datasets) == 0:
            raise ValueError("MultiDASDatasets need to have at least one member.")

        self._datasets = [dataset.copy() for dataset in datasets]
        self._metadata = pd.concat(x.metadata for x in datasets)

        # Identify dataset
        self._metadata["trace_dataset"] = sum(
            ([i] * len(dataset) for i, dataset in enumerate(self._datasets)), []
        )
        self._metadata.reset_index(inplace=True, drop=True)

    @property
    def datasets(self):
        return list(self._datasets)

    def __add__(self, other: "DASDataset | MultiDASDataset") -> "MultiDASDataset":
        if isinstance(other, DASDataset):
            return MultiDASDataset(self._datasets + [other])
        elif isinstance(other, MultiDASDataset):
            return MultiDASDataset(self._datasets + other._datasets)
        else:
            raise TypeError(
                "Can only add DASDataset and MultiDASDataset to MultiDASDataset."
            )

    @property
    def metadata(self):
        return self._metadata

    def filter(
        self, mask: np.ndarray, inplace: bool = True
    ) -> "MultiDASDataset | None":
        """
        Filters dataset, similar to :py:func:`WaveformDataset.filter`.

        :param mask: Boolean mask to apple to metadata.
        :param inplace: If true, filter inplace.
        """
        submasks = self._split_mask(mask)
        if inplace:
            for submask, dataset in zip(submasks, self.datasets):
                dataset.filter(submask, inplace=True)
            # Calculate new metadata
            self._metadata = pd.concat(x.metadata for x in self.datasets)
            return None

        else:
            return MultiDASDataset(
                [
                    dataset.filter(submask, inplace=False)
                    for submask, dataset in zip(submasks, self.datasets)
                ]
            )

    def get_sample(self, idx: int, *args, **kwargs):
        dataset_idx, local_idx = self._resolve_idx(idx)
        return self.datasets[dataset_idx].get_sample(local_idx, *args, **kwargs)

    _split_mask = MultiWaveformDataset._split_mask
    _resolve_idx = MultiWaveformDataset._resolve_idx
    train = DASDataset.train
    dev = DASDataset.dev
    test = DASDataset.test
    train_dev_test = DASDataset.train_dev_test
    get_split = DASDataset.get_split
    __len__ = DASDataset.__len__


class DASBenchmarkDataset(AbstractBenchmarkDataset, DASDataset, ABC):
    """
    This class is the base class for benchmark DAS datasets. For the functionality, see the superclasses.
    """

    _files = [DASDataset.METADATA_FILE, DASDataset.DATA_FILE]


class RandomDASDataset(DASBenchmarkDataset):
    """
    This is a purely random dataset for testing purposes. It does not contain any actual data and should only be used
    for unit tests.
    """

    def __init__(self, **kwargs):
        super().__init__(compile_from_source=True, repository_lookup=False, **kwargs)

    @classmethod
    def available_chunks(cls, force: bool = False, wait_for_file: bool = False):
        return ["0", "1"]

    def _download_dataset(self, files: list[Path], chunk: str, **kwargs):
        with DASDataWriter(self.path, chunk, files[0], files[1]) as writer:
            for i in range(5):
                data = np.random.randn(500, 600)
                record_metadata = {
                    "record_channel_spacing_m": 4.0,
                    "record_sampling_rate_hz": 125,
                    "record_start_time": datetime.datetime.now(),
                }
                annotations = {
                    "P": np.random.rand(600) * 500,
                    "S": np.random.rand(600) * 500,
                }
                writer.add_record(record_metadata, data, annotations)


class DASDataWriter:
    """
    This class allows writing DAS datasets in SeisBench format. It only writes a single chunk. To write multiple chunks,
    use multiple data writers with different `chunk` arguments but identical `path`.

    :param path: Path to write the chunk to
    :param chunk: Chunk identifier
    :param metadata_path: Overwrite for the metadata path. If provided, writes the metadata here instead of the default
                          location. The chunk key will be ignored in this case. Unless integrated into complex
                          workflows, this parameter should not be used.
    :param data_path: Same as ``.metadata_path`` but for the data file.
    :param data_type: Data type of the data. Defaults to float32.
    :param strict: If true, raise an error if the metadata does not contain the key fields.
                   Otherwise, only raise a warning.
    """

    def __init__(
        self,
        path: Path | str,
        chunk: str = "",
        metadata_path: Path | str | None = None,
        data_path: Path | str | None = None,
        data_type: type[np.floating] | type[np.integer] = np.float32,
        strict: bool = True,
    ):
        self.path = Path(path)
        self.chunk = chunk
        self.data_type = data_type
        self.strict = strict
        self._metadata_path = metadata_path
        self._data_path = data_path

        self.path.mkdir(parents=False, exist_ok=True)

        self._metadata = []
        self._data_file = h5py.File(self.data_path, "w")
        self._data_group = self._data_file.create_group("records")
        self._record_count = 0

    @property
    def metadata_path(self) -> Path:
        if self._metadata_path is None:
            return self.path / DASDataset.METADATA_FILE.replace("$CHUNK", self.chunk)
        else:
            return Path(self._metadata_path)

    @property
    def data_path(self) -> Path:
        if self._data_path is None:
            return self.path / DASDataset.DATA_FILE.replace("$CHUNK", self.chunk)
        else:
            return Path(self._data_path)

    def add_record(
        self,
        metadata: dict[str, Any],
        data: np.ndarray,
        annotations: dict[str, np.ndarray],
    ) -> None:
        """
        Add a record to the dataset. While the data and annotations will immediately be written to disk, the metadata
        will be stored in memory and written to disk when the dataset is closed.

        :param metadata: Metadata of the record. There are no mandatory fields, but warnings will be issued if typical
                         key fields are missing.
        :param data: Data of the record. The data needs to be a 2D array (time, channel).
        :param annotations: Annotations of the record. Each annotation consists of a 1D array with the same length as
                            the number of channels. The entries are in samples along the time axis. For example, an
                            annotation called ``"P"`` indicates the indices of the P wave arrival at each channel.
                            NaN values are allowed. Annotations can differ between the records.
        """
        self._validate_metadata_entry(metadata)
        self._validate_annotations(data, annotations)

        record_name = f"record_{self._record_count}"
        self._record_count += 1
        if "record_name" in metadata:
            seisbench.logger.warning(
                "Column 'record_name' already exists in metadata. Will overwrite it."
            )
        metadata["record_name"] = record_name

        record_group = self._data_group.create_group(record_name)

        record_group.create_dataset("record", data=data, dtype=self.data_type)
        annotations_group = record_group.create_group("annotations")

        for key, value in annotations.items():
            annotations_group.create_dataset(key, data=value)

        self._metadata.append(metadata)

    @staticmethod
    def _validate_annotations(
        data: np.ndarray, annotations: dict[str, np.ndarray]
    ) -> None:
        for key, value in annotations.items():
            if len(value) != data.shape[1]:
                raise ValueError(
                    f"Annotation {key} has length {len(value)}, but data has {data.shape[1]} channels."
                )

    def _validate_metadata_entry(self, metadata: dict[str, Any]) -> None:
        keys = [
            "record_sampling_rate_hz",
            "record_channel_spacing_m",
            "record_start_time",
        ]
        for key in keys:
            if key not in metadata:
                if self.strict:
                    raise ValueError(
                        f"Metadata entry {key} is missing. "
                        f"Add the key or set strict=False to ignore this error."
                    )
                else:
                    seisbench.logger.warning(
                        f"Metadata entry {metadata} is missing some required keys."
                    )

    def _finalize(self):
        metadata = pd.DataFrame(self._metadata)
        metadata.to_parquet(self.metadata_path, index=False)
        self._data_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize()
        if exc_type is None:
            return True
        else:
            seisbench.logger.error(
                f"Error in creating dataset. "
                f"Saved current progress to {self.metadata_path} and {self.data_path}. Error message:\n"
            )
            return False
