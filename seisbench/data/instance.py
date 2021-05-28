import seisbench
import seisbench.util
from .base import BenchmarkDataset

import shutil
import bz2
from abc import ABC


class InstanceTypeDataset(BenchmarkDataset, ABC):
    """
    Abstract class for all datasets in the INSTANCE structure.
    Provides a helper function for downloading the datasets to avoid code duplication.
    """

    def _download_helper(
        self, writer, metadata_url, data_url, cleanup=True, blocksize=2 ** 14
    ):
        """
        Download the dataset from the given metadata_url and data_url and unpack it.

        :param writer: WaveformDataWriter
        :param metadata_url: URL of the metadata
        :param data_url: URL of the waveform data
        :param cleanup: If true, delete the original bz2 files after conversion. Defaults to true.
        :param blocksize: Size of the chunks in decompression.
        :return:
        """
        path = self.path
        path_original = path / "original"
        path_original.mkdir(parents=True, exist_ok=True)

        for url, name in zip(
            [metadata_url, data_url], ["metadata.csv.bz2", "waveforms.hdf5.bz2"]
        ):

            def callback_download_original(path):
                seisbench.util.download_http(
                    url,
                    path,
                    desc=f"Downloading {name}",
                )

            seisbench.util.callback_if_uncached(
                path_original / name, callback_download_original, force=True
            )

            if name.startswith("metadata"):
                output_path = writer.metadata_path
            else:
                output_path = writer.waveforms_path

            seisbench.logger.warning(f"Decompressing {name}. This might take a while.")
            with bz2.BZ2File(path_original / name, "r") as fbz2:
                with open(output_path, "wb") as fout:
                    while True:
                        block = fbz2.read(blocksize)
                        if not block:
                            break
                        fout.write(block)

        if cleanup:
            seisbench.logger.warning(
                "Cleaning up source files. This might take a few minutes."
            )
            shutil.rmtree(path_original)


class InstanceNoise(InstanceTypeDataset):
    """
    INSTANCE dataset - Noise samples
    """

    def __init__(self, **kwargs):
        citation = (
            "Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2021). "
            "INSTANCE - The Italian Seismic Dataset For Machine Learning. "
            "Istituto Nazionale di Geofisica e Vulcanologia (INGV). "
            "https://doi.org/10.13127/INSTANCE"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=False, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        metadata_url = "http://repo.pi.ingv.it/instance/metadata_Instance_noise.csv.bz2"
        data_url = "http://repo.pi.ingv.it/instance/Instance_noise.hdf5.bz2"

        self._download_helper(writer, metadata_url, data_url, **kwargs)


class InstanceCounts(InstanceTypeDataset):
    """
    INSTANCE dataset - Events with waveforms in counts
    """

    def __init__(self, **kwargs):
        citation = (
            "Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2021). "
            "INSTANCE - The Italian Seismic Dataset For Machine Learning. "
            "Istituto Nazionale di Geofisica e Vulcanologia (INGV). "
            "https://doi.org/10.13127/INSTANCE"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=False, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        seisbench.logger.warning(
            "The catalog will be downloaded and unpacked. "
            "This will require ~200 GB disk storage. "
            "The resulting catalog has ~160GB. "
            "Please ensure that the storage is available on your disk."
        )

        metadata_url = (
            "http://repo.pi.ingv.it/instance/metadata_Instance_events.csv.bz2"
        )
        data_url = "http://repo.pi.ingv.it/instance/Instance_events_counts.hdf5.bz2"

        self._download_helper(writer, metadata_url, data_url, **kwargs)


class InstanceGM(InstanceTypeDataset):
    """
    INSTANCE dataset - Events with waveforms in ground motion units
    """

    def __init__(self, **kwargs):
        citation = (
            "Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2021). "
            "INSTANCE - The Italian Seismic Dataset For Machine Learning. "
            "Istituto Nazionale di Geofisica e Vulcanologia (INGV). "
            "https://doi.org/10.13127/INSTANCE"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=False, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        seisbench.logger.warning(
            "The catalog will be downloaded and unpacked. "
            "This will require ~310 GB disk storage. "
            "The resulting catalog has ~160GB. "
            "Please ensure that the storage is available on your disk."
        )

        metadata_url = (
            "http://repo.pi.ingv.it/instance/metadata_Instance_events.csv.bz2"
        )
        data_url = "http://repo.pi.ingv.it/instance/Instance_events_gm.hdf5.bz2"

        self._download_helper(writer, metadata_url, data_url, **kwargs)
