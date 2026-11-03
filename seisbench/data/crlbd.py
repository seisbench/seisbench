from abc import ABC
import shutil
import tarfile
from pathlib import Path

import requests

import seisbench
from seisbench import util as sbu

from .base import MultiWaveformDataset, WaveformBenchmarkDataset


class CRLBDBase(WaveformBenchmarkDataset, ABC):
    """
    Base class with the functionality to download the CRLBD datasets from Zenodo
    (https://doi.org/10.5281/zenodo.18768357).

    :param kwargs: Passed to :py:class:`WaveformBenchmarkDataset`.
    """

    _suffix = ""
    doi = "10.5281/zenodo.18768357"

    def __init__(self, *, component_order="ZNE", compile_from_source=True, **kwargs):
        citation = (
            "Spingos, I., Kapetanidis, V., Münchmeyer, J., Zymvragakis, A., "
            "Karakonstantis, A., Voulgaris, N., & Kaviris, G. (2026). "
            "CRLBD - The Corinth Rift Laboratory Benchmark Dataset [Dataset]. "
            "In Artificial Intelligence in Geosciences (Vol. 7, Issue 4, 100290). "
            "Zenodo. https://doi.org/10.5281/zenodo.18768357"
        )
        license = "CC BY 4.0"

        super().__init__(
            component_order=component_order,
            citation=citation,
            license=license,
            repository_lookup=True,
            compile_from_source=compile_from_source,
            **kwargs,
        )

    def _get_record_id(self):
        doi_url = f"https://doi.org/{self.doi}"
        response = requests.head(doi_url, allow_redirects=True)
        response.raise_for_status()
        return response.url.rstrip("/").split("/")[-1]

    def _download_dataset(self, writer, **kwargs):
        record_id = self._get_record_id()
        archive_filename = f"crlbd{self._suffix}.tar.gz"
        archive_path = self.path / archive_filename
        download_url = f"https://zenodo.org/records/{record_id}/files/{archive_filename}?download=1"

        seisbench.logger.info(
            f"Downloading CRLBD archive {archive_filename} from Zenodo record {record_id}"
        )
        sbu.download_http(
            download_url,
            archive_path,
            desc=f"Downloading {archive_filename}",
        )

        seisbench.logger.info("Extracting dataset files...")
        self._extract_archive(archive_path, writer)
        archive_path.unlink()

        seisbench.logger.info("Download and extraction complete")

    @staticmethod
    def _extract_archive(archive_path, writer):
        """
        Streams ``metadata.csv`` and ``waveforms.hdf5`` from the archive directly into the writer paths in a single
        pass. Member names are only used for matching, never as output paths.
        """
        targets = {
            "metadata.csv": writer.metadata_path,
            "waveforms.hdf5": writer.waveforms_path,
        }
        extracted = set()

        with tarfile.open(archive_path, "r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile() or name not in targets:
                    continue
                with tar.extractfile(member) as src, open(targets[name], "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.add(name)

        missing = set(targets) - extracted
        if missing:
            raise ValueError(
                f"Archive {archive_path.name} is missing: {', '.join(sorted(missing))}."
            )


class CRLBD(CRLBDBase):
    """
    CRLBD dataset - Event waveforms

    Local earthquake recordings from the Corinth Rift Laboratory Near-Fault Observatory (Western Gulf of Corinth, Greece).
    Dataset DOI: https://doi.org/10.5281/zenodo.18768357
    """


class CRLBDNoise(CRLBDBase):
    """
    CRLBD dataset - Noise waveforms

    Noise recordings from the Corinth Rift Laboratory Near-Fault Observatory (Western Gulf of Corinth, Greece).
    Dataset DOI: https://doi.org/10.5281/zenodo.18768357
    """

    _suffix = "noise"


class CRLBDCombined(MultiWaveformDataset):
    """
    Convenience class to jointly load :py:class:`CRLBD` and :py:class:`CRLBDNoise`.

    :param kwargs: Passed to the constructors of both :py:class:`CRLBD` and :py:class:`CRLBDNoise`
    """

    def __init__(self, **kwargs):
        super().__init__([CRLBD(**kwargs), CRLBDNoise(**kwargs)])
