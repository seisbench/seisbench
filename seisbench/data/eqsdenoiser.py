from abc import ABC
import requests

import seisbench.util as sbu
from . import MultiWaveformDataset, WaveformBenchmarkDataset


class EQSDenoiserBase(WaveformBenchmarkDataset, ABC):
    """
    Base class with the functionality to download the EQSDenoiser datasets. As the original dataset is precompiled on
    Zenodo and therefore loading from source is fast, the class overwrites the default for ``compile_from_source`` to
    ``True``.
    """

    _suffix: str = None
    doi = "10.5281/zenodo.17865808"

    def __init__(self, *, component_order="Z12", compile_from_source=True, **kwargs):
        citation = (
            "Dahmen, N., Clinton, J., Meier, M.-A., Scarabello, L. (2025). "
            "Dataset: Earthquake Seismogram Denoiser (EQS-Denoiser) [Data set]. "
            "In Bulletin of the Seismological Society of America. "
            "Zenodo. https://doi.org/10.5281/zenodo.17865808"
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

    def _get_record_id(self) -> str:
        doi_url = f"https://doi.org/{self.doi}"
        response = requests.head(doi_url, allow_redirects=True)
        record_url = response.url  # e.g., https://zenodo.org/record/1234567
        return record_url.rstrip("/").split("/")[-1]

    def _download_dataset(self, writer, **kwargs):
        record_id = self._get_record_id()
        base_url = f"https://zenodo.org/records/{record_id}/files/"
        sbu.download_http(
            url=base_url + f"metadata{self._suffix}.csv",
            target=writer.metadata_path,
            progress_bar=True,
            desc="Downloading metadata",
            precheck_timeout=0,
        )
        sbu.download_http(
            url=base_url + f"waveforms{self._suffix}.hdf5",
            target=writer.waveforms_path,
            progress_bar=True,
            desc="Downloading waveforms",
            precheck_timeout=0,
        )


class EQSDenoiserEvents(EQSDenoiserBase):
    """
    EQSDenoiser dataset - Event waveforms
    """

    _suffix = ""


class EQSDenoiserNoise(EQSDenoiserBase):
    """
    EQSDenoiser dataset - Noise waveforms
    """

    _suffix = "_noise"


class EQSDenoiserCombined(MultiWaveformDataset):
    """
    Convenience class to jointly load :py:class:`EQSDenoiserEvents` and :py:class:`EQSDenoiserNoise`.

    :param kwargs: Passed to the constructors of both :py:class:`EQSDenoiserEvents` and :py:class:`EQSDenoiserNoise`
    """

    def __init__(self, **kwargs):
        super().__init__([EQSDenoiserEvents(**kwargs), EQSDenoiserEvents(**kwargs)])
