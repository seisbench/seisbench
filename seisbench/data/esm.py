from .base import WaveformBenchmarkDataset
from .spectra_base import SpectrumBenchmarkDataset

# TODO: Add to documentation
# TODO: Add docstring, citation & license for all
CITATION = "TBD"
LICENSE = "TBD"


class ESM25WaveformsManual(WaveformBenchmarkDataset):
    def __init__(
        self,
        component_order=(
            "acc_cv_u",
            "acc_cv_v",
            "acc_cv_w",
            "acc_mp_u",
            "acc_mp_v",
            "acc_mp_w",
            "vel_mp_u",
            "vel_mp_v",
            "vel_mp_w",
            "dis_mp_u",
            "dis_mp_v",
            "dis_mp_w",
        ),
        **kwargs,
    ):
        super().__init__(
            component_order=component_order,
            license=LICENSE,
            citation=CITATION,
            repository_lookup=True,
            **kwargs,
        )

    def _download_dataset(self, writer, **kwargs):
        pass


class ESM25WaveformsConverted(WaveformBenchmarkDataset):
    def __init__(self, component_order=("acc_cv_u", "acc_cv_v", "acc_cv_w"), **kwargs):
        super().__init__(
            component_order=component_order,
            license=LICENSE,
            citation=CITATION,
            repository_lookup=True,
            **kwargs,
        )

    def _download_dataset(self, writer, **kwargs):
        pass


class ESM25SpectraManual(SpectrumBenchmarkDataset):
    def __init__(
        self,
        component_order=(
            "acc_mp_u",
            "acc_mp_v",
            "acc_mp_w",
            "dis_mp_u",
            "dis_mp_v",
            "dis_mp_w",
        ),
        **kwargs,
    ):
        super().__init__(
            component_order=component_order,
            license=LICENSE,
            citation=CITATION,
            repository_lookup=True,
            **kwargs,
        )

    def _download_dataset(self, writer, **kwargs):
        pass


# TODO: Implement
class ESM25Manual:
    """
    A convenience class combining :py:class:`ESM25WaveformsManual` and :py:class:`ESM25SpectraManual` into a joint class
    that can be accessed more easily.
    """

    pass
