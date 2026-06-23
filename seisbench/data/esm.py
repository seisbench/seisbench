from .base import WaveformBenchmarkDataset
from .spectra_base import SpectrumBenchmarkDataset

# TODO: Add to documentation
CITATION = (
    "Cianetti S., Mascandola C., Faenza L., Felicetta C., Russo E., Jozinović D., Münchmeyer J., Luzi L., "
    "Michelini A. (2026). ESM25: A Machine-Learning-Ready Snapshot of the European Engineering Strong-Motion "
    "Database. Istituto Nazionale di Geofisica e Vulcanologia (INGV). https://doi.org/10.13127/ai/esm25"
)
LICENSE = "CC BY 4.0"


class ESM25GoodMP(WaveformBenchmarkDataset):
    """
    The ESM25 (European Strong Motion) dataset. This version has been processed manually and provides acceleration,
    velocity and displacement records. This is the subset with high-quality data.
    """

    def __init__(
        self,
        component_order=(
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


class ESM25GoodCV(WaveformBenchmarkDataset):
    """
    The ESM25 (European Strong Motion) dataset. This version has been processed automatically and provides only
    acceleration records. This is the subset with high-quality data.
    """

    def __init__(
        self,
        component_order=(
            "acc_cv_u",
            "acc_cv_v",
            "acc_cv_w",
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


class ESM25BadCV(WaveformBenchmarkDataset):
    """
    The ESM25 (European Strong Motion) dataset. This version has been processed automatically and provides only
    acceleration records. This is the subset with low-quality data.
    """

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


class ESM25SpectraMP(SpectrumBenchmarkDataset):
    """
    The ESM25 (European Strong Motion) dataset. This dataset contains spectra instead of waveforms.
    This version has been processed manually and provides acceleration and displacement spectra.
    This is the subset with high-quality data.

    The code block below shows an example of how to load the spectra and access the frequency list.

    .. code-block:: python

        data_spec = sbd.ESM25SpectraMP()  # Load the dataset
        spec, meta = data_spec.get_sample(0)  # Load a sample (spectrum and metadata)
        data_spec.frequencies  # Access the frequencies for all samples

    This dataset shares the metadata with :py:class:`ESM25GoodMP`, allowing to load spectra and waveforms beloging
    together directly by index.
    """

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
