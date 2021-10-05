import seisbench
from .base import BenchmarkDataset


class Iquique(BenchmarkDataset):
    """
    Iquique Benchmark Dataset of local events used for training in Woollam (2019)
    study (see citation).

    Splits are set using standard random sampling of :py:class:`seisbench.data.base.BenchmarkDataset`.
    """

    def __init__(self, **kwargs):

        citation = (
            "Woollam, J., Rietbrock, A., Bueno, A. and De Angelis, S., 2019. "
            "Convolutional neural network for seismic phase classification, "
            "performance demonstration over a local seismic network. "
            "Seismological Research Letters, 90(2A), pp.491-502. "
            "https://doi.org/10.1785/0220180312"
        )

        seisbench.logger.warning(
            "Check available storage and memory before downloading and general use "
            "of Iquique dataset. "
            "Dataset size: waveforms.hdf5 ~5Gb, metadata.csv ~2.6Mb"
        )

        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(**kwargs):
        pass
