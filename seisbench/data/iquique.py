from seisbench.data.bnase import BenchmarkDataset


class Iquique(BenchmarkDataset):
    """
    Iquique Benchmark Dataset of local events used for training in Woollam (2019)
    study (see citation).

    Splits are set using standard random sampling of :py:class: BenchmarkDataset.
    """

    def __init__(self, **kwargs):

        citation = (
            "Woollam, J., Rietbrock, A., Bueno, A. and De Angelis, S., 2019. "
            "Convolutional neural network for seismic phase classification, "
            "performance demonstration over a local seismic network. "
            "Seismological Research Letters, 90(2A), pp.491-502. "
            "https://doi.org/10.1785/0220180312"
        )

        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(**kwargs):
        pass
