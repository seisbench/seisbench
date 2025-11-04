from .base import WaveformBenchmarkDataset


class OBST2024(WaveformBenchmarkDataset):
    """
    The OBS dataset from Niksejel & Zhang (2024)
    """

    def __init__(self, **kwargs):
        citation = (
            "Niksejel, A. and Zhang, M., 2024. OBSTransformer: a deep-learning seismic "
            "phase picker for OBS data using automated labelling and transfer learning. "
            "Geophysical Journal International, p.ggae049, https://doi.org/10.1093/gji/ggae049."
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass
