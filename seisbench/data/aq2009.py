from seisbench.data.base import WaveformBenchmarkDataset


class AQ2009Counts(WaveformBenchmarkDataset):
    """
    AQ2009 aftershocks digital units dataset from Bagagli et al. (2023)

    """

    def __init__(self, **kwargs):
        citation = (
            "Bagagli, M., Valoroso, L., Michelini, A., Cianetti, S., "
            "Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2023). "
            "AQ2009 – The 2009 Aquila Mw 6.1 earthquake aftershocks seismic "
            "dataset for machine learning application. Istituto Nazionale di "
            "Geofisica e Vulcanologia (INGV). "
            "https://doi.org/10.13127/AI/AQUILA2009"
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass


class AQ2009GM(WaveformBenchmarkDataset):
    """
    AQ2009 aftershocks ground motion dataset from Bagagli et al. (2023)

    """

    def __init__(self, **kwargs):
        citation = (
            "Bagagli, M., Valoroso, L., Michelini, A., Cianetti, S., "
            "Gaviano, S., Giunchi, C., Jozinović, D., & Lauciani, V. (2023). "
            "AQ2009 – The 2009 Aquila Mw 6.1 earthquake aftershocks seismic "
            "dataset for machine learning application. Istituto Nazionale di "
            "Geofisica e Vulcanologia (INGV). "
            "https://doi.org/10.13127/AI/AQUILA2009"
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass
