from .base import WaveformBenchmarkDataset


class LFEStacksCascadiaBostock2015(WaveformBenchmarkDataset):
    """
    Low-frequency earthquake stacks underneath Vancouver Island, Cascadia, Canada/USA based on the catalog by
    Bostock et al (2015). Compiled to SeisBench format by Münchmeyer et al (2024).
    """

    def __init__(self, component_order="Z12", **kwargs):
        citation = (
            "Münchmeyer, J., Giffard-Roisin, S., Malfante, M., Frank, W., Poli, P., Marsan, D., Socquet A. (2024). "
            "Deep learning detects uncataloged low-frequency earthquakes across regions. Seismica."
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation,
            license=license,
            repository_lookup=True,
            component_order=component_order,
            **kwargs,
        )

    def _download_dataset(self, writer, **kwargs):
        pass


class LFEStacksMexicoFrank2014(WaveformBenchmarkDataset):
    """
    Low-frequency earthquake stacks underneath Guerrero, Mexico based on the catalog by
    Frank et al (2014). Compiled to SeisBench format by Münchmeyer et al (2024).
    """

    def __init__(self, component_order="Z12", **kwargs):
        citation = (
            "Münchmeyer, J., Giffard-Roisin, S., Malfante, M., Frank, W., Poli, P., Marsan, D., Socquet A. (2024). "
            "Deep learning detects uncataloged low-frequency earthquakes across regions. Seismica."
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation,
            license=license,
            repository_lookup=True,
            component_order=component_order,
            **kwargs,
        )

    def _download_dataset(self, writer, **kwargs):
        pass


class LFEStacksSanAndreasShelly2017(WaveformBenchmarkDataset):
    """
    Low-frequency earthquake stacks on the San Andreas Fault, California, USA based on the catalog by
    Shelly (2014). Compiled to SeisBench format by Münchmeyer et al (2024).
    """

    def __init__(self, component_order="Z12", **kwargs):
        citation = (
            "Münchmeyer, J., Giffard-Roisin, S., Malfante, M., Frank, W., Poli, P., Marsan, D., Socquet A. (2024). "
            "Deep learning detects uncataloged low-frequency earthquakes across regions. Seismica."
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation,
            license=license,
            repository_lookup=True,
            component_order=component_order,
            **kwargs,
        )

    def _download_dataset(self, writer, **kwargs):
        pass
