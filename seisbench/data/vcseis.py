from seisbench.data.base import WaveformBenchmarkDataset


class VCSEIS(WaveformBenchmarkDataset):
    """
    A data set of seismic waveforms from various volcanic regions: Alaska, Hawaii, Northern California, Cascade volcanoes.

    """

    def __init__(self, **kwargs):
        citation = (
            "Zhong, Y., & Tan, Y. J. (2024). Deep-learning-based phase "
            "picking for volcano-tectonic and long-period earthquakes. "
            "Geophysical Research Letters, 51, e2024GL108438. "
            "https://doi.org/10.1029/2024GL108438"
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation,
            license=license,
            repository_lookup=True,
            **kwargs,
        )

    def _download_dataset(self, writer, **kwargs):
        pass

    def get_long_period_earthquakes(self):
        """
        Return the subset with only long-period earthquakes
        """
        return self.filter(
            self.metadata["source_type"] == "lp",
            inplace=False,
        )

    def get_regular_earthquakes(self):
        """
        Return the subset with only regular earthquakes
        """
        return self.filter(
            (
                (self.metadata["source_type"] != "lp")
                & (self.metadata["source_type"] != "noise")
            ),
            inplace=False,
        )

    def get_noise_traces(self):
        """
        Return the subset with only noise traces
        """
        return self.filter(self.metadata["source_type"] == "noise", inplace=False)

    def get_alaska_subset(self):
        """
        Select and return the data from Alaska
        """
        return self.filter(
            self.metadata["trace_chunk"].isin(["_ak_lp", "_ak_rg", "_aknoise"]),
            inplace=False,
        )

    def get_hawaii_subset(self):
        """
        Select and return the data from Hawaii
        """
        return self.filter(
            self.metadata["trace_chunk"].isin(
                ["_hw12t21_lp", "_hw12t21_rg", "_hwnoise"]
            ),
            inplace=False,
        )

    def get_northern_california_subset(self):
        """
        Select and return the data from Northern California
        """
        return self.filter(
            self.metadata["trace_chunk"].isin(["_ncedc_lp", "_ncedc_vt"]), inplace=False
        )

    def get_cascade_subset(self):
        """
        Select and return the data from Cascade volcanoes
        """
        return self.filter(
            self.metadata["trace_chunk"].isin(["_cascade_lp", "_cascade_vt"]),
            inplace=False,
        )
