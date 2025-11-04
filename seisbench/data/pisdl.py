from seisbench.data.base import WaveformBenchmarkDataset


class PiSDL(WaveformBenchmarkDataset):
    """
    A dataset for induced seismicity from different regions in Canada, Switzerland,
    Germany, and France. Induced seismic events are caused by hydraulic-fracturing
    based fluid injection, geothermal power plants, and coal mine flooding.
    In addition, the dataset contains all available low magnitude events (M_L <= 2)
    from the Swiss Seismological Service (SED) between 2009 and 2023.
    """

    def __init__(self, **kwargs):
        citation = (
            "Heuel, J., Maurer, V., Frietsch, M., Rietbrock, A. (2025)."
            "Picking Induced Seismicity with Deep Learning (piSDL) "
            "Seismica, 4 (2). "
            "https://doi.org/10.26443/seismica.v4i2.1579"
        )

        super().__init__(
            citation=citation,
            repository_lookup=True,
            **kwargs,
        )

    def get_dawson_septimus_subset(self):
        return self.filter(
            mask=(
                (self.metadata["source_latitude_deg"] >= 55.5)
                & (self.metadata["source_latitude_deg"] <= 56.5)
                & (self.metadata["source_longitude_deg"] >= -121.5)
                & (self.metadata["source_longitude_deg"] <= -119.8)
            ),
            inplace=False,
        )

    def get_insheim_subset(self):
        return self.filter(
            mask=(
                (self.metadata["source_latitude_deg"] >= 49.14)
                & (self.metadata["source_latitude_deg"] <= 49.17)
                & (self.metadata["source_longitude_deg"] >= 8.12)
                & (self.metadata["source_longitude_deg"] <= 8.18)
            ),
            inplace=False,
        )

    def get_st_gallen_subset(self):
        return self.filter(
            mask=(
                (self.metadata["source_latitude_deg"] >= 47.405)
                & (self.metadata["source_latitude_deg"] <= 47.436)
                & (self.metadata["source_longitude_deg"] >= 9.304)
                & (self.metadata["source_longitude_deg"] <= 9.334)
            ),
            inplace=False,
        )

    def get_switzerland_subset(self):
        return self.filter(
            mask=(
                (self.metadata["source_latitude_deg"] >= 45.40)
                & (self.metadata["source_latitude_deg"] <= 48.3)
                & (self.metadata["source_longitude_deg"] >= 5.68)
                & (self.metadata["source_longitude_deg"] <= 11.1)
            ),
            inplace=False,
        )

    def get_floodrisk_subset(self):
        return self.filter(
            mask=(
                (self.metadata["source_latitude_deg"] >= 51.6)
                & (self.metadata["source_latitude_deg"] <= 51.7)
                & (self.metadata["source_longitude_deg"] >= 7.6)
                & (self.metadata["source_longitude_deg"] <= 7.8)
            ),
            inplace=False,
        )

    def get_vendenheim_subset(self):
        return self.filter(
            mask=(
                (self.metadata["source_latitude_deg"] >= 48.52)
                & (self.metadata["source_latitude_deg"] <= 48.67)
                & (self.metadata["source_longitude_deg"] >= 7.76)
                & (self.metadata["source_longitude_deg"] <= 7.82)
            ),
            inplace=False,
        )

    def _download_dataset(self, writer, **kwargs):
        pass
