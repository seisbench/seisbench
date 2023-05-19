from seisbench.data.base import BenchmarkDataset


class PNW(BenchmarkDataset):
    """
    PNW ComCat dataset from Ni et al. (2023)

    """

    def __init__(self, **kwargs):
        citation = (
            "Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, "
            "S., Bodin, P., Hartog, R., & Wright, A. (2023)."
            "Curated Pacific Northwest AI-ready Seismic Dataset."
            " Seismica, 2(1). https://doi.org/10.26443/seismica.v2i1.368"
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass


class PNWExotic(BenchmarkDataset):
    """
    PNW Exotic dataset from Ni et al. (2023)

    """

    def __init__(self, **kwargs):
        citation = (
            "Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, "
            "S., Bodin, P., Hartog, R., & Wright, A. (2023)."
            "Curated Pacific Northwest AI-ready Seismic Dataset."
            " Seismica, 2(1). https://doi.org/10.26443/seismica.v2i1.368"
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass


class PNWAccelerometers(BenchmarkDataset):
    """
    PNW Accelerometers dataset from Ni et al. (2023)

    """

    def __init__(self, **kwargs):
        citation = (
            "Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, "
            "S., Bodin, P., Hartog, R., & Wright, A. (2023)."
            "Curated Pacific Northwest AI-ready Seismic Dataset."
            " Seismica, 2(1). https://doi.org/10.26443/seismica.v2i1.368"
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass


class PNWNoise(BenchmarkDataset):
    """
    PNW Noise dataset from Ni et al. (2023)

    """

    def __init__(self, **kwargs):
        citation = (
            "Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, "
            "S., Bodin, P., Hartog, R., & Wright, A. (2023)."
            "Curated Pacific Northwest AI-ready Seismic Dataset."
            " Seismica, 2(1). https://doi.org/10.26443/seismica.v2i1.368"
        )
        license = "CC BY 4.0"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass
