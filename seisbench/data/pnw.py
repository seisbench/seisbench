import random
import string

import numpy as np
from obspy.geodetics import gps2dist_azimuth

import seisbench
from seisbench.data.base import BenchmarkDataset
from seisbench.util.trace_ops import (
    rotate_stream_to_zne,
    stream_to_array,
    trace_has_spikes,
    waveform_id_to_network_station_location,
)


class PNW(BenchmarkDataset):
    """
    PNW (ComCat) dataset from Ni et al. (2023)

    """

    def __init__(self, **kwargs):
        citation = (
            "Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, "
            "S., Bodin, P., Hartog, R., & Wright, A. (2023)."
            "Curated Pacific Northwest AI-ready Seismic Dataset."
            " Seismica, 2(1). https://doi.org/10.26443/seismica.v2i1.368"
        )
        license = "CC BY 4.0"

        seisbench.logger.warning("None")

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass
