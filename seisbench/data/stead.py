import seisbench
from .base import BenchmarkDataset

from pathlib import Path
import shutil
import h5py
import pandas as pd


class STEAD(BenchmarkDataset):
    """
    STEAD dataset
    """

    def __init__(self, **kwargs):
        citation = (
            "Mousavi, S. M., Sheng, Y., Zhu, W., Beroza G.C., (2019). STanford EArthquake Dataset (STEAD): "
            "A Global Data Set of Seismic Signals for AI, IEEE Access, doi:10.1109/ACCESS.2019.2947848"
        )
        super().__init__(
            name=self.__class__.__name__.lower(), citation=citation, **kwargs
        )

    def _download_dataset(self, writer, basepath=None, **kwargs):
        download_instructions = (
            "Please download STEAD following the instructions at https://github.com/smousavi05/STEAD. "
            "Provide the locations of the STEAD unpacked files (merged.csv and merged.hdf5) as parameter basepath to the class. "
            "This step is only necessary the first time STEAD is loaded."
        )

        metadata_dict = {
            "trace_start_time": "trace_start_time",
            "trace_category": "trace_category",
            "trace_name": "trace_name",
            "p_arrival_sample": "trace_p_arrival_sample",
            "p_status": "trace_p_status",
            "p_weight": "trace_p_weight",
            "p_travel_sec": "trace_p_travel_sec",
            "s_arrival_sample": "trace_s_arrival_sample",
            "s_status": "trace_s_status",
            "s_weight": "trace_s_weight",
            "s_travel_sec": "trace_s_travel_sec",
            "back_azimuth_deg": "trace_back_azimuth_deg",
            "snr_db": "trace_snr_db",
            "coda_end_sample": "trace_coda_end_sample",
            "network_code": "station_network_code",
            "receiver_code": "station_code",
            "receiver_type": "station_type",
            "receiver_latitude": "station_latitude_deg",
            "receiver_longitude": "station_longitude_deg",
            "receiver_elevation_m": "station_elevation_m",
            "source_id": "source_id",
            "source_origin_time": "source_origin_time",
            "source_origin_uncertainty_sec": "source_origin_uncertainty_sec",
            "source_latitude": "source_latitude_deg",
            "source_longitude": "source_longitude_deg",
            "source_error_sec": "source_error_sec",
            "source_gap_deg": "source_gap_deg",
            "source_horizontal_uncertainty_km": "source_horizontal_uncertainty_km",
            "source_depth_km": "source_depth_km",
            "source_depth_uncertainty_km": "source_depth_uncertainty_km",
            "source_magnitude": "source_magnitude",
            "source_magnitude_type": "source_magnitude_type",
            "source_magnitude_author": "source_magnitude_author",
        }

        if basepath is None:
            raise ValueError(
                "No cached version of STEAD found. " + download_instructions
            )

        basepath = Path(basepath)

        if not (basepath / "merged.csv").is_file():
            raise ValueError(
                "Basepath does not contain file merged.csv. " + download_instructions
            )
        if not (basepath / "merged.hdf5").is_file():
            raise ValueError(
                "Basepath does not contain file merged.hdf5. " + download_instructions
            )

        self.path.mkdir(parents=True, exist_ok=True)
        seisbench.logger.warning(
            "Copying STEAD files to cache. This might take a while."
        )

        # Copy metadata and rename columns to SeisBench format
        metadata = pd.read_csv(basepath / "merged.csv")
        metadata.rename(columns=metadata_dict, inplace=True)
        metadata.to_csv(self.path / "metadata.csv", index=False)

        shutil.copy(basepath / "merged.hdf5", self.path / "waveforms.hdf5")

        data_format = {
            "dimension_order": "WC",
            "component_order": "ENZ",
            "sampling_rate": 100,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        with h5py.File(self.path / "waveforms.hdf5", "a") as fout:
            g_data_format = fout.create_group("data_format")
            for key in data_format.keys():
                g_data_format.create_dataset(key, data=data_format[key])
