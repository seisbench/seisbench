import shutil
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

import seisbench

from .base import WaveformBenchmarkDataset, WaveformDataWriter

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None


class CEED(WaveformBenchmarkDataset):
    """
    The CEED dataset for California from Zhu et al. (2025)
    """

    def __init__(self, **kwargs):
        citation = (
            "Zhu, W., Wang, H., Rong, B., Yu, E., Zuzlewski, S., Tepp, G., "
            "... & Allen, R. M. (2025). California Earthquake Dataset for "
            "Machine Learning and Cloud Computing. arXiv preprint arXiv:2502.11500."
        )
        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    @classmethod
    def available_chunks(cls, force=False, wait_for_file=False):
        nc_chunks = [str(x) for x in range(1987, 2024)]
        sc_chunks = [str(x) for x in range(1999, 2019)] + [
            "2019_0",
            "2019_1",
            "2019_2",
            "2020_0",
            "2020_1",
            "2021",
            "2022",
            "2023",
        ]

        return [f"nc{x}" for x in nc_chunks] + [f"sc{x}" for x in sc_chunks]

    @staticmethod
    def _ensure_hf_hub_download_available():
        assert hf_hub_download is not None, (
            "To download this dataset, huggingface_hub must be installed. "
            "For installation instructions, "
            "see https://huggingface.co/docs/huggingface_hub/installation"
        )

    def _download_dataset(self, writer: WaveformDataWriter, chunk: str, **kwargs):
        path = writer.waveforms_path.parent

        seisbench.logger.warning(f"Start chunk {chunk} from Huggingface Hub")

        area, year = chunk[:2], chunk[2:]
        filename = f"waveform_h5/{year}.h5"

        # download from huggingface hub
        hf_hub_download(
            repo_id=f"AI4EPS/quakeflow_{area}",
            filename=filename,
            repo_type="dataset",
            local_dir=path,
        )

        shutil.move(path / filename, writer.waveforms_path)

        metadata = self._create_metadata(writer.waveforms_path, chunk)
        self._add_split(metadata)
        self._adjust_hdf5(writer.waveforms_path)

        metadata.to_csv(writer.metadata_path, index=False)

    @staticmethod
    def _adjust_hdf5(path: Path) -> None:
        with h5py.File(path, "a") as f:
            # SeisBench needs a data format group
            g = f.create_group("data_format")
            g.create_dataset("dimension_order", data="CW")
            # Add a softlink from "data/" to "/" as SeisBench expects all waveforms under "data/"
            f["data"] = h5py.SoftLink("/")

    @staticmethod
    def _add_split(metadata: pd.DataFrame) -> None:
        # Temporal split, oriented after the split from the original publication with an extra dev set
        metadata["split"] = "train"
        metadata.loc[metadata["source_origin_time"] > "2020", "split"] = "dev"
        metadata.loc[metadata["source_origin_time"] > "2021", "split"] = "test"

    def _create_metadata(self, path: Path, chunk: str) -> pd.DataFrame:
        metadata = []
        with h5py.File(path, "r") as f:
            for s_event, g_event in tqdm(
                f.items(), desc=f"Compiling metadata for chunk {chunk}"
            ):
                event_metadata = {k: v for k, v in g_event.attrs.items()}
                # The key "depth_km" is used on both event and station level, so we need to rename it here
                event_metadata["source_depth_km"] = event_metadata.get("depth_km")
                for s_trace, g_trace in g_event.items():
                    trace_metadata = {
                        **event_metadata,
                        **{k: v for k, v in g_trace.attrs.items()},
                        "trace_name": f"{s_event}/{s_trace}",
                    }
                    metadata.append(trace_metadata)

        renames = {
            "component": "trace_component_order",
            "sampling_rate": "trace_sampling_rate_hz",
            "begin_time": "trace_start_time",
            "end_time": "trace_end_time",
            "snr": "trace_snr_db",
            "depth_km": "station_depth_km",
            "event_id": "source_id_list",
            "event_time": "source_origin_time",
            "event_time_index": "source_origin_time_sample",
            "latitude": "source_latitude_deg",
            "longitude": "source_longitude_deg",
            "magnitude": "source_magnitude",
            "magnitude_type": "source_magnitude_type",
            "nt": "trace_npts",
            "source": "trace_source_region",
            "azimuth": "path_azimuth_deg",
            "back_azimuth": "path_back_azimuth_deg",
            "takeoff_angle": "path_takeoff_angle_deg",
            "distance_km": "path_ep_distance_km",
            "elevation_m": "station_elevation_m",
            "local_depth_m": "station_local_depth_m",
            "instrument": "station_instrument",
            "station": "station_code",
            "network": "station_network_code",
            "location": "station_location_code",
            "unit": "trace_unit",
            # P phase attributes
            "p_phase_index": "trace_p_arrival_sample",
            "p_phase_polarity": "trace_p_polarity",
            "p_phase_score": "trace_p_score",
            "p_phase_status": "trace_p_status",
            "p_phase_time": "trace_p_time",
            # S phase attributes
            "s_phase_index": "trace_s_arrival_sample",
            "s_phase_polarity": "trace_s_polarity",
            "s_phase_score": "trace_s_score",
            "s_phase_status": "trace_s_status",
            "s_phase_time": "trace_s_time",
            # These are a range of attributes in list form.
            # They lose their format when saved as csv, but at least they are documented.
            "phase_index": "trace_phase_arrival_sample_list",
            "phase_picking_channel": "trace_phase_picking_channel_list",
            "phase_polarity": "trace_phase_polarity_list",
            "phase_remark": "trace_phase_remark_list",
            "phase_score": "trace_phase_score_list",
            "phase_status": "trace_phase_status_list",
            "phase_time": "trace_phase_time_list",
            "phase_type": "trace_phase_type_list",
        }
        drops = [
            "nx",  # Number of stations for event. Can be easily recalculated.
            "dt_s",  # Redundant with sampling rate
        ]

        metadata = pd.DataFrame(metadata)

        metadata.drop(columns=drops, inplace=True)
        metadata.rename(columns=renames, inplace=True)

        # The component order in the data is always ENZ
        metadata["trace_component_order"] = "ENZ"

        return metadata
