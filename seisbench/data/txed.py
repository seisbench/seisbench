import pickle
from pathlib import Path

import h5py

import seisbench

from .base import WaveformBenchmarkDataset, WaveformDataWriter


class TXED(WaveformBenchmarkDataset):
    """
    TEXD dataset from Chen et al.

    train/dev/test split defined in SeisBench.
    """

    def __init__(self, **kwargs):
        citation = (
            "Chen, Y., A. Savvaidis, O. M. Saad, G.-C. D. Huang, D. Siervo, V. O’Sullivan, C. McCabe, B. Uku, "
            "P. Fleck, G. Burke, N. L. Alvarez, J. Domino, and I. Grigoratos, "
            "“TXED: the texas earthquake dataset for AI,” Seismological Research Letters, vol. 1, no. 1, "
            "p. doi: 10.1785/0220230327, 2024."
        )
        license = "GPLv3"
        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer: WaveformDataWriter, basepath=None, **kwargs):
        download_instructions = (
            "Please download TXED following the instructions at https://github.com/chenyk1990/txed/. "
            "Provide the locations of the TEXD files (merged.csv and merged.hdf5) in the "
            "download_kwargs argument 'basepath'. "
            "This step is only necessary the first time TEXD is loaded."
        )

        metadata_dict = {
            "causal": "trace_causal",
            "coda_end_sample": "trace_coda_end_sample",
            "ev_depth": "source_depth_km",
            "ev_latitude": "source_latitude_deg",
            "ev_longitude": "source_longitude_deg",
            "magnitude": "source_magnitude",
            "origin_time": "source_origin_time",
            "p_arrival_sample": "trace_p_arrival_sample",
            "p_arrival_time": "trace_p_arrival_time",
            "p_uncertainty": "trace_p_arrival_uncertainty_s",
            "polarity": "trace_polarity",
            "s_arrival_sample": "trace_s_arrival_sample",
            "s_arrival_time": "trace_s_arrival_time",
            "s_uncertainty": "trace_s_arrival_uncertainty_s",
            "snr_db": "trace_snr_db",
            "sta_elevation": "station_elevation_m",
            "sta_latitude": "station_latitude_deg",
            "sta_longitude": "station_longitude_deg",
            "station": "station_code",
            "trace_category": "trace_category",
        }
        float_conversions = [
            "trace_coda_end_sample",
            "source_depth_km",
            "source_latitude_deg",
            "source_longitude_deg",
            "source_magnitude",
            "trace_p_arrival_sample",
            "trace_p_arrival_uncertainty_s",
            "trace_s_arrival_sample",
            "trace_s_arrival_uncertainty_s",
            "station_elevation_m",
            "station_latitude_deg",
            "station_longitude_deg",
        ]

        if basepath is None:
            raise ValueError(
                "No cached version of TEXD found. " + download_instructions
            )

        basepath = Path(basepath)

        if not (basepath / "TXED_20231111.h5").is_file():
            raise ValueError(
                "Basepath does not contain file TXED_20231111.h5. "
                + download_instructions
            )

        with open(basepath / "split.pkl", "rb") as f:
            split = pickle.load(f)
        split = {k: set(v) for k, v in split.items()}

        self.path.mkdir(parents=True, exist_ok=True)
        seisbench.logger.warning(
            "Converting TEXD files to SeisBench format. This might take a while."
        )

        # Writer data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": 100,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        with h5py.File(basepath / "TXED_20231111.h5") as f:
            writer.set_total(len(f))
            for trace_name in f:
                g = f[trace_name]
                org_metadata = dict(g.attrs)
                metadata = {metadata_dict[k]: v for k, v in org_metadata.items()}
                for k in float_conversions:
                    if k in metadata:
                        metadata[k] = float(metadata[k])

                metadata["source_depth_km"] /= 1e3  # m to km
                metadata["trace_name_original"] = trace_name

                for k, v in split.items():
                    if trace_name in v:
                        metadata["split"] = k
                        break
                else:
                    raise ValueError(f"Trace {trace_name} not contained in any split")

                waveforms = g["data"][()]
                waveforms = waveforms.T  # From WC to CW
                # Component order is ZNE already
                writer.add_trace(metadata, waveforms)
