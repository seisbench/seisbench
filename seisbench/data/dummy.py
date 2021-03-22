import seisbench
import seisbench.util
from .base import BenchmarkDataset

import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


class DummyDataset(BenchmarkDataset):
    """
    A dummy dataset visualizing the implementation of custom datasets
    """

    def __init__(self, **kwargs):
        citation = (
            "Münchmeyer, Jannes; Bindi, Dino; Sippl, Christian; Leser, Ulf; Tilmann, Frederik (2019): "
            "Magnitude scales, attenuation models and feature matrices for the IPOC catalog. "
            "V. 1.0. GFZ Data Services. https://doi.org/10.5880/GFZ.2.4.2019.004"
        )
        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer, trace_length=60, **kwargs):
        sampling_rate = 20

        writer.metadata_dict = {
            "time": "trace_start_time",
            "latitude": "source_latitude_deg",
            "longitude": "source_longitude_deg",
            "depth": "source_depth_km",
            "cls": "source_event_category",
            "MA": "source_magnitude",
            "ML": "source_magnitude2",
            "std_MA": "source_magnitude_uncertainty",
            "std_ML": "source_magnitude_uncertainty2",
        }
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": sampling_rate,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        path = self.path
        path.mkdir(parents=True, exist_ok=True)

        seisbench.util.download_ftp(
            "datapub.gfz-potsdam.de",
            "download/10.5880.GFZ.2.4.2019.004/IPOC_catalog_magnitudes.csv",
            path / "raw_catalog.csv",
            progress_bar=False,
        )

        metadata = pd.read_csv(path / "raw_catalog.csv")
        metadata = metadata.iloc[:100].copy()

        def to_tracename(x):
            for c in "/:.":
                x = x.replace(c, "_")
            return x

        client = Client("GFZ")
        inv = client.get_stations(
            net="CX",
            sta="PB01",
            starttime=UTCDateTime.strptime(
                "2007/01/01 00:00:00.00", "%Y/%m/%d %H:%M:%S.%f"
            ),
        )

        metadata["trace_name"] = metadata["time"].apply(to_tracename)
        metadata["station_network_code"] = "CX"
        metadata["station_code"] = "PB01"
        metadata["station_type"] = "BH"
        metadata["station_latitude_deg"] = inv[0][0].latitude
        metadata["station_longitude_deg"] = inv[0][0].longitude
        metadata["station_elevation_m"] = inv[0][0].elevation
        metadata["source_magnitude_type"] = "MA"
        metadata["source_magnitude_type2"] = "ML"

        splits = 60 * ["train"] + 10 * ["dev"] + 30 * ["test"]
        metadata["split"] = splits

        writer.set_total(len(metadata))
        for _, row in metadata.iterrows():
            time = row["time"]
            waveform = np.zeros((3, sampling_rate * trace_length))
            time = UTCDateTime.strptime(time, "%Y/%m/%d %H:%M:%S.%f")
            stream = client.get_waveforms(
                "CX", "PB01", "*", "BH?", time, time + trace_length
            )
            for cid, component in enumerate("ZNE"):
                ctrace = stream.select(channel=f"??{component}")[0]
                waveform[cid] = ctrace.data[: sampling_rate * trace_length].astype(
                    float
                )
            writer.add_trace(row, waveform)


class ChunkedDummyDataset(BenchmarkDataset):
    """
    A chunked dummy dataset visualizing the implementation of custom datasets with chunking
    """

    def __init__(self, **kwargs):
        citation = (
            "Münchmeyer, Jannes; Bindi, Dino; Sippl, Christian; Leser, Ulf; Tilmann, Frederik (2019): "
            "Magnitude scales, attenuation models and feature matrices for the IPOC catalog. "
            "V. 1.0. GFZ Data Services. https://doi.org/10.5880/GFZ.2.4.2019.004"
        )

        # Write chunks to file
        chunks_path = self.path / "chunks"
        if not chunks_path.is_file():
            self.path.mkdir(exist_ok=True, parents=True)
            with open(chunks_path, "w") as f:
                f.write("0\n1\n")

        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer, chunk, trace_length=60, **kwargs):
        sampling_rate = 20

        writer.metadata_dict = {
            "time": "trace_start_time",
            "latitude": "source_latitude_deg",
            "longitude": "source_longitude_deg",
            "depth": "source_depth_km",
            "cls": "source_event_category",
            "MA": "source_magnitude",
            "ML": "source_magnitude2",
            "std_MA": "source_magnitude_uncertainty",
            "std_ML": "source_magnitude_uncertainty2",
        }
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": sampling_rate,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        path = self.path
        path.mkdir(parents=True, exist_ok=True)

        seisbench.util.download_ftp(
            "datapub.gfz-potsdam.de",
            "download/10.5880.GFZ.2.4.2019.004/IPOC_catalog_magnitudes.csv",
            path / "raw_catalog.csv",
            progress_bar=False,
        )

        metadata = pd.read_csv(path / "raw_catalog.csv")
        if chunk == "0":
            metadata = metadata.iloc[:100].copy()
        elif chunk == "1":
            metadata = metadata.iloc[100:200].copy()
        else:
            raise ValueError(f'Unknown chunk "{chunk}"')

        def to_tracename(x):
            for c in "/:.":
                x = x.replace(c, "_")
            return x

        client = Client("GFZ")
        inv = client.get_stations(
            net="CX",
            sta="PB01",
            starttime=UTCDateTime.strptime(
                "2007/01/01 00:00:00.00", "%Y/%m/%d %H:%M:%S.%f"
            ),
        )

        metadata["trace_name"] = metadata["time"].apply(to_tracename)
        metadata["station_network_code"] = "CX"
        metadata["station_code"] = "PB01"
        metadata["station_type"] = "BH"
        metadata["station_latitude_deg"] = inv[0][0].latitude
        metadata["station_longitude_deg"] = inv[0][0].longitude
        metadata["station_elevation_m"] = inv[0][0].elevation
        metadata["source_magnitude_type"] = "MA"
        metadata["source_magnitude_type2"] = "ML"

        splits = 60 * ["train"] + 10 * ["dev"] + 30 * ["test"]
        metadata["split"] = splits

        writer.set_total(len(metadata))
        for _, row in metadata.iterrows():
            time = row["time"]
            waveform = np.zeros((3, sampling_rate * trace_length))
            time = UTCDateTime.strptime(time, "%Y/%m/%d %H:%M:%S.%f")
            stream = client.get_waveforms(
                "CX", "PB01", "*", "BH?", time, time + trace_length
            )
            for cid, component in enumerate("ZNE"):
                ctrace = stream.select(channel=f"??{component}")[0]
                waveform[cid] = ctrace.data[: sampling_rate * trace_length].astype(
                    float
                )
            writer.add_trace(row, waveform)
