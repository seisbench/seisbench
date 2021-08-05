import seisbench
import seisbench.util
from .base import BenchmarkDataset

import os
import h5py
from obspy import UTCDateTime


class LenDB(BenchmarkDataset):
    """
    Len-DB dataset from Magrini et al.
    """

    def __init__(self, **kwargs):
        citation = (
            "Magrini, Fabrizio, Jozinović, Dario, Cammarano, Fabio, Michelini, Alberto, & Boschi, Lapo. "
            "(2020). LEN-DB - Local earthquakes detection: a benchmark dataset of 3-component seismograms "
            "built on a global scale [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3648232"
        )
        license = "CC BY 4.0"
        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, cleanup=False):
        """
        Downloads and converts the dataset from the original publication

        :param writer: WaveformWriter
        :param cleanup: If true, delete the original hdf5 file after conversion. Defaults to false.
        :return:
        """
        path = self.path
        path.mkdir(parents=True, exist_ok=True)

        path_original = path / "LEN-DB.hdf5"

        # Uses callback_if_uncached only to be able to utilize the cache mechanism
        # Concurrent accesses are anyhow already controlled by the callback_if_uncached call wrapping _download_dataset
        # It's therefore considered save to set force=True
        def callback_download_original(path):
            seisbench.util.download_http(
                "https://zenodo.org/record/3648232/files/LEN-DB.hdf5?download=1",
                path,
                desc="Downloading original dataset",
            )

        seisbench.util.callback_if_uncached(
            path_original, callback_download_original, force=True
        )

        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "sampling_rate": 20,
            "unit": "km/s",
            "instrument_response": "restituted",
        }

        with h5py.File(path_original, "r") as f:
            # Set total number of traces for progress bar
            writer.set_total(len(f["AN"].keys()) + len(f["EQ"].keys()))

            # Write EQs (Earthquakes)
            for eq_name, eq_data in f["EQ"].items():
                network, station, _ = eq_name.split("_")
                eq_attributes = dict(eq_data.attrs)

                starttime = str(UTCDateTime(eq_attributes["starttime"]))
                otime = str(UTCDateTime(eq_attributes["otime"]))

                metadata = {
                    "trace_name": eq_name,
                    "trace_start_time": starttime,
                    "trace_category": "earthquake",
                    "trace_p_arrival_sample": 80,
                    "trace_p_status": "estimated",
                    "station_code": station,
                    "station_network_code": network,
                    "station_latitude_deg": eq_attributes["stla"],
                    "station_longitude_deg": eq_attributes["stlo"],
                    "station_elevation_m": eq_attributes["stel"],
                    "source_magnitude": eq_attributes["mag"],
                    "source_latitude_deg": eq_attributes["evla"],
                    "source_longitude_deg": eq_attributes["evlo"],
                    "source_depth_km": eq_attributes["evdp"] / 1e3,
                    "source_origin_time": otime,
                    "path_ep_distance_km": eq_attributes["dist"] / 1e3,
                    "path_azimuth_deg": eq_attributes["az"],
                    "path_back_azimuth_deg": eq_attributes["baz"],
                    "split": self._get_split_from_time(starttime),
                }

                writer.add_trace(metadata, eq_data[()])

            # Write ANs (Noise)
            for an_name, an_data in f["AN"].items():
                network, station, _ = an_name.split("_")
                an_attributes = dict(an_data.attrs)

                starttime = str(UTCDateTime(an_attributes["starttime"]))

                metadata = {
                    "trace_name": an_name,
                    "trace_start_time": starttime,
                    "trace_category": "noise",
                    "station_code": station,
                    "station_network_code": network,
                    "station_latitude_deg": an_attributes["stla"],
                    "station_longitude_deg": an_attributes["stlo"],
                    "station_elevation_m": an_attributes["stel"],
                    "split": self._get_split_from_time(starttime),
                }

                writer.add_trace(metadata, an_data[()])

        if cleanup:
            # Remove original dataset
            os.remove(path_original)

    @staticmethod
    def _get_split_from_time(starttime):
        train_dev_border = "2017-01-16"
        dev_test_border = "2017-08-16"

        if starttime < train_dev_border:
            split = "train"
        elif starttime < dev_test_border:
            split = "dev"
        else:
            split = "test"

        return split
