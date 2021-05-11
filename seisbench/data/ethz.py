import seisbench
from seisbench.data.base import BenchmarkDataset
from seisbench.util.trace_ops import (
    _rotate_stream_to_ZNE,
    _stream_to_array,
    _trace_has_spikes,
)

import random
import string
import requests
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import obspy
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.io.mseed import InternalMSEEDError
from obspy.geodetics import gps2dist_azimuth
from obspy.core.util.obspy_types import ObsPyException
from obspy.clients.fdsn import Client


class ETHZ(BenchmarkDataset):
    def __init__(self, **kwargs):
        citation = "ETHZ dataset"
        self.downloaded_raw = False
        super().__init__(citation=citation, repository_lookup=True, **kwargs)

        # If raw data downloaded, merge metadata rows linked to same underlying data
        if self.downloaded_raw:
            # TODO: Previous bugs metadata parsing still need to be fully checked with unit tests - write parsed info to new file to prevent loss of original data
            self._metadata = self._merge_metadata(self._metadata)  
            self._metadata.to_csv(str(self.path / "metadata_out.csv"))

    @classmethod
    def _fdsn_client(cls):
        return Client("ETH")

    @property
    def client(self):
        return self._fdsn_client()

    def _download_dataset(self, writer, time_before=60, time_after=60, **kwargs):
        seisbench.logger.info(
            "No pre-processed version of ETHZ dataset found. "
            "Download and conversion of raw data will now be "
            "performed. This may take a while."
        )

        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        inv = self.client.get_stations(includerestricted=False)
        inventory_mapper = InventoryMapper(inv)

        if (self.path / "ethz_events.xml").exists():
            seisbench.logger.info("Reading quakeml event catalog from cache.")
            catalog = obspy.read_events(
                str(self.path / "ethz_events.xml"), format="QUAKEML"
            )
        else:
            catalog = self._download_ethz_events_xml()

        self.not_in_inv_catches = 0
        self.no_data_catches = 0

        for event in catalog:
            origin, mag, fm, event_params = self._get_event_params(event)
            seisbench.logger.info(f"Downloading {event.resource_id}")

            picks = sorted(event.picks, key=lambda x: x.time)
            for pick in picks:
                try:
                    trace_params = self._get_trace_params(
                        pick, inventory_mapper, event_params
                    )
                except KeyError as e:
                    self.not_in_inv_catches += 1
                    seisbench.logger.debug(e)
                    continue

                t_start = min(pick.time for pick in picks) - time_before
                t_end = max(pick.time for pick in picks) + time_after

                try:
                    waveforms = self.client.get_waveforms(
                        network=trace_params["station_network_code"],
                        station=trace_params["station_code"],
                        location="*",
                        channel=f"{trace_params['station_channel_code']}*",
                        starttime=t_start,
                        endtime=t_end,
                    )
                except FDSNNoDataException as e:
                    seisbench.logger.debug(e)
                    self.no_data_catches += 1
                    continue

                if len(waveforms) == 0:
                    seisbench.logger.debug(
                        f'Found no waveforms for {pick.waveform_id.id[:-1]} in event {event_params["source_id"]}'
                    )
                    continue

                sampling_rate = waveforms[0].stats.sampling_rate
                if any(
                    trace.stats.sampling_rate != sampling_rate for trace in waveforms
                ):
                    seisbench.logger.warning(
                        f"Found inconsistent sampling rates for {pick.waveform_id.id[:-1]} in event {event}"
                    )

                trace_params[
                    "trace_name"
                ] = f"{event_params['source_id']}_{pick.waveform_id.id[:-1]}"

                stream = waveforms.slice(t_start, t_end)

                actual_t_start, data, completeness = _stream_to_array(
                    stream,
                    component_order=writer.data_format["component_order"],
                )

                if int((t_end - t_start) * sampling_rate) + 1 > data.shape[1]:
                    # Traces appear to be complete, but do not cover the intended time range
                    completeness *= data.shape[1] / (
                        int((t_end - t_start) * sampling_rate) + 1
                    )

                trace_params["trace_sampling_rate"] = sampling_rate
                trace_params["trace_completeness"] = completeness
                trace_params["trace_has_spikes"] = _trace_has_spikes(data)
                trace_params["trace_start_time"] = str(actual_t_start)

                sample = (pick.time - actual_t_start) * sampling_rate
                trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = int(sample)
                trace_params[f"trace_{pick.phase_hint}_status"] = pick.evaluation_mode

                try:
                    writer.add_trace({**event_params, **trace_params}, data)
                except ValueError as e:
                    # For multiple metadata rows picking same waveform, store seperatley for now.
                    seisbench.logger.debug(
                        f"Received: {e}. Extra phase type is {pick.phase_hint}. Stored in seperate metadata row."
                    )
                    continue

        self.downloaded_raw = True

    def _download_ethz_events_xml(
        self,
        starttime=obspy.UTCDateTime(2013, 1, 1),
        endtime=obspy.UTCDateTime(2021, 1, 1),
        minmagnitude=1.5,
    ):
        query = (
            f"http://arclink.ethz.ch/fdsnws/event/1/query?"
            f"starttime={starttime.isoformat()}&endtime={endtime.isoformat()}"
            f"&minmagnitude={minmagnitude}&format=text"
        )
        resp = requests.get(query)
        ev_ids = [
            line.decode(sys.stdout.encoding).split("|")[0]
            for line in resp._content.splitlines()[1:]
        ]

        catalog = obspy.Catalog(events=[])

        with tqdm(
            desc="Downlading quakeml event meta from FDSNWS", total=len(ev_ids)
        ) as pbar:
            for ev_id in ev_ids:
                catalog += self.client.get_events(eventid=ev_id, includearrivals=True)
                pbar.update()

        catalog.write(str(self.path / "ethz_events.xml"), format="QUAKEML")

        return catalog

    @staticmethod
    def _get_event_params(event):
        origin = event.preferred_origin()
        mag = event.preferred_magnitude()
        fm = event.preferred_focal_mechanism()

        if str(event.resource_id).split("/")[-1] == "1":
            # Generate custom source-id
            chars = string.ascii_lowercase + string.digits
            source_id = "sb_id_" + "".join(random.choice(chars) for _ in range(6))
        else:
            source_id = str(event.resource_id).split("/")[-1]

        event_params = {
            "source_id": source_id,
            "source_origin_time": str(origin.time),
            "source_origin_uncertainty_sec": origin.time_errors["uncertainty"],
            "source_latitude_deg": origin.latitude,
            "source_latitude_uncertainty_km": origin.latitude_errors["uncertainty"],
            "source_longitude_deg": origin.longitude,
            "source_longitude_uncertainty_km": origin.longitude_errors["uncertainty"],
            "source_depth_km": origin.depth / 1e3,
            "source_depth_uncertainty_km": origin.depth_errors["uncertainty"] / 1e3,
        }

        if mag is not None:
            event_params["source_magnitude"] = mag.mag
            event_params["source_magnitude_uncertainty"] = mag.mag_errors["uncertainty"]
            event_params["source_magnitude_type"] = mag.magnitude_type
            event_params["source_magnitude_author"] = mag.creation_info.agency_id

        if fm is not None:
            try:
                t_axis, p_axis, n_axis = (
                    fm.principal_axes.t_axis,
                    fm.principal_axes.p_axis,
                    fm.principal_axes.n_axis,
                )
                event_params["source_focal_mechanism_t_azimuth"] = t_axis.azimuth
                event_params["source_focal_mechanism_t_plunge"] = t_axis.plunge
                event_params["source_focal_mechanism_t_length"] = t_axis.length

                event_params["source_focal_mechanism_p_azimuth"] = p_axis.azimuth
                event_params["source_focal_mechanism_p_plunge"] = p_axis.plunge
                event_params["source_focal_mechanism_p_length"] = p_axis.length

                event_params["source_focal_mechanism_n_azimuth"] = n_axis.azimuth
                event_params["source_focal_mechanism_n_plunge"] = n_axis.plunge
                event_params["source_focal_mechanism_n_length"] = n_axis.length
            except AttributeError:
                # There seem to be a few broken xml files. In this case, just ignore the focal mechanism.
                pass

        return origin, mag, fm, event_params

    @staticmethod
    def _get_trace_params(pick, inventory, event_params):
        net = pick.waveform_id.network_code
        sta = pick.waveform_id.station_code

        lat, lon, elev = inventory.get_station_location(network=net, station=sta)

        if not np.isnan(lat * lon):
            back_azimuth = gps2dist_azimuth(
                event_params["source_latitude_deg"],
                event_params["source_longitude_deg"],
                lat,
                lon,
            )[2]
        else:
            back_azimuth = np.nan

        trace_params = {
            "trace_back_azimuth_deg": back_azimuth,
            "station_network_code": net,
            "station_code": sta,
            "station_channel_code": pick.waveform_id.channel_code[:-1],
            "station_location_code": pick.waveform_id.location_code,
            "station_latitude_deg": lat,
            "station_longitude_deg": lon,
            "station_elevation_m": elev,
        }

        return trace_params

    @staticmethod
    def _merge_metadata(metadata):
        """
        Merges all rows in metadata with same underlying data into a single row.
        """
        unique_metadata, parsed_metadata = [], []

        def _dataframe_to_dicts(dataframe):
            parsed_rows = []
            for row in dataframe.iterrows():
                parsed_rows.append(row[1].dropna().to_dict())

            return parsed_rows

        def _merge_dicts_w_check(dicts, error_callback_print_key="trace_name"):
            _merged_dict = {}
            for dictionary in dicts:
                for key, value in dictionary.items():
                    if key not in _merged_dict:
                        _merged_dict[key] = value
                    else:
                        if not _merged_dict[key] == value:
                            seisbench.logger.warning(
                                f"Differing values recieved for key {key}. "
                                f"Recieved: {value}, and {_merged_dict[key]}, "
                                f"{error_callback_print_key}: {_merged_dict[error_callback_print_key]}"
                            )

            return _merged_dict

        # Get all rows with same underlying trace data
        for row in metadata.itertuples():
            sel_rows = metadata.loc[metadata["trace_name"] == row.trace_name]

            if len(sel_rows) > 1:
                # Convert metadata to dict and merge into single row
                parsed_rows = _dataframe_to_dicts(sel_rows)
                merged_metadata_dict = _merge_dicts_w_check(parsed_rows, "trace_name")
                parsed_metadata.append(merged_metadata_dict)
            else:
                unique_metadata.append(row._asdict())
                unique_metadata.pop('Index', None)

        return pd.DataFrame(unique_metadata + parsed_metadata).drop_duplicates()


class InventoryMapper:
    def __init__(self, inv):
        self.nested_sta_meta = self._create_nested_metadata(inv)

    @staticmethod
    def _create_nested_metadata(inv):
        nested_station_meta = {}

        for network in inv._networks:
            network_code = network._code
            nested_station_meta[network_code] = {}

            for station in network.stations:
                station_code = station._code
                latitude = station._latitude
                longitude = station._longitude
                elevation = station._elevation

                nested_station_meta[network_code][station_code] = {
                    "network": network_code,
                    "station": station_code,
                    "latitude": latitude,
                    "longitude": longitude,
                    "elevation": elevation,
                }

        return nested_station_meta

    def get_station_location(self, network, station):
        try:
            self.nested_sta_meta[network]
        except KeyError as e:
            raise KeyError(f"network code '{e.args[0]}' not in inventory")
        try:
            sta_meta = self.nested_sta_meta[network][station]
            return (
                sta_meta["latitude"],
                sta_meta["longitude"],
                sta_meta["elevation"],
            )
        except KeyError as e:
            raise KeyError(f"station code '{e.args[0]}' not in inventory")
