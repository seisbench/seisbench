import seisbench
from seisbench.data.base import BenchmarkDataset
from seisbench.util.trace_ops import (
    rotate_stream_to_zne,
    stream_to_array,
    trace_has_spikes,
    waveform_id_to_network_station_location,
)

import random
import string
import requests
import sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import obspy
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth
from obspy.clients.fdsn import Client


class ETHZ(BenchmarkDataset):
    """
    Regional benchmark dataset of publicly available waveform data & corresponding
    metadata in Swiss Seismological Service (SED) archive. Contains data from
    (2013-2020). A pre-compiled version of the benchmark dataset in compatible
    SeisBench format is available for download from remote root. In case of download issues,
    benchmark dataset is downloaded directly from source via FDSN client and converted SeisBench
    format.

    """

    def __init__(self, **kwargs):
        citation = (
            "Each individual network has its own DOI. From publicly available data:\n"
            "CH: https://doi.org/10.12686/sed/networks/ch\n"
            "C4: https://doi.org/10.12686/sed/networks/c4\n"
            "8D: https://doi.org/10.12686/sed/networks/8d\n"
            "S:  https://doi.org/10.12686/sed/networks/s\n"
            "XT: https://doi.org/10.12686/alparray/xt_2014"
        )

        seisbench.logger.warning(
            "Check available storage and memory before downloading and general use "
            "of ETHZ dataset. "
            "Dataset size: waveforms.hdf5 ~22Gb, metadata.csv ~13Mb"
        )

        self._client = None
        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    @classmethod
    def _fdsn_client(cls):
        return Client("ETH")

    @property
    def client(self):
        if self._client is None:
            self._client = self._fdsn_client()
        return self._client

    def _download_dataset(self, writer, time_before=60, time_after=60, **kwargs):
        """
        Download dataset from raw data source via FDSN client.

        :param writer:  WaveformDataWriter instance for writing waveforms and metadata.
        :type writer: seisbench.data.base.WaveformDataWriter
        :param time_before: Extract waveform recordings from event onset - time_before, defaults to 60
        :type time_before: int, optional
        :param time_after: Extract waveform recordings up to event onset + time_after, defaults to 60
        :type time_after: int, optional

        """
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

            station_groups = defaultdict(list)
            for pick in event.picks:
                if pick.phase_hint is None:
                    continue

                station_groups[
                    waveform_id_to_network_station_location(pick.waveform_id.id)
                ].append(pick)

            for picks in station_groups.values():
                try:
                    trace_params = self._get_trace_params(
                        picks[0], inventory_mapper, event_params
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
                        channel=f"{trace_params['trace_channel']}*",
                        starttime=t_start,
                        endtime=t_end,
                    )
                except FDSNNoDataException as e:
                    seisbench.logger.debug(e)
                    self.no_data_catches += 1
                    continue

                rotate_stream_to_zne(waveforms, inv)

                if len(waveforms) == 0:
                    seisbench.logger.debug(
                        f"Found no waveforms for {waveform_id_to_network_station_location(picks[0].waveform_id.id)}"
                        f' in event {event_params["source_id"]}'
                    )
                    continue

                sampling_rate = waveforms[0].stats.sampling_rate
                if any(
                    trace.stats.sampling_rate != sampling_rate for trace in waveforms
                ):
                    seisbench.logger.warning(
                        f"Found inconsistent sampling rates for "
                        f"{waveform_id_to_network_station_location(picks[0].waveform_id.id)} in event {event}."
                        f"Resampling traces to common sampling rate."
                    )
                    waveforms.resample(sampling_rate)

                trace_params[
                    "trace_name"
                ] = f"{event_params['source_id']}_{waveform_id_to_network_station_location(picks[0].waveform_id.id)}"

                stream = waveforms.slice(t_start, t_end)

                actual_t_start, data, completeness = stream_to_array(
                    stream,
                    component_order=writer.data_format["component_order"],
                )

                if int((t_end - t_start) * sampling_rate) + 1 > data.shape[1]:
                    # Traces appear to be complete, but do not cover the intended time range
                    completeness *= data.shape[1] / (
                        int((t_end - t_start) * sampling_rate) + 1
                    )

                trace_params["trace_sampling_rate_hz"] = sampling_rate
                trace_params["trace_completeness"] = completeness
                trace_params["trace_has_spikes"] = trace_has_spikes(data)
                trace_params["trace_start_time"] = str(actual_t_start)

                for pick in picks:
                    sample = (pick.time - actual_t_start) * sampling_rate
                    trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = int(
                        sample
                    )
                    trace_params[
                        f"trace_{pick.phase_hint}_status"
                    ] = pick.evaluation_mode
                    if pick.polarity is None:
                        trace_params[
                            f"trace_{pick.phase_hint}_polarity"
                        ] = "undecidable"
                    else:
                        trace_params[
                            f"trace_{pick.phase_hint}_polarity"
                        ] = pick.polarity

                writer.add_trace({**event_params, **trace_params}, data)

    def _download_ethz_events_xml(
        self,
        starttime=obspy.UTCDateTime(2013, 1, 1),
        endtime=obspy.UTCDateTime(2021, 1, 1),
        minmagnitude=1.5,
    ):
        """
        Download QuakeML data from FDSN for all events satisfying defined parameters.
        QuakeML structure is also stored in cache.

        :param starttime: Define start time of window to select events, defaults to obspy.UTCDateTime(2013, 1, 1)
        :type starttime: obspy.core.UTCDateTime, optional
        :param endtime: Define start time of window to select events, defaults to obspy.UTCDateTime(2021, 1, 1)
        :type endtime: obspy.core.UTCDateTime, optional
        :param minmagnitude: Select all events with event mag > minmagnitude, defaults to 1.5
        :type minmagnitude: float, optional
        :return: Catalog containing all selected events
        :rtype: obspy.core.Catalog

        """
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
            desc="Downloading quakeml event meta from FDSNWS", total=len(ev_ids)
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

        if str(origin.time) < "2019-01-08":
            split = "train"
        elif str(origin.time) < "2019-09-04":
            split = "dev"
        else:
            split = "test"
        event_params["split"] = split

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
            "path_back_azimuth_deg": back_azimuth,
            "station_network_code": net,
            "station_code": sta,
            "trace_channel": pick.waveform_id.channel_code[:2],
            "station_location_code": pick.waveform_id.location_code,
            "station_latitude_deg": lat,
            "station_longitude_deg": lon,
            "station_elevation_m": elev,
        }

        return trace_params


class InventoryMapper:
    """
    Helper class to map station inventories to metadata.

    """

    def __init__(self, inv):
        self.nested_sta_meta = self._create_nested_metadata(inv)

    @staticmethod
    def _create_nested_metadata(inv):
        nested_station_meta = {}

        for network in inv._networks:
            network_code = network._code
            nested_station_meta[network_code] = {}

            for station in network.stations:
                nested_station_meta[network_code][station._code] = {
                    "network": network_code,
                    "station": station._code,
                    "latitude": station._latitude,
                    "longitude": station._longitude,
                    "elevation": station._elevation,
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
