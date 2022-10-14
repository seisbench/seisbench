import seisbench
from .base import BenchmarkDataset
from seisbench.util.trace_ops import (
    rotate_stream_to_zne,
    stream_to_array,
    trace_has_spikes,
)

from pathlib import Path
import obspy
from obspy.io.mseed import InternalMSEEDError
from obspy.geodetics import gps2dist_azimuth
import copy
from collections import defaultdict
import numpy as np
import pandas as pd


class GEOFON(BenchmarkDataset):
    """
    GEOFON dataset consisting of both regional and teleseismic picks. Mostly contains P arrivals,
    but a few S arrivals are annotated as well. Contains data from 2010-2013. The dataset will be
    downloaded from the SeisBench repository on first usage.
    """

    def __init__(self, **kwargs):
        # TODO: Add citation
        citation = "GEOFON dataset"
        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(
        self, writer, basepath=None, time_before=60, time_after=60, **kwargs
    ):
        if basepath is None:
            raise ValueError(
                "'basepath' needs to be set in the download_kwargs to start dataset conversion for source. "
                "If you are seeing this error, the SeisBench remote root might be unavailable. "
                "If it is available and you are still seeing this error, "
                "please get in touch with the developers."
            )

        component_order = "ZNE"

        writer.data_format = {
            "dimension_order": "CW",
            "component_order": component_order,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        basepath = Path(basepath)

        location_helper = LocationHelper(basepath)
        inventory = obspy.read_inventory(
            str(basepath / "inventory_without_response.xml")
        )

        for event_path in sorted(basepath.iterdir()):
            if not event_path.is_dir():
                continue
            quakeml = [
                x for x in event_path.iterdir() if x.name.endswith("preferred-only.xml")
            ]
            if len(quakeml) != 1:
                continue

            quakeml = quakeml[0]

            catalog = obspy.read_events(str(quakeml))
            if len(catalog) != 1:
                seisbench.logger.warning(
                    f"Found multiple events in catalog for {event_path.name}. Skipping."
                )
                continue

            event = catalog[0]

            event_params = self._get_event_params(event)

            station_groups = defaultdict(list)
            for pick in event.picks:
                if pick.phase_hint is None:
                    continue

                if pick.waveform_id.network_code == "IA":
                    # Skip restricted IA data
                    continue

                if not isinstance(pick.waveform_id.network_code, str):
                    # Skip traces with invalid network code
                    continue

                station_groups[pick.waveform_id.id[:-1]].append(pick)

            for picks in station_groups.values():
                self._write_picks(
                    picks,
                    event_params,
                    event_path,
                    writer,
                    location_helper,
                    inventory,
                    component_order=component_order,
                )

    @staticmethod
    def _get_event_params(event):
        origin = event.preferred_origin()
        mag = event.preferred_magnitude()
        fm = event.preferred_focal_mechanism()
        event_params = {
            "source_id": str(event.resource_id)[-11:],
            "source_origin_time": str(origin.time),
            "source_origin_uncertainty_sec": origin.time_errors["uncertainty"],
            "source_latitude_deg": origin.latitude,
            "source_latitude_uncertainty_deg": origin.latitude_errors["uncertainty"],
            "source_longitude_deg": origin.longitude,
            "source_longitude_uncertainty_deg": origin.longitude_errors["uncertainty"],
            "source_depth_km": origin.depth / 1e3,
            "source_depth_uncertainty_km": origin.depth_errors["uncertainty"] / 1e3,
        }

        if str(origin.time) < "2012-11-01":
            split = "train"
        elif str(origin.time) < "2013-03-15":
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
        return event_params

    @staticmethod
    def _get_trace_params(picks, location_helper, event_params):
        pick = picks[0]
        lat, lon, elevation, sensitivity = location_helper.find(pick.waveform_id.id)
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
            "station_network_code": pick.waveform_id.network_code,
            "station_code": pick.waveform_id.station_code,
            "trace_channel": pick.waveform_id.channel_code[:-1],
            "station_location_code": pick.waveform_id.location_code,
            "station_latitude_deg": lat,
            "station_longitude_deg": lon,
            "station_elevation_m": elevation,
            "station_sensitivity_counts_spm": sensitivity,
        }
        return trace_params

    def _write_picks(
        self,
        picks,
        event_params,
        event_path,
        writer,
        location_helper,
        inventory,
        time_before=60,
        time_after=60,
        component_order="ZNE",
        suffix="",
    ):
        picks = sorted(picks, key=lambda x: x.time)
        # For stations where spacing between two picks is larger than (time_before + time_after), write multiple traces
        for i, _ in enumerate(picks[:-1]):
            if picks[i + 1].time - picks[i].time > time_before + time_after:
                self._write_picks(
                    picks[: i + 1],
                    event_params,
                    event_path,
                    writer,
                    location_helper,
                    inventory,
                    time_before,
                    time_after,
                    component_order,
                    suffix + "0",
                )
                self._write_picks(
                    picks[i + 1 :],
                    event_params,
                    event_path,
                    writer,
                    location_helper,
                    inventory,
                    time_before,
                    time_after,
                    component_order,
                    suffix + "1",
                )

        trace_params = copy.deepcopy(event_params)
        trace_params.update(
            self._get_trace_params(picks, location_helper, event_params)
        )
        trace_params[
            "trace_name"
        ] = f"{event_params['source_id']}_{picks[0].waveform_id.id[:-1]}{suffix}"

        stream = obspy.Stream()
        loaded = set()
        for pick in picks:
            for c in "ZNE12":
                mseed_path = event_path / f"{pick.waveform_id.id[:-1]}{c}.mseed"
                if str(mseed_path) not in loaded:
                    loaded.add(str(mseed_path))
                    try:
                        stream += obspy.read(str(mseed_path))
                    except FileNotFoundError:
                        pass
                    except InternalMSEEDError:
                        pass
                    except TypeError:
                        # Thrown by obspy for specific invalid MSEED files
                        pass
        stream.merge(-1)

        if len(stream) == 0:
            seisbench.logger.warning(
                f'Found no waveforms for {picks[0].waveform_id.id[:-1]} in event {event_params["source_id"]}'
            )
            return

        sampling_rate = stream[0].stats.sampling_rate
        if any(trace.stats.sampling_rate != sampling_rate for trace in stream):
            seisbench.logger.warning(
                f"Found inconsistent sampling rates for {picks[0].waveform_id.id[:-1]} "
                f'in event {event_params["source_id"]}'
            )
            return

        rotate_stream_to_zne(stream, inventory)

        t_start = min(pick.time for pick in picks) - time_before
        t_end = max(pick.time for pick in picks) + time_after

        stream = stream.slice(t_start, t_end)

        if len(stream) == 0:
            seisbench.logger.warning(
                f'Found no waveforms for {picks[0].waveform_id.id[:-1]} in event {event_params["source_id"]}'
            )
            return

        actual_t_start, data, completeness = stream_to_array(stream, component_order)

        if int((t_end - t_start) * sampling_rate) + 1 > data.shape[1]:
            # Traces appear to be complete, but do not cover the intended time range
            completeness *= data.shape[1] / (int((t_end - t_start) * sampling_rate) + 1)

        trace_params["trace_sampling_rate_hz"] = sampling_rate
        trace_params["trace_completeness"] = completeness
        trace_params["trace_has_spikes"] = trace_has_spikes(data)
        trace_params["trace_start_time"] = str(actual_t_start)
        for pick in picks:
            sample = (pick.time - actual_t_start) * sampling_rate
            trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = sample
            trace_params[f"trace_{pick.phase_hint}_status"] = pick.evaluation_mode

        writer.add_trace(trace_params, data)


class LocationHelper:
    def __init__(self, path):
        self.path = path
        self.full_dict = {}
        self.short_dict = {}
        self.fill_dicts()

    def fill_dicts(self):
        station_list = pd.read_csv(self.path / "station_list.csv")
        for _, row in station_list.iterrows():
            trace_id = row["id"]
            network, station, location, channel = trace_id.split(".")
            if row["sensitivity"] is None or row["sensitivity"] == "None":
                sensitivity = np.nan
            else:
                sensitivity = float(row["sensitivity"])
            self.full_dict[trace_id] = (
                row["lat"],
                row["lon"],
                row["elevation"],
                sensitivity,
            )
            self.short_dict[f"{network}.{station}"] = (
                row["lat"],
                row["lon"],
                row["elevation"],
                np.nan,
            )

        station_list_additional = pd.read_csv(self.path / "station_list_additional.csv")
        for _, row in station_list_additional.iterrows():
            netsta = row["id"]
            self.short_dict[netsta] = (row["lat"], row["lon"], row["elevation"], np.nan)

    def find(self, trace_id):
        if trace_id in self.full_dict:
            return self.full_dict[trace_id]
        else:
            network, station, location, channel = trace_id.split(".")
            netsta = f"{network}.{station}"
            if netsta in self.short_dict:
                return self.short_dict[netsta]
            else:
                return np.nan, np.nan, np.nan, np.nan
