from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import seisbench
from seisbench.data.base import WaveformDataset

try:
    from pyrocko.gui.marker import PhaseMarker, save_markers
    from pyrocko.io import save
    from pyrocko.model import Event, Station, dump_events, dump_stations_yaml
    from pyrocko.trace import NoData, Trace, degapper, snuffle
except ImportError as e:
    raise ImportError("pyrocko is required for dataset inspection") from e

if TYPE_CHECKING:
    from seisbench.data.base import TraceParameters


PathStr = str | Path

_PYROCKO_POLARITY_MAP = {
    "undecideable": None,
    "up": 1,
    "down": -1,
}

logger = seisbench.logger


class DatasetInspection:
    _dataset: WaveformDataset
    _metadata: pd.DataFrame

    def __init__(self, dataset: WaveformDataset) -> None:
        self._dataset = dataset
        self._metadata = dataset.metadata

    def get_pyrocko_traces(self, sample_idx: int) -> list[Trace]:
        """
        Returns the waveforms of a sample as a list of pyrocko Trace objects.

        Requires pyrocko to be installed.

        :param idx: Idx of sample to return traces for
        :return: List of pyrocko Trace objects
        """

        ds = self._dataset

        data, metadata = ds.get_sample(sample_idx)
        traces = []

        location_code = str(metadata.get("station_location_code"))
        location_code = "" if location_code == "nan" else location_code

        for ichannel, trace_data in enumerate(data):
            channel = metadata["trace_component_order"][ichannel]
            trace = Trace(
                ydata=trace_data,
                tmin=datetime.fromisoformat(metadata["trace_start_time"]).timestamp(),
                deltat=1.0 / metadata["trace_sampling_rate_hz"],
                network=metadata["station_network_code"],
                station=metadata["station_code"],
                location=location_code,
                channel=f"{metadata['trace_channel']}{channel}",
            )
            traces.append(trace)
        return traces

    def get_pyrocko_event(self, sample_idx: int) -> Event:
        """
        Returns the event of a sample as a pyrocko Event object.

        Requires pyrocko to be installed.

        :param idx: Idx of sample to return event for, or event source_id
        :return: pyrocko Event object
        """

        metadata = self._metadata

        metadata = metadata.iloc[sample_idx].to_dict()
        return Event(
            name=f"{metadata['index']} ({metadata['split']})",
            time=metadata["source_origin_time"].timestamp(),
            lat=metadata["source_latitude_deg"],
            lon=metadata["source_longitude_deg"],
            depth=metadata["source_depth_km"] * 1e3,
            magnitude=metadata.get("source_magnitude", 0.0),
            magnitude_type=metadata.get("source_magnitude_type"),
            extras={
                "id": metadata["source_id"],
                "split": metadata["split"],
            },
        )

    def get_pyrocko_picks(self, sample_idx: int) -> list[PhaseMarker]:
        """
        Returns the picks of a sample as a list of pyrocko Pick objects.

        Requires pyrocko to be installed.

        :param idx: Idx of sample to return picks for
        :return: List of pyrocko Pick objects
        """
        metadata = self._metadata

        metadata = metadata.iloc[sample_idx].to_dict()
        sampling_rate = metadata["trace_sampling_rate_hz"]
        phases = [
            key.strip("trace_").strip("_arrival_sample")
            for key in metadata.keys()
            if key.endswith("_arrival_sample")
        ]
        location_code = metadata.get("station_location_code")
        location_code = "" if np.isnan(location_code) else location_code
        nsl = (
            metadata["station_network_code"],
            metadata["station_code"],
            location_code,
        )

        trace_start = datetime.fromisoformat(metadata["trace_start_time"])

        picks = []
        for phase in phases:
            if np.isnan(metadata[f"trace_{phase}_arrival_sample"]):
                continue

            pick_delay = metadata[f"trace_{phase}_arrival_sample"] / sampling_rate
            pick_time = trace_start.timestamp() + pick_delay
            automatic = metadata.get(f"trace_{phase}_status", "manual") == "automatic"

            pick = PhaseMarker(
                tmin=pick_time,
                tmax=pick_time,
                nslc_ids=[nsl + ("*",)],
                automatic=automatic,
                phasename=phase,
                polarity=_PYROCKO_POLARITY_MAP.get(
                    metadata.get(f"trace_{phase}_polarity", "Undecideable")
                ),
            )
            picks.append(pick)
        return picks

    def _get_station_tuple(self, sample_idx: int) -> _StationTuple:
        metadata = self._metadata.iloc[sample_idx].to_dict()
        return _StationTuple.from_metadata(metadata)

    def get_pyrocko_station(self, sample_idx: int) -> Station:
        return self._get_station_tuple(sample_idx).as_pyrocko_station()

    def pyrocko_snuffle_sample(self, sample_idx: int) -> None:
        """
        Returns a pyrocko Snuffle object containing the traces, event and picks of a sample.

        Requires pyrocko to be installed.

        :param idx: Idx of sample to return Snuffle for
        :return: pyrocko Snuffle object
        """

        traces = self.get_pyrocko_traces(sample_idx)
        event = self.get_pyrocko_event(sample_idx)
        picks = self.get_pyrocko_picks(sample_idx)
        station = self._get_station_tuple(sample_idx)

        snuffle(
            traces=traces,
            events=[event],
            markers=picks,
            stations=[station.as_pyrocko_station()],
        )

    def _get_pyrocko_data(
        self, event: int | str
    ) -> tuple[list[Trace], Event, list[PhaseMarker], set[_StationTuple]]:
        """
        Snuffle all traces, picks and the event associated with a given event.

        Requires pyrocko to be installed.

        :param event: Event identifier
        :return: pyrocko Snuffle object
        """

        ds = self._dataset

        if isinstance(event, int):
            event_source_id = ds.get_event_source_id(event)
        else:
            event_source_id = event

        idx = ds.get_event_sample_indices(event_source_id)
        if not idx:
            raise ValueError(f"No samples found for event {event_source_id}")
        event = self.get_pyrocko_event(idx[0])
        all_traces = []
        all_picks = []
        stations = set()

        for i in idx:
            all_traces += self.get_pyrocko_traces(i)
            all_picks += self.get_pyrocko_picks(i)
            stations.add(self._get_station_tuple(i))

        return all_traces, event, all_picks, stations

    def pyrocko_snuffle_event(self, event: int | str) -> None:
        """
        Snuffle all traces, picks and the event associated with a given event.

        Requires pyrocko to be installed.

        :param event: Event identifier
        :return: pyrocko Snuffle object
        """

        all_traces, event, all_picks, stations = self._get_pyrocko_data(event)

        snuffle(
            traces=all_traces,
            events=[event],
            markers=all_picks,
            stations=[sta.as_pyrocko_station() for sta in stations],
        )

    def export(self, directory: PathStr) -> None:
        """
        Exports all traces in the dataset to MiniSEED files, grouped by day.

        Requires pyrocko to be installed.

        :param directory: Directory to write MiniSEED files to.
        :return: None
        """
        directory = Path(directory)
        ds = self._dataset
        metadata = self._metadata
        n_events = self._dataset.n_events()

        (directory / "mseed").mkdir(parents=True, exist_ok=True)
        (directory / "picks").mkdir(parents=True, exist_ok=True)
        (directory / "csv").mkdir(parents=True, exist_ok=True)

        day_groups = (
            metadata.groupby("source_id", as_index=False)
            .first()
            .groupby(pd.Grouper(key="source_origin_time", freq="D"))
        )
        events = []
        stations = set()

        all_picks = []
        with tqdm(desc="Dumping MiniSEEDs", total=n_events) as pbar:
            for day, metadata_grouped in day_groups:
                n_events_day = metadata_grouped.shape[0]
                if n_events_day == 0:
                    continue

                pick_markers = []
                traces = []
                for event_idx in range(n_events_day):
                    event_traces, event, event_picks, stas = self._get_pyrocko_data(
                        metadata_grouped.iloc[event_idx]["source_id"]
                    )
                    traces += event_traces
                    pick_markers += event_picks
                    events.append(event)
                    stations.update(stas)

                filename = directory / "mseed" / f"{day.date()}.mseed"
                filename_picks = directory / "picks" / f"{day.date()}.picks"

                try:
                    traces = degapper(sorted(traces, key=lambda tr: tr.full_id))
                except NoData:
                    logger.error("Skipping day %s due to degapper NoData error", day)
                    continue
                save(traces, str(filename), format="mseed", overwrite=True)

                save_markers(pick_markers, str(filename_picks))
                all_picks += pick_markers

                seisbench.logger.info("Wrote %d events to %s", n_events_day, filename)
                pbar.update(n_events_day)

        seisbench.logger.info("Dumping events, picks and stations")
        pyrocko_stations = [sta.as_pyrocko_station() for sta in stations]

        dump_events(events, str(directory / "events.yaml"), format="yaml")
        save_markers(all_picks, str(directory / "all_picks.picks"))
        dump_stations_yaml(pyrocko_stations, str(directory / "stations.yaml"))

        dump_events_csv(events, directory / "csv" / "events.csv")
        dump_stations_csv(list(stations), directory / "csv" / "stations.csv")

        readme = directory / "README.md"
        readme.write_text(
            f"""
# SeisBench MiniSEED dump of dataset {ds.name}

This directory contains a MiniSEED dump of the SeisBench dataset '{ds.name}'.
The dataset contains {len(ds)} earthquake records associated with {n_events} events.

The data is organized as follows:
- `mseed/`: Contains MiniSEED files, one file per day.
- `picks/`: Contains pick files in pyrocko .picks format, one file per day.
- `events.yaml`: Contains all events in pyrocko .yaml format.
- `stations.yaml`: Contains all stations in pyrocko .yaml format.
- `all_picks.picks`: Contains all picks in pyrocko .picks format

## Open Pyrocko Snuffler

You can open the data in pyrocko's snuffler using the following command:
```bash
squirrel snuffler -a mseed/
```

Use `N` to navigate between events.

## Dataset Citation
{getattr(ds, "_citation") if hasattr(ds, "_citation") else "No citation provided."}

## Dataset License
{getattr(ds, "_license") if hasattr(ds, "_license") else "No license provided."}

Created with SeisBench {seisbench.__version__}
"""
        )


class _StationTuple(NamedTuple):
    network: str
    station: str
    location: str
    lat: float
    lon: float
    elevation: float

    @classmethod
    def from_metadata(cls, metadata: TraceParameters) -> _StationTuple:
        location_code = str(metadata.get("station_location_code"))
        location_code = "" if location_code == "nan" else location_code
        return cls(
            metadata["station_network_code"],
            metadata["station_code"],
            location_code,
            metadata["station_latitude_deg"],
            metadata["station_longitude_deg"],
            metadata["station_elevation_m"],
        )

    def as_pyrocko_station(self) -> Station:
        return Station(
            network=self.network,
            location=self.location,
            station=self.station,
            lat=self.lat,
            lon=self.lon,
            elevation=self.elevation,
        )

    def as_csv(self) -> str:
        return (
            f"{self.network},{self.station},{self.location},"
            f"{self.lat},{self.lon},{self.elevation},"
            f"POINT Z({self.lon} {self.lat} {self.elevation})"
        )


def dump_stations_csv(stations: list[_StationTuple], filename: PathStr) -> None:
    """
    Dumps a list of stations to a CSV file.

    :param stations: List of stations to dump
    :param filename: Filename to write CSV to
    :return: None
    """
    header = "network,station,location,latitude,longitude,elevation,WKT_geom"
    lines = [sta.as_csv() for sta in stations]
    filename = Path(filename)
    filename.write_text("\n".join([header] + lines) + "\n")
    logger.info("Wrote %d stations to %s", len(stations), filename)


def dump_events_csv(events: list[Event], filename: PathStr) -> None:
    """
    Dumps a list of events to a CSV file.

    :param events: List of events to dump
    :param filename: Filename to write CSV to
    :return: None
    """
    header = "time,latitude,longitude,depth_m,magnitude,magnitude_type,name,id,WKT_geom"
    lines = [
        (
            f"{datetime.fromtimestamp(ev.time, tz=timezone.utc)},{ev.lat},{ev.lon},"
            f"{ev.depth},{ev.magnitude},"
            f"{ev.magnitude_type},{ev.name},{ev.extras.get('id', '')}"
            f",POINT Z({ev.lon} {ev.lat} {-ev.depth})"
        )
        for ev in events
    ]
    filename = Path(filename)
    filename.write_text("\n".join([header] + lines) + "\n")
    logger.info("Wrote %d events to %s", len(events), filename)


__all__ = ["DatasetInspection"]
