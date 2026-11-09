from __future__ import annotations

import asyncio
import bisect
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Literal

import numpy as np
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn.header import URL_MAPPINGS, FDSNNoDataException
from obspy.core.event import Catalog, Event, Pick, WaveformStreamID

from seisbench import logger
from seisbench.data.base import (
    EventParameters,
    TraceParameters,
    WaveformBenchmarkDataset,
)
from seisbench.data.utils import (
    FDSNSource,
    MultiClient,
    backup_dataset,
    download_parallel,
    get_cached_inventory,
    get_split,
    get_trace_params,
    get_waveform_array,
    station_key,
)

# One QuakeML file per region, <region>.xml. Japan (Hi-net) is not available via FDSN.
REGIONS = (
    "eifel_qseek",
    "hawaii",
    "volpick_alaska",
    "volpick_cascade",
    "volpick_norcal",
)
# NCEDC holds these networks. EarthScope has copies of some of them, but not all data,
# e.g., no NN data from the 1990s.
NCEDC_NETWORKS = ("BK", "NC", "BG", "NN", "WR")

COMPONENT_ORDER = "ZNE"
SAMPLING_RATE = 100.0

# Noise windows end this long before a catalog event and start this long after one
NOISE_GAP_BEFORE_EVENT = 120.0
NOISE_GAP_AFTER_EVENT = 180.0

RNG_SEED = 31337

# Counts per region in download_stats.json
STATS_KEYS = (
    "events",
    "events_skipped",
    "events_downloaded",
    "events_without_traces",
    "traces",
    "traces_downloaded",
    "traces_failed",
    "traces_no_source",
    "noise",
    "noise_downloaded",
    "noise_failed",
    "noise_near_event",
)


@dataclass
class _Job:
    kind: Literal["event", "noise"]
    region: str
    trace_name: str
    waveform_id: WaveformStreamID
    starttime: UTCDateTime
    endtime: UTCDateTime
    params: dict = field(default_factory=dict)
    picks: list[Pick] = field(default_factory=list)


class DeepLowFrequency(WaveformBenchmarkDataset):
    """
    Deep low-frequency (DLF) earthquakes, also called deep long-period (DLP) events,
    below volcanoes in Alaska, Hawaii, Northern California, the Cascades and the
    Eifel, plus noise windows recorded at the same stations.

    The dataset is compiled from FDSN web services (EarthScope, NCEDC, BGR, GFZ). The
    catalogs are read from one QuakeML file per region in ``catalog_dir``. Each event
    carries its split and event class as comments (``split: train``,
    ``event_class: DLP``).

    Many traces have an S pick only. A missing pick means the phase was not picked,
    not that there is no arrival, so these traces are kept with an empty P arrival.

    For a fraction of the event traces, a noise window is downloaded from the same
    station before the event, with ``trace_category = "noise"``. Noise windows keep
    120 s before and 180 s after every catalog event of the region. They take the
    split of the event.

    The download writes ``download_stats.json`` (counts per region) and
    ``download_failures.csv`` (every trace that could not be downloaded) to the
    dataset folder.

    .. note ::

        The dataset is compiled from FDSN web services on first use, which requires
        ``aiohttp``. Install it with ``pip install seisbench[download]``.

    :param catalog_dir: Folder with the QuakeML catalogs, ``<region>.xml``. Defaults
        to ``quakeml`` in the dataset folder.
    :param split: Fractions of train, dev and test events. If None, the split of
        each event is taken from the catalog.
    :param noise_fraction: Fraction of the event traces that get a noise window
        from the same station. 0 disables noise windows.
    :param noise_length: Length of the noise windows in seconds
    :param eida_token: EIDA token or path to a token file, for restricted EIDA data
    :param connections: Number of parallel connections per FDSN data center
    """

    _client: MultiClient

    def __init__(
        self,
        catalog_dir: str | Path | None = None,
        split: tuple[float, float, float] | None = None,
        noise_fraction: float = 0.3,
        noise_length: float = 60.0,
        eida_token: str | None = None,
        connections: int = 4,
        compile_from_source: bool = True,
        **kwargs,
    ):
        if split is not None and (len(split) != 3 or not np.isclose(sum(split), 1.0)):
            raise ValueError(
                "split must be three fractions (train, dev, test) summing to 1."
            )
        if not 0.0 <= noise_fraction <= 1.0:
            raise ValueError("noise_fraction must be between 0 and 1.")
        if noise_length <= 0.0:
            raise ValueError("noise_length must be positive.")

        citation = """
Catalogs:
* Zhong, Y. & Tan, Y. J. (2024). Deep-learning-based phase picking for volcano-tectonic
  and long-period earthquakes. Geophysical Research Letters, 51, e2024GL108438,
  https://doi.org/10.1029/2024GL108438 (VCSEIS: Alaska, Hawaii, Northern California,
  Cascades)
* Eifel: deep low-frequency events below Laacher See detected with qseek. The picks are
  pseudo-labels of a fine-tuned PhaseNet, checked against the located arrivals.

Waveforms are provided by the EarthScope Data Services, the Northern California
Earthquake Data Center (NCEDC), BGR and GFZ GEOFON. Please cite the networks used:
Alaska Volcano Observatory (AV), Hawaiian Volcano Observatory Network (HV), Northern
California Seismic Network (NC), Berkeley Digital Seismic Network (BK), Pacific Northwest
Seismic Network (UW), Cascades Volcano Observatory (CC) and the Eifel networks (LE, GR,
BQ, HS, NH, 1P, 2D, LU), among others.
"""
        self._catalog_dir = None if catalog_dir is None else Path(catalog_dir)
        self._split = split
        self._noise_fraction = noise_fraction
        self._noise_length = noise_length
        self._eida_token = eida_token
        self._connections = connections

        super().__init__(
            citation=citation,
            license="See the data licenses of the individual networks",
            name="DeepLowFrequency",
            repository_lookup=False,
            # There is no precompiled version, the dataset is always built from FDSN
            compile_from_source=compile_from_source,
            **kwargs,
        )

    def _init_client(self):
        self._client = MultiClient(
            [
                FDSNSource(
                    URL_MAPPINGS["BGR"],
                    connections=self._connections,
                    eida_token=self._eida_token,
                ),
                FDSNSource(
                    URL_MAPPINGS["GEOFON"],
                    connections=self._connections,
                    eida_token=self._eida_token,
                ),
                FDSNSource(
                    URL_MAPPINGS["NCEDC"],
                    connections=self._connections,
                    networks=NCEDC_NETWORKS,
                ),
                FDSNSource(URL_MAPPINGS["IRIS"], connections=self._connections),
            ]
        )

    def get_catalogs(self) -> dict[str, Catalog]:
        """
        Reads the QuakeML catalog of each region.
        """
        catalog_dir = self._catalog_dir or self.path / "quakeml"
        catalogs = {}
        for region in REGIONS:
            file = catalog_dir / f"{region}.xml"
            if not file.exists():
                logger.warning("No catalog for region %s at %s.", region, file)
                continue
            logger.info("Reading catalog %s", file)
            catalogs[region] = read_events(str(file), format="QUAKEML")
            logger.info("Loaded %d events of %s.", len(catalogs[region]), region)
        if not catalogs:
            raise FileNotFoundError(
                f"No QuakeML catalogs found in {catalog_dir}. "
                "Pass the folder with catalog_dir."
            )
        return catalogs

    def _download_dataset(
        self,
        writer,
        time_before: float = 60.0,
        time_after: float = 60.0,
        **kwargs,
    ):
        """
        :param time_before: Seconds before the first pick of a trace
        :param time_after: Seconds after the last pick of a trace
        """
        logger.info(
            "No pre-downloaded dataset available, downloading from FDSN web services. "
            "This may take a while.",
        )
        self._init_client()
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": COMPONENT_ORDER,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        catalogs = self.get_catalogs()
        asyncio.run(
            self._download_waveforms(
                writer, catalogs, time_before=time_before, time_after=time_after
            )
        )

    def _get_jobs(
        self,
        region: str,
        catalog: Catalog,
        client: MultiClient,
        split_rng: np.random.Generator,
        noise_rng: np.random.Generator,
        stats: Counter,
        failures: list[dict],
        time_before: float,
        time_after: float,
    ) -> list[tuple[FDSNSource, _Job]]:
        """
        Lists the event traces of a catalog, and the noise windows before them.
        """
        origin_times = sorted(
            event.preferred_origin().time.timestamp
            for event in catalog
            if event.preferred_origin() is not None
        )

        jobs = []
        for event in catalog:
            try:
                event_params = get_event_params(event, region)
            except ValueError as exc:
                logger.warning("Skipping event %s: %s", event.resource_id, exc)
                stats["events_skipped"] += 1
                continue
            if self._split is not None:
                _, dev, test = self._split
                event_params["split"] = get_split(split_rng, test=test, dev=dev)
            elif "split" not in event_params:
                logger.warning(
                    "Skipping event %s: the catalog has no split for it. Pass split "
                    "fractions to split the events randomly.",
                    event.resource_id,
                )
                stats["events_skipped"] += 1
                continue
            stats["events"] += 1
            origin_time = UTCDateTime(event_params["source_origin_time"])

            # groupby only groups adjacent picks, so sort them by station first
            picks = sorted(event.picks, key=station_key)
            for _, pick_group in groupby(picks, key=station_key):
                pick_group = list(pick_group)
                waveform_id = pick_group[0].waveform_id
                trace_name = (
                    f"{event_params['source_id']}_{waveform_id.network_code}."
                    f"{waveform_id.station_code}.{waveform_id.location_code or ''}."
                    f"{waveform_id.channel_code[:2]}"
                )
                stats["traces"] += 1
                try:
                    source = client.get_source(waveform_id.network_code)
                except KeyError:
                    stats["traces_no_source"] += 1
                    failures.append(
                        {
                            "kind": "event",
                            "region": region,
                            "trace_name": trace_name,
                            "reason": "No data center serves network "
                            f"{waveform_id.network_code}",
                        }
                    )
                    continue
                jobs.append(
                    (
                        source,
                        _Job(
                            kind="event",
                            region=region,
                            trace_name=trace_name,
                            waveform_id=waveform_id,
                            starttime=min(p.time for p in pick_group) - time_before,
                            endtime=max(p.time for p in pick_group) + time_after,
                            params=event_params,
                            picks=pick_group,
                        ),
                    )
                )

                if noise_rng.random() >= self._noise_fraction:
                    continue
                stats["noise"] += 1
                endtime = origin_time - NOISE_GAP_BEFORE_EVENT
                starttime = endtime - self._noise_length
                if not is_quiet(origin_times, starttime, endtime):
                    stats["noise_near_event"] += 1
                    continue
                jobs.append(
                    (
                        source,
                        _Job(
                            kind="noise",
                            region=region,
                            trace_name=f"noise_{trace_name}",
                            waveform_id=waveform_id,
                            starttime=starttime,
                            endtime=endtime,
                            params={
                                "split": event_params["split"],
                                "source_region": region,
                            },
                        ),
                    )
                )
        return jobs

    async def get_waveform_data(
        self, job: _Job, inventory
    ) -> tuple[TraceParameters, np.ndarray]:
        """
        Downloads the waveforms of an event trace or a noise window.
        """
        waveform_id = job.waveform_id
        kwargs = dict(
            network=waveform_id.network_code,
            station=waveform_id.station_code,
            channel=waveform_id.channel_code[:2] + "?",
            starttime=job.starttime,
            endtime=job.endtime,
            sampling_rate=SAMPLING_RATE,
            component_order=COMPONENT_ORDER,
        )
        try:
            seed_id, waveform_params, data = await get_waveform_array(
                self._client, inventory, location=waveform_id.location_code, **kwargs
            )
        except FDSNNoDataException:
            if not waveform_id.location_code:
                raise
            # Location codes differ between archives, e.g., 1P is 00 in the Eifel
            # catalog but empty at GFZ
            seed_id, waveform_params, data = await get_waveform_array(
                self._client, inventory, location="*", **kwargs
            )

        trace_params = get_trace_params(
            WaveformStreamID(seed_string=seed_id),
            inventory,
            job.starttime,
            job.params if job.kind == "event" else None,
        )
        trace_params.update(waveform_params)
        trace_params["trace_name"] = job.trace_name
        trace_params["trace_category"] = job.kind

        start = UTCDateTime(trace_params["trace_start_time"])
        sampling_rate = trace_params["trace_sampling_rate_hz"]
        for pick in job.picks:
            sample = round((pick.time - start) * sampling_rate)
            trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = sample
            trace_params[f"trace_{pick.phase_hint}_status"] = pick.evaluation_mode

        return trace_params, data

    async def _download_waveforms(
        self,
        writer,
        catalogs: dict[str, Catalog],
        time_before: float,
        time_after: float,
    ):
        # Separate generators, so the noise fraction doesn't change the split
        split_rng = np.random.default_rng(RNG_SEED)
        noise_rng = np.random.default_rng(RNG_SEED + 1)
        stats: dict[str, Counter] = defaultdict(Counter)
        failure_reasons: dict[str, Counter] = defaultdict(Counter)
        failures: list[dict] = []
        events_written: dict[str, set[str]] = defaultdict(set)

        async with self._client as client:
            stations, networks = set(), set()
            for catalog in catalogs.values():
                for event in catalog:
                    for pick in event.picks:
                        stations.add(pick.waveform_id.station_code)
                        networks.add(pick.waveform_id.network_code)
            inv = await get_cached_inventory(
                client,
                self.path / "inventory.xml",
                stations=stations,
                networks=networks,
                force_download=True,
            )

            jobs = []
            for region, catalog in catalogs.items():
                jobs += self._get_jobs(
                    region,
                    catalog,
                    client,
                    split_rng,
                    noise_rng,
                    stats[region],
                    failures,
                    time_before=time_before,
                    time_after=time_after,
                )

            async def download(job: _Job):
                return await self.get_waveform_data(job, inv)

            def write(job: _Job, result, exc: Exception | None):
                region_stats = stats[job.region]
                prefix = "traces" if job.kind == "event" else "noise"
                if exc is not None:
                    reason = f"{type(exc).__name__}: {exc}"
                    logger.debug("Could not download %s: %s", job.trace_name, reason)
                    region_stats[f"{prefix}_failed"] += 1
                    failure_reasons[job.region][type(exc).__name__] += 1
                    failures.append(
                        {
                            "kind": job.kind,
                            "region": job.region,
                            "trace_name": job.trace_name,
                            "reason": reason,
                        }
                    )
                    return
                trace_params, data = result
                writer.add_trace({**job.params, **trace_params}, data)
                region_stats[f"{prefix}_downloaded"] += 1
                if job.kind == "event":
                    events_written[job.region].add(job.params["source_id"])

            await download_parallel(jobs, download, write)

        for region, region_stats in stats.items():
            region_stats["events_downloaded"] = len(events_written[region])
            region_stats["events_without_traces"] = (
                region_stats["events"] - region_stats["events_downloaded"]
            )
        self._write_download_stats(stats, failure_reasons, failures)

    def _write_download_stats(
        self,
        stats: dict[str, Counter],
        failure_reasons: dict[str, Counter],
        failures: list[dict],
    ):
        total = sum(stats.values(), Counter())
        total_reasons = sum(failure_reasons.values(), Counter())
        summary = {
            region: {
                **{key: counts[key] for key in STATS_KEYS},
                "failure_reasons": dict(reasons),
            }
            for region, counts, reasons in [
                *((region, stats[region], failure_reasons[region]) for region in stats),
                ("total", total, total_reasons),
            ]
        }
        with open(self.path / "download_stats.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(self.path / "download_failures.csv", "w", newline="") as f:
            csv_writer = csv.DictWriter(
                f, fieldnames=["kind", "region", "trace_name", "reason"]
            )
            csv_writer.writeheader()
            csv_writer.writerows(failures)

        for region, counts in [*stats.items(), ("total", total)]:
            logger.info(
                "%s: %d/%d events, %d/%d traces downloaded (%d failed, %d without "
                "data center), %d/%d noise windows downloaded (%d failed, %d too "
                "close to an event).",
                region,
                counts["events_downloaded"],
                counts["events"],
                counts["traces_downloaded"],
                counts["traces"],
                counts["traces_failed"],
                counts["traces_no_source"],
                counts["noise_downloaded"],
                counts["noise"],
                counts["noise_failed"],
                counts["noise_near_event"],
            )


def get_source_id(event: Event) -> str:
    """
    ``smi:local/phasenet_dlf_v4b/hawaii/hawaii2_60234378`` becomes
    ``hawaii:hawaii2_60234378``
    """
    region, event_id = str(event.resource_id).split("/")[-2:]
    return f"{region}:{event_id}"


def get_event_params(event: Event, region: str) -> EventParameters:
    origin = event.preferred_origin()
    if origin is None:
        raise ValueError("Event has no preferred origin.")
    comments = dict(
        comment.text.split(": ", 1)
        for comment in event.comments
        if ": " in comment.text
    )

    params: EventParameters = EventParameters(
        source_id=get_source_id(event),
        source_origin_time=str(origin.time),
        source_latitude_deg=origin.latitude,
        source_longitude_deg=origin.longitude,
        source_depth_km=np.nan if origin.depth is None else origin.depth / 1e3,
    )
    if "split" in comments:
        params["split"] = comments["split"]
    params["source_region"] = region
    params["source_event_class"] = comments.get("event_class")
    for description in event.event_descriptions:
        if description.type == "region name":
            params["source_volcano"] = description.text

    magnitude = event.preferred_magnitude()
    if magnitude is not None:
        params["source_magnitude"] = magnitude.mag
        params["source_magnitude_type"] = magnitude.magnitude_type

    return params


def is_quiet(
    origin_times: list[float], starttime: UTCDateTime, endtime: UTCDateTime
) -> bool:
    """
    Whether a window keeps the noise gaps to all events. ``origin_times`` are sorted
    timestamps.
    """
    i = bisect.bisect_right(origin_times, starttime.timestamp - NOISE_GAP_AFTER_EVENT)
    return (
        i == len(origin_times)
        or origin_times[i] >= endtime.timestamp + NOISE_GAP_BEFORE_EVENT
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="Folder with the QuakeML catalogs, <region>.xml.",
    )
    parser.add_argument(
        "--eida-token",
        type=str,
        default=None,
        help="Path to EIDA token file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compile a new dataset. An existing dataset is moved to a backup "
        "folder next to it first.",
    )
    args = parser.parse_args()
    logger.setLevel("INFO")
    if args.eida_token is not None:
        path = Path(args.eida_token).expanduser().resolve()
        if not path.is_file():
            parser.error(f"EIDA token file {path} does not exist.")
        args.eida_token = str(path)
    if args.catalog_dir is not None:
        args.catalog_dir = args.catalog_dir.expanduser().resolve()

    if args.force:
        backup_dataset(DeepLowFrequency._path_internal())

    ds = DeepLowFrequency(
        catalog_dir=args.catalog_dir,
        eida_token=args.eida_token,
        component_order="ZNE",
    )
    print(ds)
