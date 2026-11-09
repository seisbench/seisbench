from __future__ import annotations

import asyncio
import bisect
import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import groupby
from pathlib import Path
from typing import Literal, NamedTuple

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

# Events closer than this (s) in a catalog form a cluster, e.g., a swarm. A cluster is
# never split between train, dev and test, so its waveforms don't leak across them.
SPLIT_CLUSTER_GAP = 3600.0

RNG_SEED = 31337

Kind = Literal["event", "noise"]
Status = Literal["downloaded", "failed", "no_source", "near_event"]


class Outcome(NamedTuple):
    """
    What happened to one event trace or noise window.
    """

    kind: Kind
    region: str
    source_id: str
    trace_name: str
    status: Status
    error: str = ""
    message: str = ""


class Job(NamedTuple):
    """
    An event trace or noise window to download.
    """

    kind: Kind
    region: str
    source_id: str
    trace_name: str
    waveform_id: WaveformStreamID
    starttime: UTCDateTime
    endtime: UTCDateTime
    params: dict
    picks: tuple[Pick, ...] = ()

    def outcome(self, status: Status, exc: Exception | None = None) -> Outcome:
        return Outcome(
            kind=self.kind,
            region=self.region,
            source_id=self.source_id,
            trace_name=self.trace_name,
            status=status,
            error="" if exc is None else type(exc).__name__,
            message="" if exc is None else str(exc),
        )


class RegionStats(NamedTuple):
    """
    Download counts of one region.
    """

    region: str
    events: int
    events_downloaded: int
    traces: int
    traces_downloaded: int
    traces_failed: int
    traces_no_source: int
    noise: int
    noise_downloaded: int
    noise_failed: int
    noise_near_event: int
    failure_reasons: dict[str, int]

    @classmethod
    def from_outcomes(
        cls, region: str, events: int, outcomes: Iterable[Outcome]
    ) -> RegionStats:
        outcomes = list(outcomes)
        count = Counter((outcome.kind, outcome.status) for outcome in outcomes)
        return cls(
            region=region,
            events=events,
            events_downloaded=len(
                {
                    outcome.source_id
                    for outcome in outcomes
                    if outcome.kind == "event" and outcome.status == "downloaded"
                }
            ),
            traces=sum(n for (kind, _), n in count.items() if kind == "event"),
            traces_downloaded=count["event", "downloaded"],
            traces_failed=count["event", "failed"],
            traces_no_source=count["event", "no_source"],
            noise=sum(n for (kind, _), n in count.items() if kind == "noise"),
            noise_downloaded=count["noise", "downloaded"],
            noise_failed=count["noise", "failed"],
            noise_near_event=count["noise", "near_event"],
            failure_reasons=dict(
                Counter(o.error for o in outcomes if o.status == "failed")
            ),
        )

    def __str__(self) -> str:
        return (
            f"{self.region}: {self.events_downloaded}/{self.events} events, "
            f"{self.traces_downloaded}/{self.traces} traces downloaded "
            f"({self.traces_failed} failed, {self.traces_no_source} without data "
            f"center), {self.noise_downloaded}/{self.noise} noise windows downloaded "
            f"({self.noise_failed} failed, {self.noise_near_event} too close to an "
            "event)."
        )


class DownloadStats(NamedTuple):
    """
    Download counts per region and in total, and every trace that was not
    downloaded.
    """

    regions: tuple[RegionStats, ...]
    total: RegionStats
    failures: tuple[Outcome, ...]

    @classmethod
    def from_outcomes(
        cls, events: Mapping[str, int], outcomes: Iterable[Outcome]
    ) -> DownloadStats:
        """
        :param events: Number of events per region
        """
        outcomes = list(outcomes)
        return cls(
            regions=tuple(
                RegionStats.from_outcomes(
                    region, n_events, (o for o in outcomes if o.region == region)
                )
                for region, n_events in events.items()
            ),
            total=RegionStats.from_outcomes("total", sum(events.values()), outcomes),
            failures=tuple(o for o in outcomes if o.status != "downloaded"),
        )

    def __str__(self) -> str:
        return "\n".join(str(stats) for stats in (*self.regions, self.total))

    def write(self, folder: Path) -> None:
        """
        Writes the counts to ``download_stats.json`` and the failures to
        ``download_failures.csv``.
        """
        with open(folder / "download_stats.json", "w") as f:
            json.dump(
                {
                    stats.region: stats._asdict()
                    for stats in (*self.regions, self.total)
                },
                f,
                indent=2,
            )
        with open(folder / "download_failures.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(Outcome._fields)
            writer.writerows(self.failures)


class DeepLowFrequency(WaveformBenchmarkDataset):
    """
    Deep low-frequency (DLF) earthquakes, also called deep long-period (DLP) events,
    below volcanoes in Alaska, Hawaii, Northern California, the Cascades and the
    Eifel, plus noise windows recorded at the same stations.

    The dataset is compiled from FDSN web services (EarthScope, NCEDC, BGR, GFZ). The
    catalogs are read from one QuakeML file per region in ``catalog_dir``. The event
    class is an event comment (``event_class: DLP``). The events of each catalog are
    split into train, dev and test with the fractions in ``split``. Events less than
    an hour apart, e.g., of a swarm, form a cluster, and whole clusters are assigned
    randomly, so that overlapping waveforms do not end up in different splits.

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
    :param split: Fractions of train, dev and test events, applied to each catalog
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
        split: tuple[float, float, float] = (0.7, 0.1, 0.2),
        noise_fraction: float = 0.3,
        noise_length: float = 60.0,
        eida_token: str | None = None,
        connections: int = 4,
        compile_from_source: bool = True,
        **kwargs,
    ):
        if len(split) != 3 or not np.isclose(sum(split), 1.0):
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
        rng: np.random.Generator,
        outcomes: list[Outcome],
        time_before: float,
        time_after: float,
    ) -> list[tuple[FDSNSource, Job]]:
        """
        Lists the event traces of a catalog, and the noise windows before them.
        Traces and windows that can't be downloaded are added to ``outcomes``.
        """
        events = []
        for event in catalog:
            if event.preferred_origin() is None:
                logger.warning("Skipping event %s without origin.", event.resource_id)
                continue
            events.append(event)
        origin_times = [event.preferred_origin().time.timestamp for event in events]
        splits = split_events(origin_times, self._split, rng)
        logger.info(
            "Split of %s: %s events.",
            region,
            ", ".join(f"{n} {split}" for split, n in sorted(Counter(splits).items())),
        )
        origin_times.sort()

        jobs = []
        for event, split in zip(events, splits):
            event_params = get_event_params(event, region)
            event_params["split"] = split
            source_id = event_params["source_id"]
            origin_time = UTCDateTime(event_params["source_origin_time"])

            # groupby only groups adjacent picks, so sort them by station first
            picks = sorted(event.picks, key=station_key)
            for _, pick_group in groupby(picks, key=station_key):
                pick_group = tuple(pick_group)
                waveform_id = pick_group[0].waveform_id
                trace_name = (
                    f"{source_id}_{waveform_id.network_code}."
                    f"{waveform_id.station_code}.{waveform_id.location_code or ''}."
                    f"{waveform_id.channel_code[:2]}"
                )
                event_job = Job(
                    kind="event",
                    region=region,
                    source_id=source_id,
                    trace_name=trace_name,
                    waveform_id=waveform_id,
                    starttime=min(p.time for p in pick_group) - time_before,
                    endtime=max(p.time for p in pick_group) + time_after,
                    params=event_params,
                    picks=pick_group,
                )
                noise_job = None
                if rng.random() < self._noise_fraction:
                    endtime = origin_time - NOISE_GAP_BEFORE_EVENT
                    noise_job = Job(
                        kind="noise",
                        region=region,
                        source_id=source_id,
                        trace_name=f"noise_{trace_name}",
                        waveform_id=waveform_id,
                        starttime=endtime - self._noise_length,
                        endtime=endtime,
                        params={"split": split, "source_region": region},
                    )

                try:
                    source = client.get_source(waveform_id.network_code)
                except KeyError as exc:
                    outcomes += [
                        job.outcome("no_source", exc)
                        for job in (event_job, noise_job)
                        if job is not None
                    ]
                    continue
                jobs.append((source, event_job))
                if noise_job is None:
                    continue
                if is_quiet(origin_times, noise_job.starttime, noise_job.endtime):
                    jobs.append((source, noise_job))
                else:
                    outcomes.append(noise_job.outcome("near_event"))
        return jobs

    async def get_waveform_data(
        self, job: Job, inventory
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
        rng = np.random.default_rng(RNG_SEED)
        outcomes: list[Outcome] = []

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
                    rng,
                    outcomes,
                    time_before=time_before,
                    time_after=time_after,
                )

            async def download(job: Job):
                return await self.get_waveform_data(job, inv)

            def write(job: Job, result, exc: Exception | None):
                if exc is not None:
                    logger.debug("Could not download %s: %s", job.trace_name, exc)
                    outcomes.append(job.outcome("failed", exc))
                    return
                trace_params, data = result
                writer.add_trace({**job.params, **trace_params}, data)
                outcomes.append(job.outcome("downloaded"))

            await download_parallel(jobs, download, write)

        stats = DownloadStats.from_outcomes(
            {region: len(catalog) for region, catalog in catalogs.items()}, outcomes
        )
        stats.write(self.path)
        logger.info("Download statistics:\n%s", stats)


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


def split_events(
    origin_times: list[float],
    fractions: tuple[float, float, float],
    rng: np.random.Generator,
) -> list[Literal["train", "dev", "test"]]:
    """
    Assigns events to train, dev and test in about the given fractions. Events less
    than :py:data:`SPLIT_CLUSTER_GAP` apart form a cluster. In random order, each
    cluster goes as a whole to the split that is furthest below its share.

    :param origin_times: Origin timestamps of the events
    :return: The split of each event
    """
    clusters: list[list[int]] = []
    for i in np.argsort(origin_times):
        if clusters and (
            origin_times[i] - origin_times[clusters[-1][-1]] < SPLIT_CLUSTER_GAP
        ):
            clusters[-1].append(i)
        else:
            clusters.append([i])

    # Each cluster goes to the split that is furthest below its share
    names = ("train", "dev", "test")
    targets = np.array(fractions) * len(origin_times)
    counts = np.zeros(3)
    splits = [None] * len(origin_times)
    for i_cluster in rng.permutation(len(clusters)):
        i_split = int(np.argmax(targets - counts))
        for i in clusters[i_cluster]:
            splits[i] = names[i_split]
        counts[i_split] += len(clusters[i_cluster])
    return splits


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


__all__ = ["DeepLowFrequency"]
