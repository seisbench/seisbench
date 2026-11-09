"""
Helpers to download station metadata and waveforms from FDSN web services.
"""

from __future__ import annotations

import asyncio
import io
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, Literal, TypeVar

import numpy as np
import obspy
from obspy import Inventory, Stream, UTCDateTime, read_inventory
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from obspy.core.event.base import QuantityError, WaveformStreamID
from obspy.core.event.origin import Pick
from obspy.geodetics import gps2dist_azimuth

from seisbench import logger
from seisbench.data.base import EventParameters, TraceParameters
from seisbench.util.trace_ops import (
    rotate_stream_to_zne,
    stream_to_array,
    trace_has_spikes,
)

if TYPE_CHECKING:
    import aiohttp


def _import_aiohttp():
    try:
        import aiohttp
    except ImportError as exc:
        raise ImportError(
            "Compiling datasets from FDSN web services requires "
            "aiohttp. Install it with `pip install seisbench[download]`."
        ) from exc
    return aiohttp


def _format_time(time: datetime | UTCDateTime) -> str:
    return UTCDateTime(time).isoformat()


class FDSNSource:
    """
    An FDSN web service data center.

    :param base_url: Base URL of the data center, e.g., ``https://geofon.gfz.de``
    :param connections: Number of parallel connections used to download waveforms
    :param eida_token: EIDA token or path to a token file, for restricted data
    :param networks: Only use this source for these networks. Pins networks to a
        data center when others hold copies of them, too.
    """

    def __init__(
        self,
        base_url: str,
        connections: int = 4,
        eida_token: str | None = None,
        networks: Iterable[str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.connections = connections
        self.eida_token = eida_token
        self.networks = None if networks is None else set(networks)
        self._auth = None

    def __repr__(self) -> str:
        return f"FDSNSource({self.base_url!r})"

    async def authenticate(self, session: aiohttp.ClientSession) -> None:
        """
        Exchanges the EIDA token for temporary credentials, which are then used
        for all waveform requests.
        """
        if not self.eida_token:
            return
        aiohttp = _import_aiohttp()

        token = self.eida_token
        if Path(token).is_file():
            token = Path(token).read_text()

        url = f"{self.base_url}/fdsnws/dataselect/1/auth"
        if not url.startswith("https://"):
            raise ValueError(f"Refusing to send the EIDA token to {url} without TLS.")
        async with session.post(url, data=token.encode()) as response:
            if response.status != 200:
                raise FDSNException(
                    f"EIDA authentication at {url} failed with HTTP "
                    f"{response.status}: {(await response.text())[:200]}"
                )
            login, password = (await response.text()).strip().split(":")
        self._auth = aiohttp.DigestAuthMiddleware(login, password)
        logger.info("Authenticated at %s with EIDA token.", self.base_url)

    async def _get(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict[str, str],
        **kwargs,
    ) -> bytes | None:
        async with session.get(url, params=params, **kwargs) as response:
            # 204 and 404 are both valid "no data" responses in FDSN web services
            if response.status in (204, 404):
                return None
            if response.status != 200:
                raise FDSNException(
                    f"{response.url} returned HTTP {response.status}: "
                    f"{(await response.text())[:200]}"
                )
            return await response.read()

    async def get_stations(
        self,
        session: aiohttp.ClientSession,
        stations: str,
        starttime: datetime | UTCDateTime,
        endtime: datetime | UTCDateTime,
        level: str,
        networks: set[str] | None = None,
    ) -> Inventory:
        if self.networks is not None:
            networks = self.networks if networks is None else networks & self.networks
            if not networks:
                return Inventory()
        logger.info("Fetching inventory from %s", self.base_url)
        params = {
            "network": "*" if networks is None else ",".join(sorted(networks)),
            "station": stations,
            "location": "*",
            "channel": "*",
            "level": level,
            "starttime": _format_time(starttime),
            "endtime": _format_time(endtime),
            "format": "xml",
        }
        body = await self._get(
            session, f"{self.base_url}/fdsnws/station/1/query", params
        )
        if body is None:
            return Inventory()
        return await asyncio.to_thread(
            read_inventory, io.BytesIO(body), format="STATIONXML"
        )

    async def get_waveforms(
        self,
        session: aiohttp.ClientSession,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: datetime | UTCDateTime,
        endtime: datetime | UTCDateTime,
    ) -> Stream | None:
        params = {
            "network": network,
            "station": station,
            "location": location or "*",
            "channel": channel or "*",
            "starttime": _format_time(starttime),
            "endtime": _format_time(endtime),
        }
        if self._auth is not None:
            url = f"{self.base_url}/fdsnws/dataselect/1/queryauth"
            body = await self._get(session, url, params, middlewares=(self._auth,))
        else:
            url = f"{self.base_url}/fdsnws/dataselect/1/query"
            body = await self._get(session, url, params)
        if body is None:
            return None
        return await asyncio.to_thread(obspy.read, io.BytesIO(body), format="MSEED")


class MultiClient:
    """
    Async FDSN client that routes each network to the data center serving it.

    Use it as an async context manager. It holds one HTTP session for all sources.
    Networks are mapped to sources by :py:meth:`get_inventory`.
    """

    def __init__(self, sources: list[FDSNSource]):
        self.sources = sources
        self._network: dict[str, FDSNSource] = {}
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> MultiClient:
        aiohttp = _import_aiohttp()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(sock_connect=30, sock_read=120),
        )
        try:
            await asyncio.gather(
                *(source.authenticate(self._session) for source in self.sources)
            )
        except BaseException:
            await self._session.close()
            raise
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._session.close()
        self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("MultiClient must be used as an async context manager.")
        return self._session

    def get_source(self, network: str) -> FDSNSource:
        try:
            return self._network[network]
        except KeyError:
            raise KeyError(
                f"No source known for network {network}. "
                "Call get_inventory() first to discover networks."
            ) from None

    async def get_waveforms(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: datetime = datetime.min,
        endtime: datetime = datetime.max,
    ) -> Stream:
        source = self.get_source(network)
        logger.debug("Fetching waveforms from %s", source.base_url)
        stream = await source.get_waveforms(
            self.session, network, station, location, channel, starttime, endtime
        )
        if stream is None:
            raise FDSNNoDataException(
                f"{source.base_url} has no data for {network}.{station} "
                f"between {starttime} and {endtime}."
            )
        return stream

    async def get_inventory(
        self,
        stations: str | Iterable[str],
        starttime: datetime = datetime.min,
        endtime: datetime = datetime.max,
        level: Literal["response", "channel", "station", "network"] = "channel",
        networks: Iterable[str] | None = None,
    ) -> Inventory:
        """
        Fetches the inventory from all sources and maps each network to the first
        source, in the order of :py:attr:`sources`, that serves it.

        :param networks: Only query these networks, all networks if None
        """
        if isinstance(stations, str):
            stations = [stations]
        station_list = ",".join(sorted(stations))
        if networks is not None:
            networks = set(networks)

        inventories = await asyncio.gather(
            *(
                source.get_stations(
                    self.session, station_list, starttime, endtime, level, networks
                )
                for source in self.sources
            )
        )
        inventory = Inventory()
        for source, inv in zip(self.sources, inventories):
            for network in inv:
                self._network.setdefault(network.code, source)
            inventory.extend(inv)
        return inventory


async def get_cached_inventory(
    client: MultiClient,
    inventory_file: Path,
    stations: Iterable[str],
    networks: Iterable[str] | None = None,
    force_download: bool = False,
) -> Inventory:
    """
    Downloads the inventory of the stations, or loads it from ``inventory_file``.
    Either way, the client learns which source serves which network. Must be called
    within the client's context.
    """
    if not inventory_file.exists() or force_download:
        inv = await client.get_inventory(stations=stations, networks=networks)
        await asyncio.to_thread(inv.write, str(inventory_file), format="STATIONXML")
    else:
        logger.info("Loading inventory from %s", inventory_file)
        inv = await asyncio.to_thread(read_inventory, str(inventory_file))
        # The file doesn't record which source serves which network
        await client.get_inventory(
            stations=stations, networks=networks, level="network"
        )
    return inv


Job = TypeVar("Job")
Result = TypeVar("Result")


async def download_parallel(
    jobs: Iterable[tuple[FDSNSource, Job]],
    download: Callable[[Job], Awaitable[Result]],
    write: Callable[[Job, Result | None, Exception | None], None],
    log_every: int = 1000,
) -> None:
    """
    Runs ``download`` for all jobs. Each source has its own queue, worked on by as
    many tasks as the source has connections. ``write`` is called for every job,
    with the result or the exception ``download`` raised. It runs in one thread at a
    time, so a dataset writer is never used concurrently.

    :param jobs: Pairs of the source serving a job and the job
    """
    queues: dict[FDSNSource, asyncio.Queue] = defaultdict(asyncio.Queue)
    for source, job in jobs:
        queues[source].put_nowait(job)
    n_jobs = sum(queue.qsize() for queue in queues.values())
    logger.info(
        "Downloading %d waveforms (%s).",
        n_jobs,
        ", ".join(
            f"{queue.qsize()} from {source.base_url}"
            for source, queue in queues.items()
        ),
    )
    results = asyncio.Queue(maxsize=64)

    async def worker(queue: asyncio.Queue):
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await results.put((job, await download(job), None))
            except Exception as exc:
                await results.put((job, None, exc))

    async def writer():
        n_written = 0
        for i_job in range(n_jobs):
            job, result, exc = await results.get()
            await asyncio.to_thread(write, job, result, exc)
            n_written += exc is None
            if (i_job + 1) % log_every == 0 or i_job + 1 == n_jobs:
                logger.info(
                    "Processed %d/%d waveforms, %d downloaded.",
                    i_job + 1,
                    n_jobs,
                    n_written,
                )

    await asyncio.gather(
        writer(),
        *(
            worker(queue)
            for source, queue in queues.items()
            for _ in range(source.connections)
        ),
    )


def select_location(stream: Stream, location: str) -> Stream:
    """
    Keeps the traces of one location code: the requested one if present, else the
    first one. Several locations would otherwise be mixed into one array.
    """
    locations = sorted({trace.stats.location for trace in stream})
    if len(locations) <= 1:
        return stream
    selected = location if location in locations else locations[0]
    return stream.select(location=selected)


async def homogenize_sampling_rate(st: Stream, sampling_rate: float) -> float:
    """
    Homogenizes the sampling rate of the stream to the given sampling rate.
    """
    if sampling_rate <= 0.0:
        sampling_rate = st[0].stats.sampling_rate
    for trace in st:
        if trace.stats.sampling_rate != sampling_rate:
            logger.info(
                "Resampling trace %s from %.2f Hz to %.2f Hz.",
                trace.id,
                trace.stats.sampling_rate,
                sampling_rate,
            )
            await asyncio.to_thread(trace.resample, sampling_rate)
    return sampling_rate


async def get_waveform_array(
    client: MultiClient,
    inventory: Inventory,
    network: str,
    station: str,
    location: str,
    channel: str,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    sampling_rate: float,
    component_order: str,
) -> tuple[str, TraceParameters, np.ndarray]:
    """
    Downloads a waveform window, rotates it to ``component_order``, resamples it and
    converts it to an array.

    :return: SEED id of the first downloaded trace, the trace parameters describing
        the array, and the array
    """
    try:
        stream = await client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime.datetime,
            endtime=endtime.datetime,
        )
    except FDSNException:
        logger.debug("Error fetching waveforms for %s.%s", network, station)
        raise

    if not len(stream):
        raise ValueError(
            f"No waveforms found for {network}.{station}.{location}.{channel} "
            f"in time range {starttime} - {endtime}.",
        )
    stream = select_location(stream, location)
    seed_id = stream[0].id

    rotate_stream_to_zne(stream, inventory=inventory)
    sampling_rate = await homogenize_sampling_rate(stream, sampling_rate)

    stream = stream.slice(starttime, endtime)
    if not len(stream):
        raise ValueError(
            f"No waveforms for {seed_id} between {starttime} and {endtime}."
        )
    actual_t_start, data, completeness = stream_to_array(
        stream,
        component_order=component_order,
    )
    desired_samples = int((endtime - starttime) * sampling_rate) + 1
    if desired_samples > data.shape[1]:
        # Traces appear to be complete, but do not cover the intended time range
        completeness *= data.shape[1] / desired_samples

    trace_params = TraceParameters(
        station_location_code=stream[0].stats.location,
        trace_sampling_rate_hz=sampling_rate,
        trace_completeness=completeness,
        trace_has_spikes=trace_has_spikes(data),
        trace_start_time=str(actual_t_start),
        trace_component_order=component_order,
    )
    return seed_id, trace_params, data


def station_key(pick: Pick) -> tuple[str, str, str]:
    """
    Groups picks by station, e.g., with :py:func:`itertools.groupby`.
    """
    waveform_id = pick.waveform_id
    return (
        waveform_id.network_code or "",
        waveform_id.station_code or "",
        waveform_id.location_code or "",
    )


def uncertainty(errors: QuantityError | None) -> float:
    if errors is None or errors.uncertainty is None:
        return np.nan
    return errors.uncertainty


def get_trace_params(
    waveform_id: WaveformStreamID,
    inventory: Inventory,
    time: UTCDateTime,
    event_params: EventParameters | None = None,
) -> TraceParameters:
    """
    Station parameters of a channel at the given time. With ``event_params``, the
    back azimuth from the event is included.
    """
    try:
        coordinates = inventory.get_coordinates(
            waveform_id.get_seed_string(),
            datetime=time,
        )
    except Exception as r:
        raise ValueError(
            f"Could not get coordinates for {waveform_id.get_seed_string()}: {r}"
        ) from r

    trace_params = TraceParameters(
        station_network_code=waveform_id.network_code,
        station_code=waveform_id.station_code,
        trace_channel=waveform_id.channel_code[:2],
        station_location_code=waveform_id.location_code,
        station_latitude_deg=coordinates["latitude"],
        station_longitude_deg=coordinates["longitude"],
        station_elevation_m=coordinates["elevation"],
    )
    if event_params is not None:
        if not np.isnan(coordinates["latitude"] * coordinates["longitude"]):
            back_azimuth = gps2dist_azimuth(
                event_params["source_latitude_deg"],
                event_params["source_longitude_deg"],
                coordinates["latitude"],
                coordinates["longitude"],
            )[2]
        else:
            back_azimuth = np.nan
        trace_params["path_back_azimuth_deg"] = back_azimuth
    return trace_params


def get_split(
    rng: np.random.Generator, test: float = 0.1, dev: float = 0.1
) -> Literal["train", "dev", "test"]:
    """
    Returns a random split.
    """
    r = rng.random()
    train = 1.0 - test - dev
    if r < train:
        return "train"
    elif r < train + dev:
        return "dev"
    else:
        return "test"


def backup_dataset(path: Path) -> None:
    """
    Moves an existing dataset folder to a timestamped backup folder next to it.
    """
    if not path.exists():
        return
    backup_path = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%dT%H%M%S}")
    logger.info("Moving existing dataset to %s", backup_path)
    path.rename(backup_path)
