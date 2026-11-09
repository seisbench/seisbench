"""
A client that downloads station metadata and waveforms from several data centers,
FDSN web services and NIED Hi-net, and routes each network to the one serving it.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    NamedTuple,
    TypeVar,
)

import numpy as np
import obspy
from obspy import Inventory, Stream, UTCDateTime, read_inventory
from obspy.core.inventory import Channel, Network, Station
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

from seisbench import logger
from seisbench.data.base import TraceParameters
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


def _import_hinetpy():
    try:
        import HinetPy
    except ImportError as exc:
        raise ImportError(
            "Downloading Hi-net data requires HinetPy and the Hi-net win32tools. "
            "Install HinetPy with `pip install seisbench[hinet]`."
        ) from exc
    return HinetPy


def _format_time(time: datetime | UTCDateTime) -> str:
    return UTCDateTime(time).isoformat()


class WaveformRequest(NamedTuple):
    """
    A waveform window that will be requested from a :py:class:`Client`.
    """

    network: str
    station: str
    location: str
    channel: str
    starttime: UTCDateTime
    endtime: UTCDateTime


class Client(ABC):
    """
    A data center that :py:class:`MultiClient` routes networks to.

    :param connections: Number of parallel connections used to download waveforms
    :param networks: Only use this client for these networks. Pins networks to a
        data center when others hold copies of them, too.
    """

    def __init__(self, connections: int = 4, networks: Iterable[str] | None = None):
        self.connections = connections
        self.networks = None if networks is None else set(networks)

    def served_networks(self, networks: set[str] | None) -> set[str] | None:
        """
        The requested networks this client serves. None means all networks.
        """
        if self.networks is None:
            return networks
        return self.networks if networks is None else networks & self.networks

    async def authenticate(self, session: aiohttp.ClientSession) -> None:
        """
        Called once when the :py:class:`MultiClient` opens.
        """

    def plan(self, requests: Iterable[WaveformRequest]) -> None:
        """
        Announces waveform requests that will follow, so that clients can combine
        them. Does nothing by default.
        """

    @abstractmethod
    async def get_stations(
        self,
        session: aiohttp.ClientSession,
        stations: str,
        starttime: datetime | UTCDateTime,
        endtime: datetime | UTCDateTime,
        level: str,
        networks: set[str] | None = None,
    ) -> Inventory:
        """
        :param stations: Comma-separated station codes
        :param networks: Only these networks, all networks if None
        """

    @abstractmethod
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
        """
        :return: The waveforms, or None if there are no data
        """


class FDSNSource(Client):
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
        super().__init__(connections=connections, networks=networks)
        self.base_url = base_url.rstrip("/")
        self.eida_token = eida_token
        self._auth = None

    def __repr__(self) -> str:
        return f"FDSNSource({self.base_url!r})"

    def __str__(self) -> str:
        return self.base_url

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
        networks = self.served_networks(networks)
        if networks is not None and not networks:
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

    def __init__(self, sources: list[Client]):
        self.sources = sources
        self._network: dict[str, Client] = {}
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

    def get_source(self, network: str) -> Client:
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
        logger.debug("Fetching waveforms from %s", source)
        stream = await source.get_waveforms(
            self.session, network, station, location, channel, starttime, endtime
        )
        if stream is None:
            raise FDSNNoDataException(
                f"{source} has no data for {network}.{station} "
                f"between {starttime} and {endtime}."
            )
        return stream

    def plan(self, requests: Iterable[WaveformRequest]) -> None:
        """
        Announces waveform requests to the sources serving them, see
        :py:meth:`Client.plan`. Requests for unknown networks are ignored.
        """
        by_source = defaultdict(list)
        for request in requests:
            source = self._network.get(request.network)
            if source is not None:
                by_source[source].append(request)
        for source, source_requests in by_source.items():
            source.plan(source_requests)

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


# Hi-net works in Japan Standard Time
JST_OFFSET = 9 * 3600.0
# Hi-net components are U(p), N and E
HINET_COMPONENTS = {"U": "Z", "N": "N", "E": "E"}


class HinetRequest(NamedTuple):
    """
    A Hi-net request: whole minutes (UTC) of several stations.
    """

    starttime: UTCDateTime
    endtime: UTCDateTime
    stations: frozenset[str]

    @property
    def key(self) -> tuple[int, int, frozenset[str]]:
        """
        Hashable key, UTCDateTime is not hashable.
        """
        return self.starttime.ns, self.endtime.ns, self.stations


def _floor_minute(time: UTCDateTime) -> UTCDateTime:
    return UTCDateTime(time.timestamp // 60 * 60)


def _ceil_minute(time: UTCDateTime) -> UTCDateTime:
    return UTCDateTime(-(-time.timestamp // 60) * 60)


class HinetSource(Client):
    """
    NIED Hi-net (`doi:10.17598/NIED.0003 <https://doi.org/10.17598/NIED.0003>`_), via
    `HinetPy <https://github.com/seisman/HinetPy>`_. Requires an account, HinetPy and
    the Hi-net win32tools (``catwin32`` and ``win2sac_32``) on the ``PATH``.

    Hi-net serves whole minutes, for the stations selected in the account. Requests
    announced with :py:meth:`plan` are combined into requests of several stations and
    minutes, and the extracted data are cached. Other requests are fetched one by
    one. The account's station selection is changed by every request, so the
    requests run one at a time.

    Stations are labeled ``<network>.<station>..EH<Z|N|E>`` (short-period, 100 Hz).
    The data are velocity as extracted by ``win2sac_32``, not counts.

    :param user: Hi-net user name
    :param password: Hi-net password
    :param network: Network code of the stations
    :param max_span: Maximum length of a combined request in minutes
    :param max_gap: Requests up to this many minutes apart are combined
    :param retries: Attempts per request, for dropped connections
    :param cache_size: Number of combined requests kept in memory
    """

    code = "0101"

    def __init__(
        self,
        user: str,
        password: str,
        network: str = "N",
        max_span: int = 20,
        max_gap: int = 5,
        retries: int = 3,
        cache_size: int = 16,
    ):
        super().__init__(connections=1, networks=[network])
        self.network = network
        self.max_span = max_span
        self.max_gap = max_gap
        self.retries = retries
        self.cache_size = cache_size
        self._user = user
        self._password = password
        self._client = None
        self._stations = None
        self._planned: dict[str, list[HinetRequest]] = defaultdict(list)
        self._cache: OrderedDict[tuple, asyncio.Future] = OrderedDict()
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"HinetSource(network={self.network!r})"

    def __str__(self) -> str:
        return "NIED Hi-net"

    async def authenticate(self, session: aiohttp.ClientSession) -> None:
        HinetPy = _import_hinetpy()
        self._client = await asyncio.to_thread(
            HinetPy.Client, self._user, self._password
        )
        logger.info("Logged in to Hi-net as %s.", self._user)

    async def get_stations(
        self,
        session: aiohttp.ClientSession,
        stations: str,
        starttime: datetime | UTCDateTime,
        endtime: datetime | UTCDateTime,
        level: str,
        networks: set[str] | None = None,
    ) -> Inventory:
        """
        Station coordinates and ZNE channels of the requested stations. The level,
        start and end time are ignored.
        """
        if not self.served_networks(networks):
            return Inventory()
        if self._stations is None:
            logger.info("Fetching the Hi-net station list")
            self._stations = await asyncio.to_thread(
                self._client.get_station_list, self.code
            )
        codes = set(stations.split(","))

        network = Network(code=self.network)
        for hinet_station in self._stations:
            code = hinet_station.name.split(".")[-1]
            if code not in codes or hinet_station.latitude is None:
                continue
            coordinates = dict(
                latitude=hinet_station.latitude,
                longitude=hinet_station.longitude,
                elevation=hinet_station.elevation or 0.0,
            )
            channels = [
                Channel(
                    code=f"EH{component}",
                    location_code="",
                    depth=0.0,
                    azimuth=azimuth,
                    dip=dip,
                    sample_rate=100.0,
                    **coordinates,
                )
                for component, azimuth, dip in (
                    ("Z", 0, -90),
                    ("N", 0, 0),
                    ("E", 90, 0),
                )
            ]
            network.stations.append(
                Station(code=code, site=None, channels=channels, **coordinates)
            )
        return Inventory(networks=[network] if network.stations else [])

    def plan(self, requests: Iterable[WaveformRequest]) -> None:
        """
        Combines the requests into requests of whole minutes, for all stations
        within :py:attr:`max_span` minutes. Requests less than :py:attr:`max_gap`
        minutes apart are combined, e.g., an event and the noise before it.
        """
        current = None
        for request in sorted(requests, key=lambda r: r.starttime):
            start = _floor_minute(UTCDateTime(request.starttime))
            end = _ceil_minute(UTCDateTime(request.endtime))
            if (
                current is not None
                and start <= current.endtime + self.max_gap * 60
                and max(end, current.endtime) - current.starttime <= self.max_span * 60
            ):
                current = HinetRequest(
                    current.starttime,
                    max(end, current.endtime),
                    current.stations | {request.station},
                )
            else:
                self._add_planned(current)
                current = HinetRequest(start, end, frozenset({request.station}))
        self._add_planned(current)

    def _add_planned(self, request: HinetRequest | None) -> None:
        if request is not None:
            for station in request.stations:
                self._planned[station].append(request)

    def _find_planned(
        self, station: str, starttime: UTCDateTime, endtime: UTCDateTime
    ) -> HinetRequest:
        for request in self._planned.get(station, ()):
            if request.starttime <= starttime and endtime <= request.endtime:
                return request
        return HinetRequest(
            _floor_minute(starttime), _ceil_minute(endtime), frozenset({station})
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
        starttime, endtime = UTCDateTime(starttime), UTCDateTime(endtime)
        request = self._find_planned(station, starttime, endtime)

        if request.key in self._cache:
            self._cache.move_to_end(request.key)
            future = self._cache[request.key]
        else:
            future = asyncio.get_running_loop().create_future()
            self._cache[request.key] = future
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            try:
                async with self._lock:
                    future.set_result(await asyncio.to_thread(self._fetch, request))
            except Exception as exc:
                future.set_exception(exc)
                self._cache.pop(request.key, None)
        stream = await asyncio.shield(future)

        selected = Stream(
            [
                trace
                for trace in stream.select(station=station)
                if fnmatch(trace.stats.channel, channel or "*")
            ]
        )
        selected = selected.slice(starttime, endtime)
        return selected if len(selected) else None

    def _fetch(self, request: HinetRequest) -> Stream:
        """
        Requests and extracts the data of a combined request, with retries for
        dropped connections. Runs in a thread.
        """
        for attempt in range(1, self.retries + 1):
            try:
                return self._fetch_once(request)
            except OSError as exc:  # requests' ConnectionError is an OSError
                if attempt == self.retries:
                    raise
                logger.warning(
                    "Hi-net request failed (%s), retrying %d/%d.",
                    exc,
                    attempt,
                    self.retries - 1,
                )
                time.sleep(10)

    def _fetch_once(self, request: HinetRequest) -> Stream:
        HinetPy = _import_hinetpy()
        names = [f"{self.network}.{station}" for station in sorted(request.stations)]
        span = round((request.endtime - request.starttime) / 60)
        logger.debug(
            "Requesting %d Hi-net stations, %s +%d min",
            len(names),
            request.starttime,
            span,
        )

        with tempfile.TemporaryDirectory() as tmp:
            self._client.select_stations(self.code, names)
            data, ctable = self._client.get_continuous_waveform(
                self.code,
                (request.starttime + JST_OFFSET).datetime,
                span,
                outdir=tmp,
                threads=1,
            )
            if data is None:
                return Stream()
            HinetPy.win32.extract_sac(
                data, ctable, outdir=tmp, filter_by_name=names, processes=1
            )
            stream = Stream()
            for path in sorted(Path(tmp).glob("*.SAC")):
                # Files are named <network>.<station>.<component>.SAC
                network, station, component = path.name.split(".")[:3]
                if component not in HINET_COMPONENTS:
                    continue
                trace = obspy.read(str(path), format="SAC")[0]
                trace.stats.network = network
                trace.stats.station = station
                trace.stats.location = ""
                trace.stats.channel = f"EH{HINET_COMPONENTS[component]}"
                trace.stats.starttime -= JST_OFFSET
                stream += trace
        return stream


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
    jobs: Iterable[tuple[Client, Job]],
    download: Callable[[Job], Awaitable[Result]],
    write: Callable[[Job, Result | None, Exception | None], None],
    log_every: int = 1000,
    checkpoint: Callable[[], None] | None = None,
    checkpoint_interval: float = 600.0,
) -> None:
    """
    Runs ``download`` for all jobs. Each source has its own queue, worked on by as
    many tasks as the source has connections. ``write`` is called for every job,
    with the result or the exception ``download`` raised. It runs in one thread at a
    time, so a dataset writer is never used concurrently.

    :param jobs: Pairs of the source serving a job and the job
    :param checkpoint: Called every ``checkpoint_interval`` seconds, between two calls
        of ``write``, e.g., :py:meth:`~seisbench.data.WaveformDataWriter.checkpoint`
    """
    queues: dict[Client, asyncio.Queue] = defaultdict(asyncio.Queue)
    for source, job in jobs:
        queues[source].put_nowait(job)
    n_jobs = sum(queue.qsize() for queue in queues.values())
    logger.info(
        "Downloading %d waveforms (%s).",
        n_jobs,
        ", ".join(f"{queue.qsize()} from {source}" for source, queue in queues.items()),
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
        last_checkpoint = time.monotonic()
        for i_job in range(n_jobs):
            job, result, exc = await results.get()
            await asyncio.to_thread(write, job, result, exc)
            n_written += exc is None
            if (
                checkpoint is not None
                and time.monotonic() - last_checkpoint >= checkpoint_interval
            ):
                await asyncio.to_thread(checkpoint)
                last_checkpoint = time.monotonic()
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
