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
from collections import defaultdict
from datetime import datetime
from fnmatch import fnmatch
from fractions import Fraction
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
from scipy import signal
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

import seisbench
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
    A waveform window to request from a :py:class:`Source`.
    """

    network: str
    station: str
    location: str
    channel: str
    starttime: UTCDateTime
    endtime: UTCDateTime


class Source(ABC):
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
        stations: set[str],
        starttime: datetime | UTCDateTime,
        endtime: datetime | UTCDateTime,
        level: str,
        networks: set[str] | None = None,
    ) -> Inventory:
        """
        :param stations: Station codes
        :param networks: Only these networks, all networks if None
        """

    @abstractmethod
    async def get_waveforms(
        self, session: aiohttp.ClientSession, request: WaveformRequest
    ) -> Stream | None:
        """
        :return: The waveforms, or None if there are no data
        """


class _UnauthorizedError(FDSNException):
    """
    The server rejected the credentials (HTTP 401).
    """


def _read_eida_token(token: str) -> str:
    """
    Returns the token itself, or reads it from a token file.

    :raises FileNotFoundError: If ``token`` is neither a token nor an existing file
    """
    if "BEGIN PGP" in token:
        return token
    path = Path(token).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"EIDA token file {path} does not exist.")
    return path.read_text()


class FDSNSource(Source):
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
        self.eida_token = _read_eida_token(eida_token) if eida_token else None
        self._auth = None
        self._auth_lock = asyncio.Lock()

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

        url = f"{self.base_url}/fdsnws/dataselect/1/auth"
        if not url.startswith("https://"):
            raise ValueError(f"Refusing to send the EIDA token to {url} without TLS.")
        async with session.post(url, data=self.eida_token.encode()) as response:
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
            if response.status == 401:
                raise _UnauthorizedError(f"{response.url} returned HTTP 401.")
            if response.status != 200:
                raise FDSNException(
                    f"{response.url} returned HTTP {response.status}: "
                    f"{(await response.text())[:200]}"
                )
            return await response.read()

    async def get_stations(
        self,
        session: aiohttp.ClientSession,
        stations: set[str],
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
            "station": ",".join(sorted(stations)),
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

    async def _get_authenticated(
        self, session: aiohttp.ClientSession, params: dict[str, str]
    ) -> bytes | None:
        """
        Requests restricted waveforms. The server forgets the temporary credentials
        after 24 h at the latest, or when it restarts. Then the EIDA token is
        exchanged again, once for all concurrent requests, and the request repeated.
        """
        url = f"{self.base_url}/fdsnws/dataselect/1/queryauth"
        auth = self._auth
        try:
            return await self._get(session, url, params, middlewares=(auth,))
        except _UnauthorizedError:
            async with self._auth_lock:
                if self._auth is auth:
                    logger.warning(
                        "Credentials at %s were rejected, authenticating again.",
                        self.base_url,
                    )
                    await self.authenticate(session)
            return await self._get(session, url, params, middlewares=(self._auth,))

    async def get_waveforms(
        self, session: aiohttp.ClientSession, request: WaveformRequest
    ) -> Stream | None:
        params = {
            "network": request.network,
            "station": request.station,
            "location": request.location or "*",
            "channel": request.channel or "*",
            "starttime": _format_time(request.starttime),
            "endtime": _format_time(request.endtime),
        }
        if self._auth is not None:
            body = await self._get_authenticated(session, params)
        else:
            url = f"{self.base_url}/fdsnws/dataselect/1/query"
            body = await self._get(session, url, params)
        if body is None:
            return None
        return await asyncio.to_thread(obspy.read, io.BytesIO(body), format="MSEED")


class MultiClient:
    """
    Async client for several data centers (:py:class:`Source`), e.g., FDSN web
    services and Hi-net. It routes each network to the source serving it.

    Use it as an async context manager. It holds one HTTP session for all sources.
    Networks are mapped to sources by :py:meth:`get_inventory`.
    """

    def __init__(self, sources: list[Source]):
        self.sources = sources
        self._network: dict[str, Source] = {}
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

    def get_source(self, network: str) -> Source:
        try:
            return self._network[network]
        except KeyError:
            raise KeyError(
                f"No source known for network {network}. "
                "Call get_inventory() first to discover networks."
            ) from None

    async def get_waveforms(self, request: WaveformRequest) -> Stream:
        source = self.get_source(request.network)
        logger.debug("Fetching waveforms from %s", source)
        stream = await source.get_waveforms(self.session, request)
        if stream is None:
            raise FDSNNoDataException(
                f"{source} has no data for {request.network}.{request.station} "
                f"between {request.starttime} and {request.endtime}."
            )
        return stream

    def plan(self, requests: Iterable[WaveformRequest]) -> None:
        """
        Announces waveform requests to the sources serving them, see
        :py:meth:`Source.plan`. Requests for unknown networks are ignored.
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
        stations = {stations} if isinstance(stations, str) else set(stations)
        if networks is not None:
            networks = set(networks)

        inventories = await asyncio.gather(
            *(
                source.get_stations(
                    self.session, stations, starttime, endtime, level, networks
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
    A Hi-net request: whole blocks of several stations, as UTC timestamps.
    """

    starttime: float
    endtime: float
    stations: frozenset[str]


class HinetSource(Source):
    """
    NIED Hi-net (`doi:10.17598/NIED.0003 <https://doi.org/10.17598/NIED.0003>`_), via
    `HinetPy <https://github.com/seisman/HinetPy>`_. Requires an account, HinetPy and
    the Hi-net win32tools (``catwin32`` and ``win2sac_32``) on the ``PATH``.

    Downloads are cached in files of :py:attr:`block_length` minutes per station, in
    ``cache_dir``. A request is rounded out to whole blocks and only the blocks that
    are not cached yet are downloaded, so datasets with other windows or sampling
    rates reuse the downloads. With requests announced by :py:meth:`plan`, a download
    also covers the adjacent planned blocks of the station, e.g., the noise before an
    event, and all stations planned in these blocks.

    The account's station selection is changed by every request, so the source has
    a single connection and the requests run one at a time. Don't run two downloads
    with the same account at the same time.

    Stations are labeled ``<network>.<station>..EH<Z|N|E>`` (short-period, 100 Hz).
    The data are velocity as extracted by ``win2sac_32``, not counts.

    :param user: Hi-net user name
    :param password: Hi-net password
    :param network: Network code of the stations
    :param cache_dir: Folder of the cached downloads, by default ``hinet`` in the
        SeisBench cache
    :param block_length: Length of the cached blocks in minutes
    :param max_span: Maximum length of a download in minutes, when it is extended by
        adjacent planned blocks
    :param retries: Attempts per request, for dropped connections
    """

    code = "0101"

    def __init__(
        self,
        user: str,
        password: str,
        network: str = "N",
        cache_dir: Path | str | None = None,
        block_length: int = 5,
        max_span: int = 30,
        retries: int = 3,
    ):
        super().__init__(connections=1, networks=[network])
        self.network = network
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else seisbench.cache_root / "hinet"
        )
        self.block_length = block_length
        self.max_span = max_span
        self.retries = retries
        self._user = user
        self._password = password
        self._client = None
        self._stations = None
        # Block index -> stations
        self._planned: dict[int, set[str]] = defaultdict(set)

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
        stations: set[str],
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
        network = Network(code=self.network)
        for hinet_station in self._stations:
            code = hinet_station.name.split(".")[-1]
            if code not in stations or hinet_station.latitude is None:
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
        Records the blocks the stations are requested in.
        """
        for request in requests:
            for block in self._blocks(request.starttime, request.endtime):
                self._planned[block].add(request.station)

    def _blocks(self, starttime: UTCDateTime, endtime: UTCDateTime) -> range:
        """
        Indices of the blocks covering the time window, including its end sample.
        """
        length = self.block_length * 60
        return range(
            int(UTCDateTime(starttime).timestamp // length),
            int(UTCDateTime(endtime).timestamp // length) + 1,
        )

    def _cache_file(self, station: str, block: int) -> Path:
        start = UTCDateTime(block * self.block_length * 60)
        name = f"{self.network}.{station}.{start.strftime('%Y%m%dT%H%M')}.mseed"
        return self.cache_dir / self.network / station / name

    def _is_cached(self, station: str, block: int) -> bool:
        return self._cache_file(station, block).is_file()

    def _download_request(self, station: str, blocks: range) -> HinetRequest:
        """
        The blocks of a request, extended by the adjacent blocks planned for the
        station that are not cached, up to :py:attr:`max_span`. The request is for
        the station and all stations planned in these blocks and not cached.
        """
        first, last = blocks[0], blocks[-1]
        max_blocks = max(self.max_span // self.block_length, len(blocks))

        def is_missing(block: int) -> bool:
            return station in self._planned.get(block, ()) and not self._is_cached(
                station, block
            )

        while last - first + 1 < max_blocks and is_missing(first - 1):
            first -= 1
        while last - first + 1 < max_blocks and is_missing(last + 1):
            last += 1

        stations = {station}
        for block in range(first, last + 1):
            stations |= {
                other
                for other in self._planned.get(block, ())
                if not self._is_cached(other, block)
            }
        length = self.block_length * 60
        return HinetRequest(first * length, (last + 1) * length, frozenset(stations))

    async def get_waveforms(
        self, session: aiohttp.ClientSession, request: WaveformRequest
    ) -> Stream | None:
        starttime, endtime = (
            UTCDateTime(request.starttime),
            UTCDateTime(request.endtime),
        )
        blocks = self._blocks(starttime, endtime)
        # The single connection runs one request at a time, so no locking is needed
        if not all(self._is_cached(request.station, block) for block in blocks):
            await asyncio.to_thread(
                self._fetch, self._download_request(request.station, blocks)
            )
        stream = await asyncio.to_thread(self._read_cache, request.station, blocks)

        selected = Stream(
            [
                trace
                for trace in stream
                if fnmatch(trace.stats.channel, request.channel or "*")
            ]
        )
        selected = selected.slice(starttime, endtime)
        return selected if len(selected) else None

    def _read_cache(self, station: str, blocks: range) -> Stream:
        stream = Stream()
        for block in blocks:
            path = self._cache_file(station, block)
            if path.stat().st_size:  # Empty files: no data
                stream += obspy.read(str(path), format="MSEED")
        return stream.merge(method=-1)  # Joins contiguous blocks

    def _fetch(self, request: HinetRequest) -> None:
        """
        Downloads a request, with retries for dropped connections, and writes it to
        the cache. Runs in a thread.
        """
        for attempt in range(1, self.retries + 1):
            try:
                stream = self._fetch_once(request)
                break
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
        self._write_cache(request, stream)

    def _write_cache(self, request: HinetRequest, stream: Stream) -> None:
        """
        Writes one file per station and block of the request. Stations without
        data get an empty file, so they are not requested again.
        """
        length = self.block_length * 60
        for station in request.stations:
            traces = stream.select(station=station)
            for block in range(
                int(request.starttime // length), int(request.endtime // length)
            ):
                start = UTCDateTime(block * length)
                part = Stream(
                    [
                        trace.slice(start, start + length - trace.stats.delta / 2)
                        for trace in traces
                    ]
                )
                part.traces = [trace for trace in part if trace.stats.npts]
                path = self._cache_file(station, block)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".partial")
                if len(part):
                    part.write(str(tmp), format="MSEED")
                else:
                    tmp.write_bytes(b"")
                tmp.replace(path)

    def _fetch_once(self, request: HinetRequest) -> Stream:
        HinetPy = _import_hinetpy()
        names = [f"{self.network}.{station}" for station in sorted(request.stations)]
        starttime = UTCDateTime(request.starttime)
        span = round((request.endtime - request.starttime) / 60)
        logger.debug(
            "Requesting %d Hi-net stations, %s +%d min", len(names), starttime, span
        )

        with tempfile.TemporaryDirectory() as tmp:
            self._client.select_stations(self.code, names)
            data, ctable = self._client.get_continuous_waveform(
                self.code,
                (starttime + JST_OFFSET).datetime,
                span,
                outdir=tmp,
                threads=1,
            )
            if data is None:
                # HinetPy logs the reason, e.g., a failed request
                raise ConnectionError("Hi-net returned no data.")
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
    jobs: Iterable[tuple[Source, Job]],
    download: Callable[[Job], Awaitable[Result]],
    write: Callable[[Job, Result | None, Exception | None], None],
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
    queues: dict[Source, asyncio.Queue] = defaultdict(asyncio.Queue)
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
            if (i_job + 1) % 1000 == 0 or i_job + 1 == n_jobs:
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


def resample(stream: Stream, sampling_rate: float) -> None:
    """
    Resamples the traces in place with a polyphase filter (Kaiser window). Unlike
    ObsPy's FFT resampling, the anti-alias filter doesn't depend on the trace length.
    The first sample keeps its time. The mean is removed before and added back after,
    so that the offset of raw counts causes no transients at the trace edges.
    """
    for trace in stream:
        if trace.stats.sampling_rate == sampling_rate:
            continue
        ratio = Fraction(sampling_rate / trace.stats.sampling_rate).limit_denominator(
            1000
        )
        logger.debug(
            "Resampling trace %s from %.2f Hz to %.2f Hz.",
            trace.id,
            trace.stats.sampling_rate,
            sampling_rate,
        )
        mean = trace.data.mean()
        trace.data = (
            signal.resample_poly(
                trace.data - mean,
                ratio.numerator,
                ratio.denominator,
                window=("kaiser", 10),
            )
            + mean
        )
        trace.stats.sampling_rate = sampling_rate


async def homogenize_sampling_rate(st: Stream, sampling_rate: float) -> float:
    """
    Homogenizes the sampling rate of the stream to the given sampling rate, see
    :py:func:`resample`. Resamples in a thread.
    """
    if sampling_rate <= 0.0:
        sampling_rate = st[0].stats.sampling_rate
    if any(trace.stats.sampling_rate != sampling_rate for trace in st):
        await asyncio.to_thread(resample, st, sampling_rate)
    return sampling_rate


async def get_waveform_array(
    client: MultiClient,
    inventory: Inventory,
    request: WaveformRequest,
    sampling_rate: float,
    component_order: str,
) -> tuple[str, TraceParameters, np.ndarray]:
    """
    Downloads a waveform window, rotates it to ``component_order``, resamples it and
    converts it to an array.

    :return: SEED id of the first downloaded trace, the trace parameters describing
        the array, and the array
    """
    starttime, endtime = request.starttime, request.endtime
    stream = await client.get_waveforms(request)
    if not len(stream):
        raise ValueError(f"No waveforms found for {request}.")
    stream = select_location(stream, request.location)
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
