"""
Helpers to download station metadata and waveforms from FDSN web services.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import obspy
from obspy import Inventory, Stream, UTCDateTime, read_inventory
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

from seisbench import logger

if TYPE_CHECKING:
    import aiohttp


def _import_aiohttp():
    try:
        import aiohttp
    except ImportError as exc:
        raise ImportError(
            "Compiling the BohemiaSaxony dataset from FDSN web services requires "
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
    """

    def __init__(
        self,
        base_url: str,
        connections: int = 4,
        eida_token: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.connections = connections
        self.eida_token = eida_token
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
    ) -> Inventory:
        logger.info("Fetching inventory from %s", self.base_url)
        params = {
            "network": "*",
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
    ) -> Inventory:
        """
        Fetches the inventory from all sources and maps each network to the first
        source, in the order of :py:attr:`sources`, that serves it.
        """
        if isinstance(stations, str):
            stations = [stations]
        station_list = ",".join(stations)

        inventories = await asyncio.gather(
            *(
                source.get_stations(
                    self.session, station_list, starttime, endtime, level
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
