from __future__ import annotations

import asyncio
import io
import random
import string
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, cast
from urllib.parse import urljoin
from zipfile import ZipFile

import numpy as np
import obspy
from obspy import Inventory, Stream, UTCDateTime, read_inventory
from obspy.clients.fdsn.header import (
    URL_MAPPINGS,
    FDSNException,
    FDSNNoDataException,
)
from obspy.core.event.base import QuantityError, WaveformStreamID
from obspy.core.event.catalog import Catalog, Event
from obspy.core.event.origin import Pick
from obspy.geodetics import degrees2kilometers, gps2dist_azimuth
from obspy.io.nordic.core import read_nordic

import seisbench
from seisbench import logger
from seisbench.data.base import BenchmarkDataset, EventParameters, TraceParameters
from seisbench.util import download_http
from seisbench.util.trace_ops import (
    rotate_stream_to_zne,
    stream_to_array,
    trace_has_spikes,
)

if TYPE_CHECKING:
    import aiohttp

REMOVE_CHANNELS = {
    "HHT",  # Stray channel or typo
    "BHN",  # Low sampling rate channel BH
    "BHE",
    "BHZ",
    "SHN",  # Low sampling rate channel SH
    "SHE",
    "SHZ",
}

STATION_MAPPING = {
    "LACW": "LAC",
    "STCW": "STC",
    "VACW": "VAC",
    "KVCW": "KVC",
    "KRCW": "KRC",
    "ZEIT": "ZEITZ",
    "LEIB": "LEIB1",
}

COMPONENT_ORDER = "ZNE"

# Networks the catalog picks come from (see the catalog README), used to resolve
# station codes that exist in several networks
PREFERRED_NETWORKS = ("SX", "WB", "TH", "BW", "CZ", "GR")
RNG = np.random.default_rng(31337)

PICK_WHITELIST = {
    "Pg",
    "Sg",
}

WEBNET_SWITCH_SH_CH_CHANNEL = datetime(2015, 5, 26)


ASCII_SET = string.ascii_lowercase + string.digits
SOURCE_ID_RNG = random.Random(31337)


class BohemiaSaxony(BenchmarkDataset):
    """
    Regional benchmark dataset of waveform data and metadata
    for the North-West Bohemia and Saxony region in Germany/Czech Republic.

    .. warning ::

        This dataset contains restricted data from the West Bohemia Local Seismic Network (WEBNET).
        To compile the full dataset, you will need to provide an EIDA token.
        Please see the `WEBNET site <https://doi.org/10.7914/SN/WB>`_ for more information.

    .. note ::

        The dataset is compiled from FDSN web services on first use, which requires
        ``aiohttp``. Install it with ``pip install seisbench[download]``.

    :param eida_token: EIDA token or path to a token file, for the restricted WB data
    :param connections: Number of parallel connections per FDSN data center
    """

    _client: MultiClient

    def __init__(
        self,
        eida_token: str | None = None,
        connections: int = 4,
        **kwargs,
    ):
        citation = """
The data is compiled from data of the networks from Saxony Seismic Network (SX),
West Bohemia Local Seismic Network (WEBNET; WB), Thuringia Seismological Network (TH),
and Czech Regional Seismic Network (CZ) and is provided by the BGR (Bundesanstalt für
Geowissenschaften und Rohstoffe) and GFZ GEOFON.

The dataset includes restricted data from the WB network, which requires an EIDA token
for access. Please provide the path to your EIDA token file when initializing the
dataset via the `eida_token` argument.

Catalog and Picks:
* Earthquakes in Saxony (Germany) and surroundings
  from 2006 to 2024 -- onsets and locations,
  https://opara.zih.tu-dresden.de/items/5387886f-25f2-4faf-8dca-33981d898ab9

Seismic Networks:
* SXNET Saxon Seismic Network, https://doi.org/10.7914/SN/SX
* West Bohemia Local Seismic Network (WEBNET), https://doi.org/10.7914/SN/WB
* Thüringer Seismologisches Netz, https://doi.org/10.7914/SN/TH
* Czech Regional Seismic Network, https://doi.org/10.7914/SN/CZ
"""
        self._eida_token = eida_token
        self._connections = connections

        super().__init__(
            citation=citation,
            license="CC0-1.0",
            repository_lookup=False,
            **kwargs,
        )

    def _init_client(self):
        self._client = MultiClient(
            [
                FDSNSource(URL_MAPPINGS["BGR"], connections=self._connections),
                FDSNSource(
                    URL_MAPPINGS["GEOFON"],
                    connections=self._connections,
                    eida_token=self._eida_token,
                ),
                FDSNSource(URL_MAPPINGS["LMU"], connections=self._connections),
            ]
        )

    def _download_catalog_colm(self, path: Path = Path.cwd()) -> Path:
        files = list((self.path / "final2").glob("cll_*.txt"))
        if files:
            logger.debug("Catalog files already exist, skipping download.")
            return path
        logger.info("Downloading Bohemia Saxony catalog files.")
        catalog_url = urljoin(seisbench.remote_root, "auxiliary/collm-catalog.zip")
        catalog_zip = path / "collm-catalog.zip"
        try:
            download_http(
                catalog_url,
                catalog_zip,
                desc="Downloading Bohemia Saxony catalog",
            )
            with ZipFile(catalog_zip, "r") as zip_file:
                zip_file.extractall(path)
        finally:
            catalog_zip.unlink(missing_ok=True)
        # Fixup
        bad_file = path / "final2" / "cll_2019.txt"
        bad_file.write_text(bad_file.read_text().replace("GRZ!", "GRZ1"))

        return path

    async def get_inventory(
        self,
        catalog: Catalog,
        force_download: bool = False,
    ) -> Inventory:
        """
        Downloads the inventory of all stations in the catalog, or loads it from
        the dataset folder. Must be called within the client's context.
        """
        inventory_file = self.path / "inventory.xml"
        stations = get_stations(catalog)
        if not inventory_file.exists() or force_download:
            starttime, endtime = get_catalog_timerange(catalog)
            inv = await self._client.get_inventory(
                stations=stations
                # starttime=starttime,
                # endtime=endtime,
            )
            await asyncio.to_thread(inv.write, str(inventory_file), format="STATIONXML")

        else:
            logger.info("Loading inventory from %s", inventory_file)
            inv = await asyncio.to_thread(read_inventory, str(inventory_file))
            # The file doesn't record which source serves which network
            await self._client.get_inventory(stations=stations, level="network")

        return inv

    def get_catalog(self) -> Catalog:
        self._download_catalog_colm(self.path)
        nordic_files = sorted((self.path / "final2").glob("cll_*.txt"))

        catalog = Catalog()
        for file in nordic_files:
            logger.info(f"Reading NORDIC file {file}")
            catalog += read_nordic(str(file))

        logger.info("Loaded catalog with %d events.", len(catalog))
        return catalog

    async def get_station_waveform_data(
        self,
        event_params: EventParameters,
        picks: list[Pick],
        inventory: Inventory,
        sampling_rate: float = 100.0,
        time_before: float = 60.0,
        time_after: float = 60.0,
    ) -> tuple[TraceParameters, np.ndarray]:
        waveform_id = picks[0].waveform_id

        trace_params = get_trace_params(
            waveform_id,
            inventory,
            event_params,
        )

        tmin = min(p.time for p in picks) - time_before
        tmax = max(p.time for p in picks) + time_after

        try:
            stream = await self._client.get_waveforms(
                network=waveform_id.network_code,
                station=waveform_id.station_code,
                location=waveform_id.location_code,
                channel=waveform_id.channel_code[:2] + "*",
                starttime=tmin.datetime,
                endtime=tmax.datetime,
            )
        except FDSNException as exc:
            logger.error(
                "Error fetching waveforms for %s", waveform_id.get_seed_string()
            )
            raise exc

        if not len(stream):
            raise ValueError(
                f"No waveforms found for {waveform_id} in time range {tmin} - {tmax}.",
            )

        rotate_stream_to_zne(stream, inventory=inventory)
        sampling_rate = await homogenize_sampling_rate(stream, sampling_rate)

        stream = stream.slice(tmin, tmax)
        actual_t_start, data, completeness = stream_to_array(
            stream,
            component_order=COMPONENT_ORDER,
        )
        desired_samples = int((tmax - tmin) * sampling_rate) + 1
        if desired_samples > data.shape[1]:
            # Traces appear to be complete, but do not cover the intended time range
            completeness *= data.shape[1] / desired_samples

        trace_params["trace_sampling_rate_hz"] = sampling_rate
        trace_params["trace_completeness"] = completeness
        trace_params["trace_has_spikes"] = trace_has_spikes(data)
        trace_params["trace_start_time"] = str(actual_t_start)
        trace_params["trace_component_order"] = COMPONENT_ORDER

        trace_params["trace_name"] = (
            f"{event_params['source_id']}_{waveform_id.network_code}."
            f"{waveform_id.station_code}.{waveform_id.location_code or ''}."
            f"{waveform_id.channel_code[:2]}"
        )

        for pick in picks:
            sample = (pick.time - actual_t_start) * sampling_rate
            trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = int(sample)
            trace_params[f"trace_{pick.phase_hint}_status"] = pick.evaluation_mode
            if pick.polarity is None:
                trace_params[f"trace_{pick.phase_hint}_polarity"] = "undecidable"
            else:
                trace_params[f"trace_{pick.phase_hint}_polarity"] = pick.polarity

        return trace_params, data

    def _download_dataset(
        self,
        writer,
        time_before: float = 60.0,
        time_after: float = 60.0,
        **kwargs,
    ):
        logger.info(
            "No pre-downloaded dataset available, downloading from BGR and Geofon. "
            "This may take a while.",
        )
        if not self._eida_token:
            logger.warning(
                "No EIDA token provided. "
                "Restricted WB network data will not be accessible.",
            )
        self._init_client()
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": COMPONENT_ORDER,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        cat = self.get_catalog()
        asyncio.run(
            self._download_waveforms(
                writer, cat, time_before=time_before, time_after=time_after
            )
        )

    async def _download_waveforms(
        self,
        writer,
        catalog: Catalog,
        time_before: float,
        time_after: float,
    ):
        """
        Downloads the waveforms of all stations in the catalog. Each source has its
        own queue, worked on by as many tasks as the source has connections. A single
        task writes the traces, so the writer is never used concurrently.
        """
        async with self._client as client:
            inv = await self.get_inventory(catalog, force_download=True)
            catalog = fixup_catalog(catalog, inv)

            queues = {source: asyncio.Queue() for source in client.sources}
            n_stations = 0
            for event in catalog:
                split = get_split()
                try:
                    event_params = get_event_params(event)
                except ValueError as exc:
                    logger.warning("Skipping event %s: %s", event.short_str(), exc)
                    continue
                event_params["split"] = split

                # groupby only groups adjacent picks, so sort them by station first
                picks = sorted(event.picks, key=_station_key)
                for _, pick_group in groupby(picks, key=_station_key):
                    pick_group = list(pick_group)
                    network = pick_group[0].waveform_id.network_code
                    try:
                        source = client.get_source(network)
                    except KeyError as exc:
                        logger.warning("Skipping station: %s", exc)
                        continue
                    queues[source].put_nowait((event, event_params, pick_group))
                    n_stations += 1

            logger.info(
                "Downloading %d stations of %d events (%s).",
                n_stations,
                len(catalog),
                ", ".join(
                    f"{queue.qsize()} from {source.base_url}"
                    for source, queue in queues.items()
                ),
            )
            results = asyncio.Queue(maxsize=64)

            async def download(queue: asyncio.Queue):
                while True:
                    try:
                        event, event_params, picks = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        result = await self.get_station_waveform_data(
                            event_params=event_params,
                            picks=picks,
                            inventory=inv,
                            time_before=time_before,
                            time_after=time_after,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Error processing station %s of event %s: %s",
                            picks[0].waveform_id.get_seed_string(),
                            event.short_str(),
                            exc,
                        )
                        result = None
                    await results.put((event_params, result))

            async def write():
                n_written = 0
                for i_station in range(n_stations):
                    event_params, result = await results.get()
                    if result is not None:
                        trace_params, data = result
                        await asyncio.to_thread(
                            writer.add_trace,
                            {**event_params, **trace_params},
                            data,
                        )
                        n_written += 1
                    if (i_station + 1) % 1000 == 0 or i_station + 1 == n_stations:
                        logger.info(
                            "Processed %d/%d stations, wrote %d traces.",
                            i_station + 1,
                            n_stations,
                            n_written,
                        )

            await asyncio.gather(
                write(),
                *(
                    download(queue)
                    for source, queue in queues.items()
                    for _ in range(source.connections)
                ),
            )


def _station_key(pick: Pick) -> tuple[str, str, str]:
    waveform_id = pick.waveform_id
    return (
        waveform_id.network_code or "",
        waveform_id.station_code or "",
        waveform_id.location_code or "",
    )


def get_catalog_timerange(catalog: Catalog) -> tuple[datetime, datetime]:
    """
    Get the time range of the catalog.

    :param catalog: Catalog to get the time range from
    :return: Start and end time of the catalog
    """
    tmin = datetime.max
    tmax = datetime.min

    for ev in catalog:
        ev = cast(Event, ev)
        origin = ev.preferred_origin()
        tmin = min(tmin, origin.time.datetime)
        tmax = max(tmax, origin.time.datetime)

    return tmin, tmax


def get_stations(catalog: Catalog) -> set[str]:
    """
    Get the networks and stations from the catalog.

    :param catalog: Catalog to get the networks and stations from
    :return: Dictionary with networks as keys and sets of stations as values
    """
    stations = set()

    for ev in catalog:
        ev = cast(Event, ev)
        for pick in ev.picks:
            if pick.waveform_id is not None:
                station = pick.waveform_id.station_code
                # Request the FDSN station names that fixup_catalog maps to
                stations.add(STATION_MAPPING.get(station, station))

    return stations


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


def _uncertainty(errors: QuantityError | None) -> float:
    if errors is None or errors.uncertainty is None:
        return np.nan
    return errors.uncertainty


def get_event_params(event: Event) -> EventParameters:
    origin = event.preferred_origin()
    magnitude = event.preferred_magnitude()

    sb_id = str(event.resource_id).split("/")[-1]
    if sb_id == "1":
        sb_id = "sb_id_" + "".join(SOURCE_ID_RNG.choice(ASCII_SET) for _ in range(6))

    if origin is None:
        raise ValueError("Event has no preferred origin.")

    # The Nordic reader stores the horizontal errors in degrees
    latitude_uncertainty_km = degrees2kilometers(_uncertainty(origin.latitude_errors))
    longitude_uncertainty_km = degrees2kilometers(
        _uncertainty(origin.longitude_errors)
    ) * np.cos(np.radians(origin.latitude))

    params: EventParameters = EventParameters(
        split="train",
        source_id=sb_id,
        source_origin_time=str(origin.time),
        source_origin_uncertainty_sec=_uncertainty(origin.time_errors),
        source_latitude_deg=origin.latitude,
        source_latitude_uncertainty_km=latitude_uncertainty_km,
        source_longitude_deg=origin.longitude,
        source_longitude_uncertainty_km=longitude_uncertainty_km,
        source_depth_km=np.nan if origin.depth is None else origin.depth / 1e3,
        source_depth_uncertainty_km=_uncertainty(origin.depth_errors) / 1e3,
    )
    if magnitude is not None:
        params["source_magnitude"] = magnitude.mag
        params["source_magnitude_uncertainty"] = _uncertainty(magnitude.mag_errors)
        params["source_magnitude_type"] = magnitude.magnitude_type
        if magnitude.creation_info is not None:
            params["source_magnitude_author"] = magnitude.creation_info.agency_id

    return params


def get_trace_params(
    waveform_id: WaveformStreamID,
    inventory: Inventory,
    event_params: EventParameters,
) -> TraceParameters:
    try:
        coordinates = inventory.get_coordinates(
            waveform_id.get_seed_string(),
            datetime=UTCDateTime(event_params["source_origin_time"]),
        )
    except Exception as r:
        raise ValueError(
            f"Could not get coordinates for {waveform_id.get_seed_string()}: {r}"
        ) from r

    if not np.isnan(coordinates["latitude"] * coordinates["longitude"]):
        back_azimuth = gps2dist_azimuth(
            event_params["source_latitude_deg"],
            event_params["source_longitude_deg"],
            coordinates["latitude"],
            coordinates["longitude"],
        )[2]
    else:
        back_azimuth = np.nan

    trace_params = TraceParameters(
        trace_name="foo",
        path_back_azimuth_deg=back_azimuth,
        station_network_code=waveform_id.network_code,
        station_code=waveform_id.station_code,
        trace_channel=waveform_id.channel_code[:2],
        station_location_code=waveform_id.location_code,
        station_latitude_deg=coordinates["latitude"],
        station_longitude_deg=coordinates["longitude"],
        station_elevation_m=coordinates["elevation"],
    )

    return trace_params


def fixup_catalog(catalog: Catalog, inventory: Inventory) -> Catalog:
    """
    Fixes the channel codes of the picks.
    """
    station_index = get_station_index(inventory)
    for event in catalog.events.copy():
        origin = event.preferred_origin()
        if origin is None:
            logger.info("Removing event %s without origin.", event.resource_id)
            catalog.events.remove(event)
            continue
        origin_time = origin.time
        for pick in event.picks.copy():
            if pick.phase_hint not in PICK_WHITELIST:
                event.picks.remove(pick)
                continue

            waveform_id = pick.waveform_id

            try:
                waveform_id.channel_code = "%sH%s" % tuple(waveform_id.channel_code)
            except TypeError:
                # This happens if the channel code is already in the correct format.
                event.picks.remove(pick)
                continue

            # Fix station names to FDSN
            waveform_id.station_code = STATION_MAPPING.get(
                waveform_id.station_code,
                waveform_id.station_code,
            )

            try:
                waveform_id.network_code = get_network_code(
                    waveform_id.station_code, station_index, origin_time
                )
            except KeyError:
                event.picks.remove(pick)
                continue

            if waveform_id.network_code == "WB":
                if origin_time.datetime < WEBNET_SWITCH_SH_CH_CHANNEL:
                    waveform_id.channel_code = "SH%s" % waveform_id.channel_code[-1]
                else:
                    waveform_id.channel_code = "CH%s" % waveform_id.channel_code[-1]

            if waveform_id.channel_code in REMOVE_CHANNELS:
                event.picks.remove(pick)

        if not event.picks:
            logger.info("Removing event %s with no picks.", event.resource_id)
            catalog.events.remove(event)

    return catalog


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


StationIndex = dict[str, list[tuple[str, UTCDateTime | None, UTCDateTime | None]]]


def get_station_index(inventory: Inventory) -> StationIndex:
    """
    Maps each station code to the networks and epochs it appears in.
    """
    index: StationIndex = defaultdict(list)
    for network in inventory:
        for station in network:
            index[station.code].append(
                (network.code, station.start_date, station.end_date)
            )
    return index


def is_temporary_network(network_code: str) -> bool:
    # FDSN convention: temporary network codes start with a digit or X, Y, Z
    return network_code[0].isdigit() or network_code[0] in "XYZ"


def get_network_code(
    station_code: str,
    station_index: StationIndex,
    time: UTCDateTime | None = None,
) -> str:
    """
    Resolves the network of a station code. Some codes exist in several networks,
    e.g., LAC in WB and ZB. Only networks where the station operates at the given
    time are considered. Of those, the dataset's networks are preferred, then other
    permanent networks, then temporary ones.
    """
    epochs = station_index.get(station_code)
    if not epochs:
        raise KeyError(f"Unknown network code for station {station_code}.")

    networks = {
        network
        for network, start, end in epochs
        if time is None
        or ((start is None or start <= time) and (end is None or time <= end))
    }
    if not networks:
        raise KeyError(f"Station {station_code} is not operating at {time}.")

    def priority(network: str) -> tuple[int, int, str]:
        if network in PREFERRED_NETWORKS:
            return (0, PREFERRED_NETWORKS.index(network), network)
        return (1 + is_temporary_network(network), 0, network)

    return min(networks, key=priority)


def get_split(test: float = 0.1, dev: float = 0.1) -> Literal["train", "dev", "test"]:
    """
    Returns a random split for the dataset.
    """
    r = RNG.random()
    train = 1.0 - test - dev
    if r < train:
        return "train"
    elif r < train + dev:
        return "dev"
    else:
        return "test"


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--eida-token",
        type=str,
        default=None,
        help="Path to EIDA token file.",
    )
    args = parser.parse_args()
    logger.setLevel("INFO")
    if args.eida_token is not None:
        path = Path(args.eida_token).expanduser().resolve()
        if not path.is_file():
            parser.error(f"EIDA token file {path} does not exist.")
        args.eida_token = str(path)

    ds = BohemiaSaxony(eida_token=args.eida_token, component_order="ZNE")
    print(ds)
