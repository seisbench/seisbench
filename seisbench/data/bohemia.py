from __future__ import annotations

import asyncio
import random
import string
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Literal, cast
from urllib.parse import urljoin
from zipfile import ZipFile

import numpy as np
from obspy import Inventory, Stream, UTCDateTime, read_inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from obspy.core.event.base import WaveformStreamID
from obspy.core.event.catalog import Catalog, Event
from obspy.core.event.origin import Pick
from obspy.geodetics import gps2dist_azimuth
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
RNG = np.random.default_rng(31337)

PICK_WHITELIST = {
    "Pg",
    "Sg",
}

WEBNET_SWITCH_SH_CH_CHANNEL = datetime(2015, 5, 26)

STATION_NETWORK_CACHE: dict[str, str] = {}


ASCII_SET = string.ascii_lowercase + string.digits


class BohemiaSaxony(BenchmarkDataset):
    """
    Regional benchmark dataset of waveform data and metadata
    for the North-West Bohemia and Saxony region in Germany/Czech Republic.

    .. warning ::

        This dataset contains restricted data from the West Bohemia Local Seismic Network (WEBNET).
        To compile the full dataset, you will need to provide an EIDA token.
        Please see the `WEBNET site <https://doi.org/10.7914/SN/WB>`_ for more information.

    """

    _client: MultiClient

    def __init__(self, eida_token: str | None = None, **kwargs):
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

        super().__init__(
            citation=citation,
            license="CC0-1.0",
            repository_lookup=False,
            **kwargs,
        )

    def _init_client(self):
        self._client = MultiClient()
        self._client.add_client(Client("BGR"))
        self._client.add_client(Client("GEOFON", eida_token=self._eida_token))
        self._client.add_client(Client("LMU"))

    def _download_catalog_colm(self, path: Path = Path.cwd()) -> Path:
        files = list((self.path / "final2").glob("cll_*.txt"))
        if files:
            logger.debug("Catalog files already exist, skipping download.")
            return path
        logger.info("Downloading Bohemia Saxony catalog files.")
        with NamedTemporaryFile(suffix=".zip", delete=False, dir=path) as temp_file:
            catalog_url = urljoin(seisbench.remote_root, "auxiliary/collm-catalog.zip")
            download_http(
                catalog_url,
                temp_file.name,
                desc="Downloading Bohemia Saxony catalog",
            )
            with ZipFile(temp_file.name, "r") as zip_file:
                zip_file.extractall(path)
        # Fixup
        bad_file = path / "final2" / "cll_2019.txt"
        bad_file.write_text(bad_file.read_text().replace("GRZ!", "GRZ1"))

        return path

    def get_inventory(
        self,
        catalog: Catalog,
        force_download: bool = False,
    ) -> Inventory:
        inventory_file = self.path / "inventory.xml"
        if not inventory_file.exists() or force_download:
            stations = get_stations(catalog)
            starttime, endtime = get_catalog_timerange(catalog)
            inv = self._client.get_inventory(
                stations=stations
                # starttime=starttime,
                # endtime=endtime,
            )
            inv.write(str(inventory_file), format="STATIONXML")

        else:
            logger.info("Loading inventory from %s", inventory_file)
            inv: Inventory = read_inventory(str(inventory_file))

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
        event: Event,
        picks: list[Pick],
        inventory: Inventory,
        sampling_rate: float = 100.0,
        time_before: float = 60.0,
        time_after: float = 60.0,
    ) -> tuple[EventParameters, TraceParameters, np.ndarray]:
        waveform_id = picks[0].waveform_id

        event_params = get_event_params(event)
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
            f"{event_params['source_id']}_{'.'.join(waveform_id.get_seed_string())}"
        )

        for pick in picks:
            sample = (pick.time - actual_t_start) * sampling_rate
            if np.isnan(sample):
                print("############ NaN sample for pick")
                print("sample", sample)
                print("sampling_rate", sampling_rate)
                print("pick.time", pick.time)
                print("actual_t_start", actual_t_start)
            trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = int(sample)
            trace_params[f"trace_{pick.phase_hint}_status"] = pick.evaluation_mode
            if pick.polarity is None:
                trace_params[f"trace_{pick.phase_hint}_polarity"] = "undecidable"
            else:
                trace_params[f"trace_{pick.phase_hint}_polarity"] = pick.polarity

        return event_params, trace_params, data

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
        inv = self.get_inventory(cat, force_download=True)

        cat = fixup_catalog(cat, inv)

        async def download_event_data(
            event_work: list,
            timeout: float | None = 60.0,
            split: Literal["train", "dev", "test"] = "train",
        ) -> int:
            n_stations = 0
            n_work = len(event_work)
            try:
                for i_result, result in enumerate(
                    asyncio.as_completed(event_work, timeout=timeout)
                ):
                    try:
                        event_params, trace_params, data = await result
                        event_params["split"] = split
                    except Exception as exc:
                        logger.warning(
                            "Error processing event %s: %s",
                            event.short_str(),
                            exc,
                        )
                        continue
                    await asyncio.to_thread(
                        writer.add_trace,
                        {**event_params, **trace_params},
                        data,
                    )
                    logger.debug(
                        "Processed station %s (%d/%d)",
                        event.short_str(),
                        i_result + 1,
                        n_work,
                    )
                    n_stations += 1
            except asyncio.TimeoutError:
                logger.error(
                    "Timeout while downloading event data for %s.", event.resource_id
                )
            return n_stations

        n_stations_total = 0
        n_events = len(cat)
        logger.info("Downloading data for %d events.", n_events)

        for i_event, event in enumerate(cat):
            event_work = []
            for _, pick_group in groupby(
                event.picks,
                key=lambda p: (
                    p.waveform_id.network_code,
                    p.waveform_id.station_code,
                    p.waveform_id.location_code,
                ),
            ):
                work = self.get_station_waveform_data(
                    event=event,
                    picks=list(pick_group),
                    inventory=inv,
                    time_before=time_before,
                    time_after=time_after,
                )
                event_work.append(work)

            n_stations = asyncio.run(download_event_data(event_work, split=get_split()))
            n_stations_total += n_stations
            logger.info(
                "Downloaded %d waveform examples. %d stations for event %s (%d/%d)",
                n_stations_total,
                n_stations,
                event.short_str(),
                i_event + 1,
                n_events,
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
                stations.add(station)

    return stations


class MultiClient:
    _clients: list[Client]
    _network: dict[str, Client]
    _executor_pool: ThreadPoolExecutor

    def __init__(self):
        self._clients: list[Client] = []
        self._network: dict[str, Client] = {}
        self._executor_pool = ThreadPoolExecutor(max_workers=8)

    def add_client(self, client: Client):
        self._clients.append(client)

    async def get_waveforms(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: datetime = datetime.min,
        endtime: datetime = datetime.max,
    ) -> Stream:
        if not self._network:
            raise ValueError(
                "No networks discovered. "
                "Please call get_inventory() first to discover networks."
            )
        client = self._network[network]

        logger.debug("Fetching waveforms from client %s", client.base_url)
        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(
            self._executor_pool,
            partial(
                client.get_waveforms,
                network=network,
                station=station,
                location=location or "*",
                channel=channel or "*",
                starttime=UTCDateTime(starttime),
                endtime=UTCDateTime(endtime),
            ),
        )
        return cast(Stream, stream)

    def get_inventory(
        self,
        stations: str | Iterable[str],
        starttime: datetime = datetime.min,
        endtime: datetime = datetime.max,
        level: Literal["response", "station", "channel"] = "channel",
    ) -> Inventory:
        inventory = Inventory()

        async def get_inventory(client: Client):
            logger.info("Fetching inventory from client %s", client.base_url)
            inv = await asyncio.to_thread(
                client.get_stations,
                network="*",
                station=",".join(stations)
                if isinstance(stations, Iterable)
                else stations,
                location="*",
                channel="*",
                level=level,
                starttime=UTCDateTime(starttime),
                endtime=UTCDateTime(endtime),
            )
            for network in inv:
                self._network[network.code] = client

            inventory.extend(inv)

        async def worker():
            tasks = [get_inventory(client) for client in self._clients]
            return await asyncio.gather(*tasks)

        asyncio.run(worker())

        return inventory


def get_event_params(event: Event):
    origin = event.preferred_origin()
    magnitude = event.preferred_magnitude()

    sb_id = str(event.resource_id).split("/")[-1]
    if sb_id == "1":
        sb_id = "sb_id_" + "".join(random.choice(ASCII_SET) for _ in range(6))

    if origin is None:
        raise ValueError("Event has no preferred origin.")

    params: EventParameters = EventParameters(
        split="train",
        source_id=sb_id,
        source_origin_time=str(origin.time),
        source_origin_uncertainty_sec=origin.time_errors["uncertainty"],
        source_latitude_deg=origin.latitude,
        source_latitude_uncertainty_km=origin.latitude_errors["uncertainty"],
        source_longitude_deg=origin.longitude,
        source_longitude_uncertainty_km=origin.longitude_errors["uncertainty"],
        source_depth_km=origin.depth / 1e3,
        source_depth_uncertainty_km=origin.depth_errors["uncertainty"] / 1e3,
    )
    if magnitude is not None:
        params["source_magnitude"] = magnitude.mag
        params["source_magnitude_uncertainty"] = magnitude.mag_errors["uncertainty"]
        params["source_magnitude_type"] = magnitude.magnitude_type
        params["source_magnitude_author"] = magnitude.creation_info.agency_id

    return params


def get_trace_params(
    waveform_id: WaveformStreamID,
    inventory: Inventory,
    event_params: EventParameters,
) -> TraceParameters:
    try:
        coordinates = inventory.get_coordinates(waveform_id.get_seed_string())
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
    for event in catalog.events.copy():
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
                    waveform_id.station_code, inventory
                )
            except KeyError:
                event.picks.remove(pick)
                continue

            if waveform_id.network_code == "WB":
                origin_time = event.preferred_origin().time.datetime
                if origin_time < WEBNET_SWITCH_SH_CH_CHANNEL:
                    waveform_id.channel_code = "SH%s" % waveform_id.channel_code[-1]
                else:
                    waveform_id.channel_code = "CH%s" % waveform_id.channel_code[-1]

            if waveform_id.channel_code in REMOVE_CHANNELS:
                event.picks.remove(pick)

        if not event.picks:
            logger.info("Removing event %s with no picks.", event.resource_id)
            catalog.events.remove(event)

    for station, network in STATION_NETWORK_CACHE.items():
        if not network:
            logger.warning(
                "Station %s has no network code.",
                station,
            )
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


def get_network_code(station_code: str, inventory: Inventory) -> str:
    if station_code in STATION_NETWORK_CACHE:
        return STATION_NETWORK_CACHE[station_code]

    for network in inventory:
        for station in network:
            if station.code == station_code:
                STATION_NETWORK_CACHE[station_code] = network.code
                return network.code
    raise KeyError(f"Unknown network code for station {station_code}.")


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
    logger.root.setLevel("INFO")
    if args.eida_token is not None:
        path = Path(args.eida_token)
        if not path.exists():
            parser.error(f"EIDA token file {path} does not exist.")
        args.eida_token = str(path.expanduser().resolve())

    ds = BohemiaSaxony(eida_token=args.eida_token, output_order="ZNE")
