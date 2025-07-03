from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from obspy import Inventory
from obspy.clients.fdsn import Client

from seisbench.data.base import BenchmarkDataset
from seisbench.util import download_http

NETWORK_MAP = {
    "WB": "Geofon",
    "CZ": "Geofon",
    "SX": "BGR",
}

CATALOG_URL = "https://opara.zih.tu-dresden.de/bitstreams/94e4ab28-ae8e-4495-b102-53d2d28fe138/download"


class BohemiaSaxony(BenchmarkDataset):
    """
    Regional benchmark dataset of waveform data and metadata
    for the Bohemia and Saxony region in Germany/Czech Republic.

    """

    _client_cache: dict[str, Client] = {}
    _inventory: None | Inventory = None

    def __init__(self):
        citation = (
            "The data is compiled from data of the networks from Saxony network (SX),"
            " Webnet (WB)\n"
            "Catalog and Picks: https://doi.org/10.25532/OPARA-771\n"
            "Seismic Networks:\n"
            "WB - https://doi.org/10.7914/SN/WB\n"
            "SX - https://doi.org/10.7914/SN/SX\n"
        )

        super().__init__(citation=citation)

    def get_client(self, network: str) -> Client:
        cache_key = NETWORK_MAP.get(network)
        if cache_key is None:
            raise ValueError(f"Unknown network {network}")
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = Client(cache_key)
        return self._client_cache[cache_key]

    def _download_catalog(self, path: Path = Path.cwd()) -> Path:
        with NamedTemporaryFile(suffix=".zip", delete=False, dir=path) as temp_file:
            download_http(
                CATALOG_URL,
                temp_file.name,
                desc="Downloading Bohemia Saxony catalog",
            )
            with ZipFile(temp_file.name, "r") as zip_file:
                zip_file.extractall(path)
        return path

    def load_inventory(self) -> Inventory:
        path = self._download_catalog()

    def get_inventory(self):
        if self._inventory is None:
            self._inventory = self.load_inventory()
        return self._inventory
