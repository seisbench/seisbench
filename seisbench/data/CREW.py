import seisbench.util
from seisbench.data.base import BenchmarkDataset


class CREW(BenchmarkDataset):
    """
    Curated Regional Earthquake Waveforms (CREW dataset)

    """

    def __init__(self, component_order="ENZ", **kwargs):
        citation = (
        "Aguilar Suarez, A. L., & Beroza, G. (2024). "
        "Curated Regional Earthquake Waveforms (CREW) Dataset. "
        "Seismica, 3(1). https://doi.org/10.26443/seismica.v3i1.1049"
        )

        self._write_chunk_file()

        super().__init__(
            citation=citation,
            repository_lookup=True,
            component_order=component_order,
            **kwargs,
        )

    @staticmethod
    def available_chunks(*args, **kwargs):
        return [
            "000",
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "008",
            "009",
            "010",
            "011",
            "012",
            "013",
            "014",
            "015",
            "016",
            "017",
            "018",
            "019",
            "020",
            "021",
            "022",
            "023",
            "024",
            "025",
            "026",
            "027",
            "028",
            "029",
            "030",
            "031",
            "032",
            "033",
            "034",
            "035",
            "036",
            "037",
            "038",
            "039",
            "040",
            "041",
            "042",
            "043",
            "044",
            "045",
            "046",
            "047",
            "048",
            "049",
            "050",
            "052",
            "053",
            "054",
            "055",
            "056",
            "057",
            "058",
            "059",
            "060",
            "061",
            "062",
            "063",
            "064",
            "065",
            "066",
            "067",
            "068",
            "069",
            "070",
            "071",
            "072",
            "073",
            "074",
            "075",
            "076",
            "077",
            "078",
            "079",
            "080",
            "081",
            "082",
            "083",
            "084",
            "085",
            "086",
            "087",
            "088",
            "089",
            "090",
            "091",
            "092",
            "093",
            "094",
            "095",
            "096",
            "097",
            "098",
            "099",
            "100",
            "101",
            "102",
            "103",
            "104",
            "105",
            "106",
            "107",
            "108",
            "109",
            "110",
            "111",
            "112",
            "113",
            "114",
            "115",
            "116",
            "117",
            "118",
            "119",
            "120",
            "121",
            "122",
            "123",
            "124",
            "125",
            "126",
            "127",
            "128",
            "129",
            "130",
            "131",
            "132",
            "133",
            "134",
            "135",
            "136",
            "137",
            "138",
            "139",
            "140",
            "141",
            "142",
            "143",
            "144",
            "145",
            "146",
            "147",
            "148",
            "149",
            "150",
            "151",
            "152",
            "153",
            "154",
            "155",
            "156",
        ]

    def _write_chunk_file(self):
        """
        Write out the chunk file

        :return: None
        """
        chunks_path = self.path / "chunks"

        if chunks_path.is_file():
            return

        chunks = self.available_chunks()
        chunks_str = "\n".join(chunks) + "\n"

        self.path.mkdir(exist_ok=True, parents=True)
        with open(chunks_path, "w") as f:
            f.write(chunks_str)

    def _download_dataset(self, writer, chunk, **kwargs):
        path = self.path
        path.mkdir(parents=True, exist_ok=True)

        files = {
            "chunks": "https://nextcloud.gfz-potsdam.de/s/m2HxZc73eY6B7q9",
            "metadata001.csv":
            "metadata002.csv":
            "metadata003.csv":
            "metadata004.csv":
            "metadata005.csv":
            "metadata006.csv":
            "metadata007.csv":
            "metadata008.csv":
            "metadata009.csv":
            "metadata010.csv":
            "waveforms001.hdf5":
            "waveforms002.hdf5":
            "waveforms003.hdf5":
            "waveforms004.hdf5":
            "waveforms005.hdf5":
            "waveforms006.hdf5":
            "waveforms007.hdf5":
            "waveforms008.hdf5":
            "waveforms009.hdf5":
            "waveforms010.hdf5":

            "metadata000000.csv": "https://nextcloud.gfz-potsdam.de/s/KbqFs6wEyggRKcj",
            "metadata201805.csv": "https://nextcloud.gfz-potsdam.de/s/TATinXZxSd4d95S",
            "metadata201806.csv": "https://nextcloud.gfz-potsdam.de/s/wMCy99osYNJBfXc",
            "metadata201807.csv": "https://nextcloud.gfz-potsdam.de/s/dKti3P6YkJ9iZbg",
            "metadata201808.csv": "https://nextcloud.gfz-potsdam.de/s/QW8dHLQH7CJbQWi",
            "metadata201809.csv": "https://nextcloud.gfz-potsdam.de/s/wbxnrtTBKFJ6Z6d",
            "metadata201810.csv": "https://nextcloud.gfz-potsdam.de/s/PYRPrWFw5xtRGmk",
            "metadata201811.csv": "https://nextcloud.gfz-potsdam.de/s/dZaTcnHydJ4j7Z5",
            "metadata201812.csv": "https://nextcloud.gfz-potsdam.de/s/ZxXBySawqyfmRS6",
            "metadata201901.csv": "https://nextcloud.gfz-potsdam.de/s/rybGmnsAGKKX7Ey",
            "metadata201902.csv": "https://nextcloud.gfz-potsdam.de/s/54xPmS8f43fZkbH",
            "metadata201903.csv": "https://nextcloud.gfz-potsdam.de/s/wjjM5RdDCG2g2kL",
            "metadata201904.csv": "https://nextcloud.gfz-potsdam.de/s/JtfmWatwCcE4PnC",
            "metadata201905.csv": "https://nextcloud.gfz-potsdam.de/s/kzR97rtrwpzQGjm",
            "metadata201906.csv": "https://nextcloud.gfz-potsdam.de/s/Hi3t5M5aNKMG8BC",
            "metadata201907.csv": "https://nextcloud.gfz-potsdam.de/s/Lw7ctKoHggboKME",
            "metadata201908.csv": "https://nextcloud.gfz-potsdam.de/s/9cGNdSpcweJ3Sn2",
            "waveforms000000.hdf5": "https://nextcloud.gfz-potsdam.de/s/xyxPmRCxtRsHew8",
            "waveforms201805.hdf5": "https://nextcloud.gfz-potsdam.de/s/d7jZxMnRfEapc6K",
            "waveforms201806.hdf5": "https://nextcloud.gfz-potsdam.de/s/AEbkK4g5GjKyxAP",
            "waveforms201807.hdf5": "https://nextcloud.gfz-potsdam.de/s/2cnFsLtGzyGde4S",
            "waveforms201808.hdf5": "https://nextcloud.gfz-potsdam.de/s/dq22nes6PYacY8z",
            "waveforms201809.hdf5": "https://nextcloud.gfz-potsdam.de/s/A2LTGDEqKpTZcBP",
            "waveforms201810.hdf5": "https://nextcloud.gfz-potsdam.de/s/rqxac9GiWL4koYH",
            "waveforms201811.hdf5": "https://nextcloud.gfz-potsdam.de/s/Ys3tfx7LjTJXy6y",
            "waveforms201812.hdf5": "https://nextcloud.gfz-potsdam.de/s/3nD7LHL4RB3ff4B",
            "waveforms201901.hdf5": "https://nextcloud.gfz-potsdam.de/s/MZqmjYd6BdzaEpc",
            "waveforms201902.hdf5": "https://nextcloud.gfz-potsdam.de/s/bDWPwmp2xtDfNDy",
            "waveforms201903.hdf5": "https://nextcloud.gfz-potsdam.de/s/84zpanXFEBCbjYE",
            "waveforms201904.hdf5": "https://nextcloud.gfz-potsdam.de/s/zs2npZgNfEA9e3F",
            "waveforms201905.hdf5": "https://nextcloud.gfz-potsdam.de/s/jKLGLngyHcBiTCz",
            "waveforms201906.hdf5": "https://nextcloud.gfz-potsdam.de/s/FEEK8wWtiPGWQaS",
            "waveforms201907.hdf5": "https://nextcloud.gfz-potsdam.de/s/x2Dtbo2pFSpJCoT",
            "waveforms201908.hdf5": "https://nextcloud.gfz-potsdam.de/s/6PSRaBEfAp3wTGg",
        }

        metadata_name = f"metadata{chunk}.csv"
        waveform_name = f"waveforms{chunk}.hdf5"
        seisbench.util.download_http(
            files[metadata_name] + "/download",
            path / writer.metadata_path,
            desc=f"Downloading {metadata_name}",
        )
        seisbench.util.download_http(
            files[waveform_name] + "/download",
            path / writer.waveforms_path,
            desc=f"Downloading {waveform_name}",
        )
