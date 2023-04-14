import seisbench.util
from seisbench.data.base import BenchmarkDataset


class OBS(BenchmarkDataset):
    """
    OBS Benchmark Dataset of local events

    Default component order is 'Z12H'. You can easily omit one component like, e.g., hydrophone by explicitly passing
    parameter 'component_order="Z12"'. This way, the dataset can be input to land station pickers that use only 3
    components.
    """

    def __init__(self, component_order="Z12H", **kwargs):
        citation = (
            "Bornstein, T., Lange, D., Münchmeyer, J., Woollam, J., Rietbrock, A., Barcheck, G., "
            "Grevemeyer, I., Tilmann, F. (2023). PickBlue: Seismic phase picking for ocean bottom "
            "seismometers with deep learning. arxiv preprint. http://arxiv.org/abs/2304.06635"
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
            "201805",
            "201806",
            "201807",
            "201808",
            "201809",
            "201810",
            "201811",
            "201812",
            "201901",
            "201902",
            "201903",
            "201904",
            "201905",
            "201906",
            "201907",
            "201908",
            "000000",
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
