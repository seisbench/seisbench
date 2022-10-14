from seisbench.data.base import BenchmarkDataset
import seisbench.util

class OBS(BenchmarkDataset):
    """
    OBS Benchmark Dataset of local events

    Default component order is 'Z12H'. You can easily omit one component like, e.g., hydrophone by explicitly passing
    parameter 'component_order="Z12"'. This way, the dataset can be input to land station pickers that use only 3
    components.
    """

    def __init__(self, **kwargs):

        citation = (
            "OBS dataset by Bornstein et al."
        )
        if 'component_order' not in kwargs:
            kwargs['component_order'] = 'Z12H'

        super().__init__(citation=citation, **kwargs)

    def _download_dataset(self, writer, **kwargs):
        path = self.path
        self.path.mkdir(parents=True, exist_ok=True)
        seisbench.logger.warning(
            f"Downloading files to SeisBench cache {path}. This might take a while."
        )

        files = {
            'chunks': "https://nextcloud.gfz-potsdam.de/s/m2HxZc73eY6B7q9",
            'metadata000000.csv': "https://nextcloud.gfz-potsdam.de/s/KbqFs6wEyggRKcj",
            'metadata201805.csv': "https://nextcloud.gfz-potsdam.de/s/TATinXZxSd4d95S",
            'metadata201806.csv': "https://nextcloud.gfz-potsdam.de/s/wMCy99osYNJBfXc",
            'metadata201807.csv': "https://nextcloud.gfz-potsdam.de/s/dKti3P6YkJ9iZbg",
            'metadata201808.csv': "https://nextcloud.gfz-potsdam.de/s/QW8dHLQH7CJbQWi",
            'metadata201809.csv': "https://nextcloud.gfz-potsdam.de/s/wbxnrtTBKFJ6Z6d",
            'metadata201810.csv': "https://nextcloud.gfz-potsdam.de/s/PYRPrWFw5xtRGmk",
            'metadata201811.csv': "https://nextcloud.gfz-potsdam.de/s/dZaTcnHydJ4j7Z5",
            'metadata201812.csv': "https://nextcloud.gfz-potsdam.de/s/ZxXBySawqyfmRS6",
            'metadata201901.csv': "https://nextcloud.gfz-potsdam.de/s/rybGmnsAGKKX7Ey",
            'metadata201902.csv': "https://nextcloud.gfz-potsdam.de/s/54xPmS8f43fZkbH",
            'metadata201903.csv': "https://nextcloud.gfz-potsdam.de/s/wjjM5RdDCG2g2kL",
            'metadata201904.csv': "https://nextcloud.gfz-potsdam.de/s/JtfmWatwCcE4PnC",
            'metadata201905.csv': "https://nextcloud.gfz-potsdam.de/s/kzR97rtrwpzQGjm",
            'metadata201906.csv': "https://nextcloud.gfz-potsdam.de/s/Hi3t5M5aNKMG8BC",
            'metadata201907.csv': "https://nextcloud.gfz-potsdam.de/s/Lw7ctKoHggboKME",
            'metadata201908.csv': "https://nextcloud.gfz-potsdam.de/s/9cGNdSpcweJ3Sn2",
            'waveforms000000.csv': "https://nextcloud.gfz-potsdam.de/s/xyxPmRCxtRsHew8",
            'waveforms201805.csv': "https://nextcloud.gfz-potsdam.de/s/d7jZxMnRfEapc6K",
            'waveforms201806.csv': "https://nextcloud.gfz-potsdam.de/s/AEbkK4g5GjKyxAP",
            'waveforms201807.csv': "https://nextcloud.gfz-potsdam.de/s/2cnFsLtGzyGde4S",
            'waveforms201808.csv': "https://nextcloud.gfz-potsdam.de/s/dq22nes6PYacY8z",
            'waveforms201809.csv': "https://nextcloud.gfz-potsdam.de/s/A2LTGDEqKpTZcBP",
            'waveforms201810.csv': "https://nextcloud.gfz-potsdam.de/s/rqxac9GiWL4koYH",
            'waveforms201811.csv': "https://nextcloud.gfz-potsdam.de/s/Ys3tfx7LjTJXy6y",
            'waveforms201812.csv': "https://nextcloud.gfz-potsdam.de/s/3nD7LHL4RB3ff4B",
            'waveforms201901.csv': "https://nextcloud.gfz-potsdam.de/s/MZqmjYd6BdzaEpc",
            'waveforms201902.csv': "https://nextcloud.gfz-potsdam.de/s/bDWPwmp2xtDfNDy",
            'waveforms201903.csv': "https://nextcloud.gfz-potsdam.de/s/84zpanXFEBCbjYE",
            'waveforms201904.csv': "https://nextcloud.gfz-potsdam.de/s/zs2npZgNfEA9e3F",
            'waveforms201905.csv': "https://nextcloud.gfz-potsdam.de/s/jKLGLngyHcBiTCz",
            'waveforms201906.csv': "https://nextcloud.gfz-potsdam.de/s/FEEK8wWtiPGWQaS",
            'waveforms201907.csv': "https://nextcloud.gfz-potsdam.de/s/x2Dtbo2pFSpJCoT",
            'waveforms201908.csv': "https://nextcloud.gfz-potsdam.de/s/MYkPRQCkgWM2Y5S",
        }
        for filename, link in files.items():
            seisbench.util.download_http(
                link+"/download", path / filename, desc=f"Downloading {filename}"
            )

