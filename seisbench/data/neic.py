import json
import shutil
import tarfile
from collections import defaultdict

import numpy as np

import seisbench
import seisbench.util

from .base import WaveformBenchmarkDataset

# Conversion from earth radius
DEG2KM = 2 * np.pi * 6371 / 360


class NEIC(WaveformBenchmarkDataset):
    """
    NEIC dataset from Yeck and Patton
    """

    def __init__(self, **kwargs):
        citation = (
            "Yeck, W.L., and Patton, J., 2020, Waveform Data and Metadata used to "
            "National Earthquake Information Center Deep-Learning Models: "
            "U.S. Geological Survey data release, https://doi.org/10.5066/P9OHF4WL."
        )
        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer, cleanup=True, blocksize=2**14):
        """
        Downloads and converts the dataset from the original publication

        :param writer: WaveformWriter
        :param cleanup: If true, delete the original and temporary data files after conversion. Defaults to true.
        :param blocksize: Number of waveform samples to read from disk at once
        :return:
        """
        seisbench.logger.warning(
            "Converting this catalog from source will require ~250 GB disk storage. "
            "The resulting catalog has ~75GB. "
            "Please ensure that the storage is available on your disk."
        )

        path = self.path
        path_original = path / "original"
        path_original.mkdir(parents=True, exist_ok=True)
        path_meta = path_original / "meta.json"

        # Download metadata in json format to extract download links for the data
        seisbench.util.download_http(
            "https://www.sciencebase.gov/catalog/item/5ed528ff82ce2832f047eee6?format=json",
            path_meta,
            progress_bar=False,
        )

        # Load metadata
        with open(path_meta, "r") as fmeta:
            meta = json.load(fmeta)

        # Download data files
        for f in meta["files"]:
            # Uses callback_if_uncached only to be able to utilize the cache mechanism.
            # Concurrent accesses are anyhow already controlled by the callback_if_uncached
            # call wrapping _download_dataset.
            # It's therefore considered save to set force=True.
            def callback_download_original(path):
                seisbench.util.download_http(
                    f["url"],
                    path,
                    desc=f"Downloading file {f['name']}",
                )

            seisbench.util.callback_if_uncached(
                path_original / f["name"], callback_download_original, force=True
            )

        # Note: The following lines could also each be wrapped into a callback_if_uncached
        #       However, concatenating and unpacking do not take too long and this way the code is easier.

        # Concatenate partitioned files
        seisbench.logger.warning(
            "Concatenating partitioned tar.gz archives. This might take a few minutes."
        )
        partitioned_files = ["PWF_Test.tar.gz", "PWF_Train.tar.gz", "SWF_Train.tar.gz"]
        for partitioned_file in partitioned_files:
            members = sorted(
                [
                    x
                    for x in path_original.iterdir()
                    if x.name.startswith(partitioned_file)
                    and not x.name == partitioned_file
                ]
            )
            with open(path_original / partitioned_file, "wb") as fout:
                for file in members:
                    with open(file, "rb") as fin:
                        data = fin.read(1000000)  # Read 1MB parts
                        while len(data) > 0:
                            fout.write(data)
                            data = fin.read(1000000)  # Read 1MB parts

        # Unpack files
        seisbench.logger.warning(
            "Unpacking tar.gz archives. This might take a few minutes."
        )
        path_unpacked = path / "unpacked"
        path_unpacked.mkdir(parents=True, exist_ok=True)

        tar_files = [x for x in path_original.iterdir() if x.name.endswith(".tar.gz")]
        for tar_path in tar_files:
            with tarfile.open(tar_path, "r:gz") as file:
                seisbench.util.safe_extract_tar(file, path_unpacked)

        groups = [("P", "Train"), ("S", "Train"), ("P", "Test"), ("S", "Test")]

        total_samples = (
            np.load(path_unpacked / "PAzi_Train.npy").shape[0]
            + np.load(path_unpacked / "SAzi_Train.npy").shape[0]
            + np.load(path_unpacked / "PAzi_Test.npy").shape[0]
            + np.load(path_unpacked / "SAzi_Test.npy").shape[0]
        )

        # Select 10 percent of the training events for development
        # As the train test split is random, but event wise, a similar strategy is employed here
        train_ids = np.concatenate(
            [
                np.load(path_unpacked / "PEID_Train.npy"),
                np.load(path_unpacked / "SEID_Train.npy"),
            ]
        )
        train_ids = np.unique(train_ids)
        dev_ids = set(train_ids[::10])

        writer.set_total(total_samples)

        # TODO: Verify that these are unrestituted counts
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "sampling_rate": 40,
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        eq_counts = defaultdict(lambda: 0)

        for wavetype, split in groups:
            # Flush cache after train/dev is complete
            if wavetype == "P" and split == "Test":
                writer.flush_hdf5()

            azimuth = np.load(path_unpacked / f"{wavetype}Azi_{split}.npy")
            distance = np.load(path_unpacked / f"{wavetype}Dist_{split}.npy")
            event_id = np.load(path_unpacked / f"{wavetype}EID_{split}.npy")
            magnitude = np.load(path_unpacked / f"{wavetype}Mag_{split}.npy")

            p = 0
            while p < azimuth.shape[0]:
                # Recreate memmap each epoch to avoid memory "leak"
                # For details see
                # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once
                waveforms = np.load(
                    path_unpacked / f"{wavetype}WF_{split}.npy", mmap_mode="r"
                )

                block_azimuth = azimuth[p : p + blocksize]
                block_distance = distance[p : p + blocksize]
                block_event_id = event_id[p : p + blocksize]
                block_magnitude = magnitude[p : p + blocksize]
                block_waveforms = waveforms[
                    p : p + blocksize
                ].copy()  # Copy causes the load into memory

                for azi, dist, eid, mag, wf in zip(
                    block_azimuth,
                    block_distance,
                    block_event_id,
                    block_magnitude,
                    block_waveforms,
                ):
                    trace_station_id = eq_counts[eid]
                    eq_counts[eid] += 1
                    trace_name = f"{eid}_st{trace_station_id}"

                    trace_split = split.lower()
                    if eid in dev_ids:
                        trace_split = "dev"

                    metadata = {
                        "trace_name": trace_name,
                        "trace_category": "earthquake",
                        f"trace_{wavetype.lower()}_arrival_sample": 1200,
                        f"trace_{wavetype.lower()}_status": "manual",
                        "source_magnitude": mag,
                        "source_id": eid,
                        "path_ep_distance_km": dist * DEG2KM,
                        "path_back_azimuth_deg": azi,
                        "split": trace_split,
                    }

                    writer.add_trace(metadata, wf)

                p += blocksize

        if cleanup:
            seisbench.logger.warning(
                "Cleaning up source and temporary files. This might take a few minutes."
            )
            shutil.rmtree(path_unpacked)
            shutil.rmtree(path_original)


class MLAAPDE(WaveformBenchmarkDataset):
    """
    MLAAPDE dataset from Cole et al. (2023)

    Note that the SeisBench version is not identical to the precompiled version
    distributed directly through USGS but uses a different data selection.
    In addition, custom versions of MLAAPDE can be compiled with the software
    provided by the original authors. These datasets can be exported in
    SeisBench format.
    """

    def __init__(self, **kwargs):
        citation = (
            "Cole, H. M., Yeck, W. L., & Benz, H. M. (2023). "
            "MLAAPDE: A Machine Learning Dataset for Determining "
            "Global Earthquake Source Parameters. "
            "Seismological Research Letters, 94(5), 2489-2499. "
            "https://doi.org/10.1785/0220230021"
            "\n\n"
            "Cole H. M. and W. L. Yeck, 2022, "
            "Global Earthquake Machine Learning Dataset: "
            "Machine Learning Asset Aggregation of the PDE (MLAAPDE): "
            "U.S. Geological Survey data release, doi:10.5066/P96FABIB"
        )
        license = "MLAAPDE code under CC0 1.0 Universal, data licenses dependent on the underlying networks"

        super().__init__(
            citation=citation, license=license, repository_lookup=True, **kwargs
        )

    def _download_dataset(self, writer, **kwargs):
        pass

    @staticmethod
    def available_chunks(*args, **kwargs):
        return [
            "_201307",
            "_201308",
            "_201309",
            "_201310",
            "_201311",
            "_201312",
            "_201401",
            "_201402",
            "_201403",
            "_201404",
            "_201405",
            "_201406",
            "_201407",
            "_201408",
            "_201409",
            "_201410",
            "_201411",
            "_201412",
            "_201501",
            "_201502",
            "_201503",
            "_201504",
            "_201505",
            "_201506",
            "_201507",
            "_201508",
            "_201509",
            "_201510",
            "_201511",
            "_201512",
            "_201601",
            "_201602",
            "_201603",
            "_201604",
            "_201605",
            "_201606",
            "_201607",
            "_201608",
            "_201609",
            "_201610",
            "_201611",
            "_201612",
            "_201701",
            "_201702",
            "_201703",
            "_201704",
            "_201705",
            "_201706",
            "_201707",
            "_201708",
            "_201709",
            "_201710",
            "_201711",
            "_201712",
            "_201901",
            "_201902",
            "_201903",
            "_201904",
            "_201905",
            "_201906",
            "_201907",
            "_201908",
            "_201909",
            "_201910",
            "_201911",
            "_201912",
            "_202001",
            "_202002",
            "_202003",
            "_202004",
            "_202005",
            "_202006",
            "_202007",
            "_202008",
            "_202009",
            "_202010",
            "_202011",
            "_202012",
            "_202101",
            "_202102",
            "_202103",
            "_202104",
            "_202105",
            "_202106",
            "_202107",
            "_202108",
            "_202109",
            "_202110",
            "_202111",
            "_202112",
            "_201801",
            "_201802",
            "_201803",
            "_201804",
            "_201805",
            "_201806",
            "_201807",
            "_201808",
            "_201809",
            "_201811",
            "_201810",
            "_201812",
            "_202201",
            "_202202",
            "_202203",
            "_202204",
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
