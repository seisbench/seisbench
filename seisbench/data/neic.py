import seisbench
import seisbench.util
from .base import BenchmarkDataset

import json
import tarfile
import numpy as np
from collections import defaultdict
import shutil


# Conversion from earth radius
DEG2KM = 2 * np.pi * 6371 / 360


class NEIC(BenchmarkDataset):
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
                file.extractall(path_unpacked)

        groups = [("P", "Train"), ("S", "Train"), ("P", "Test"), ("S", "Test")]

        total_samples = (
            np.load(path_unpacked / f"PAzi_Train.npy").shape[0]
            + np.load(path_unpacked / f"SAzi_Train.npy").shape[0]
            + np.load(path_unpacked / f"PAzi_Test.npy").shape[0]
            + np.load(path_unpacked / f"SAzi_Test.npy").shape[0]
        )

        # Select 10 percent of the training events for development
        # As the train test split is random, but event wise, a similar strategy is employed here
        train_ids = np.concatenate(
            [
                np.load(path_unpacked / f"PEID_Train.npy"),
                np.load(path_unpacked / f"SEID_Train.npy"),
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
