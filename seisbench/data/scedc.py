import seisbench
import seisbench.util
from .base import BenchmarkDataset

import h5py
from collections import defaultdict
import numpy as np
import shutil


class SCEDC(BenchmarkDataset):
    """
    SCEDC waveform archive.
    With entire catalog > 500Gb, utilizes SeisBench chunk
    reading to keep operations manageable.
    """

    def __init__(self, **kwargs):
        citation = (
            "SCEDC (2013): Southern California Earthquake Center."
            "doi:10.7909/C3WD3xH1"
        )
        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer, chunk, basepath=None, **kwargs):
        download_instructions = "SCEDC dataset currently unavailable for download at this point in dev. process."

        basepath = self.path

        if not basepath.exists():
            raise ValueError(
                "No cached version of SCEDC found. " + download_instructions
            )

        chunks_path = basepath.path / "chunks"
        if not chunks_path.is_file():
            basepath.path.mkdir(exist_ok=True, parents=True)
            with open(chunks_path, "w") as f:
                f.write("\n".join(["_b{:02d}".format(i) for i in range(2)]))


# TODO: Check with Zach Ross if this dataset really only differs from Ross2018JGRPick through the class rebalancing.
#       If so, it this should be stated in the SeisBench documentation and probably also be reflected in the naming.
class Ross2018JGRFM(BenchmarkDataset):
    """
    First motion polarity dataset belonging to the publication:
    Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). P wave arrival picking and first‐motion polarity determination
    with deep learning. Journal of Geophysical Research: Solid Earth, 123, 5120– 5129.
    https://doi.org/10.1029/2017JB015251

    Note that this dataset contains picks as well.
    .. warning::
        This dataset only contains traces for the Z component.
        It therefore ignores the default SeisBench the component_order.
    """

    def __init__(self, **kwargs):
        citation = (
            "Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). "
            "P wave arrival picking and first‐motion polarity determination with deep learning. "
            "Journal of Geophysical Research: Solid Earth, 123, 5120– 5129. https://doi.org/10.1029/2017JB015251"
        )
        super().__init__(
            citation=citation, repository_lookup=False, component_order="Z", **kwargs
        )

    def _download_dataset(self, writer, cleanup=False, blocksize=2 ** 14):
        """
        Downloads and converts the dataset from the original publication

        :param writer: WaveformWriter
        :param cleanup: If true, delete the original files after conversion. Defaults to false.
        :param blocksize: Number of waveform samples to read from disk at once.
        :return:
        """

        path = self.path
        path_original = path / "original"
        path_original.mkdir(parents=True, exist_ok=True)

        # Maps ids to strings for the polarity
        polarity_list = ["up", "down", "unknown"]

        # Download data files
        data_urls = [
            "https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5",
            "https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5",
        ]

        for f in data_urls:
            # Uses callback_if_uncached only to be able to utilize the cache mechanism
            # Concurrent accesses are anyhow already controlled by the callback_if_uncached call wrapping _download_dataset
            # It's therefore considered save to set force=True
            filename = f[f.rfind("/") + 1 :]

            def callback_download_original(path):
                seisbench.util.download_http(
                    f,
                    path,
                    desc=f"Downloading file {filename}",
                )

            seisbench.util.callback_if_uncached(
                path_original / filename, callback_download_original, force=True
            )

        with h5py.File(
            path_original / "scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5", "r"
        ) as f_train:
            train_samples = f_train["X"].shape[0]
        with h5py.File(
            path_original / "scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5", "r"
        ) as f_test:
            test_samples = f_test["X"].shape[0]

        writer.blocksize = 2 ** 16
        writer.set_total(train_samples + test_samples)
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "Z",
            "measurement": "velocity",
            "sampling_rate": 100,
            "unit": "none/normalized",
            "instrument_response": "not restituted",
        }

        eq_counts = defaultdict(lambda: 0)

        for split in ["train", "test"]:
            with h5py.File(
                path_original / f"scsn_p_2000_2017_6sec_0.5r_fm_{split}.hdf5", "r"
            ) as f:
                # Preload all small arrays to avoid disk seeks
                y = f["Y"][:]
                dist = f["dist"][:]
                evids = f["evids"][:]
                mag = f["mag"][:]
                sncls = f["sncls"][:]
                snr = f["snr"][:]

                # Use 10 percent of the training events as development set
                if split == "train":
                    dev_ids = set(np.unique(evids)[::10])
                else:
                    dev_ids = set()

                wf_block = None
                for i in range(f["X"].shape[0]):
                    # Preload block of waveforms
                    if i % blocksize == 0:
                        wf_block = f["X"][i : i + blocksize]
                    wf = wf_block[i % blocksize].reshape(
                        1, -1
                    )  # Load waveforms and add (virtual) channel axis

                    eid = f"{evids[i]}_{sncls[i].decode()}"
                    trace_station_id = eq_counts[eid]
                    eq_counts[eid] += 1
                    trace_name = f"{eid}_{trace_station_id}"

                    if evids[i] in dev_ids:
                        trace_split = "dev"
                    else:
                        trace_split = split

                    net, sta, cha = sncls[i].decode().split(".")
                    polarity = polarity_list[y[i]]

                    metadata = {
                        "trace_name": trace_name,
                        "trace_category": "earthquake",
                        "trace_p_arrival_sample": 300,
                        "trace_p_status": "manual",
                        "trace_snr_db": snr[i],
                        "trace_channel": cha,
                        "trace_polarity": polarity,
                        "station_network_code": net,
                        "station_code": sta,
                        "source_magnitude": mag[i],
                        "source_id": evids[i],
                        "path_ep_distance_km": dist[i],
                        "split": trace_split,
                    }

                    writer.add_trace(metadata, wf)

        if cleanup:
            shutil.rmtree(path_original)


class Ross2018JGRPick(BenchmarkDataset):
    """
    Pick dataset belonging to the publication:
    Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). P wave arrival picking and first‐motion polarity determination
    with deep learning. Journal of Geophysical Research: Solid Earth, 123, 5120– 5129.
    https://doi.org/10.1029/2017JB015251

    Note that this dataset contains polarities as well.
    .. warning::
        This dataset only contains traces for the Z component.
        It therefore ignores the default SeisBench the component_order.
    """

    def __init__(self, **kwargs):
        citation = (
            "Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). "
            "P wave arrival picking and first‐motion polarity determination with deep learning. "
            "Journal of Geophysical Research: Solid Earth, 123, 5120– 5129. https://doi.org/10.1029/2017JB015251"
        )
        super().__init__(
            citation=citation, repository_lookup=False, component_order="Z", **kwargs
        )

    def _download_dataset(self, writer, cleanup=False, blocksize=2 ** 14):
        """
        Downloads and converts the dataset from the original publication

        :param writer: WaveformWriter
        :param cleanup: If true, delete the original files after conversion. Defaults to false.
        :param blocksize: Number of waveform samples to read from disk at once
        :return:
        """

        path = self.path
        path_original = path / "original"
        path_original.mkdir(parents=True, exist_ok=True)

        # Maps ids to strings for the polarity
        polarity_list = ["up", "down", "unknown"]

        # Download data files
        data_urls = [
            "https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5",
            "https://service.scedc.caltech.edu/ftp/Ross_FinalTrainedModels/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5",
        ]

        for f in data_urls:
            # Uses callback_if_uncached only to be able to utilize the cache mechanism
            # Concurrent accesses are anyhow already controlled by the callback_if_uncached call wrapping _download_dataset
            # It's therefore considered save to set force=True
            filename = f[f.rfind("/") + 1 :]

            def callback_download_original(path):
                seisbench.util.download_http(
                    f,
                    path,
                    desc=f"Downloading file {filename}",
                )

            seisbench.util.callback_if_uncached(
                path_original / filename, callback_download_original, force=True
            )

        with h5py.File(
            path_original / "scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5", "r"
        ) as f_train:
            train_samples = f_train["X"].shape[0]
        with h5py.File(
            path_original / "scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5", "r"
        ) as f_test:
            test_samples = f_test["X"].shape[0]

        writer.blocksize = 2 ** 16
        writer.set_total(train_samples + test_samples)
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "Z",
            "measurement": "velocity",
            "sampling_rate": 100,
            "unit": "none/normalized",
            "instrument_response": "not restituted",
        }

        eq_counts = defaultdict(lambda: 0)

        for split in ["train", "test"]:
            with h5py.File(
                path_original / f"scsn_p_2000_2017_6sec_0.5r_pick_{split}.hdf5", "r"
            ) as f:
                # Preload all small arrays to avoid disk seeks
                fm = f["fm"][:]
                dist = f["dist"][:]
                evids = f["evids"][:]
                mag = f["mag"][:]
                sncls = f["sncls"][:]
                snr = f["snr"][:]

                # Use 10 percent of the training events as development set
                if split == "train":
                    dev_ids = set(np.unique(evids)[::10])
                else:
                    dev_ids = set()

                wf_block = None
                for i in range(f["X"].shape[0]):
                    # Preload block of waveforms
                    if i % blocksize == 0:
                        wf_block = f["X"][i : i + blocksize]
                    wf = wf_block[i % blocksize].reshape(
                        1, -1
                    )  # Load waveforms and add (virtual) channel axis

                    eid = f"{evids[i]}_{sncls[i].decode()}"
                    trace_station_id = eq_counts[eid]
                    eq_counts[eid] += 1
                    trace_name = f"{eid}_{trace_station_id}"

                    if evids[i] in dev_ids:
                        trace_split = "dev"
                    else:
                        trace_split = split

                    net, sta, cha = sncls[i].decode().split(".")
                    polarity = polarity_list[fm[i]]

                    metadata = {
                        "trace_name": trace_name,
                        "trace_category": "earthquake",
                        "trace_p_arrival_sample": 300,
                        "trace_p_status": "manual",
                        "trace_snr_db": snr[i],
                        "trace_channel": cha,
                        "trace_polarity": polarity,
                        "station_network_code": net,
                        "station_code": sta,
                        "source_magnitude": mag[i],
                        "source_id": evids[i],
                        "path_ep_distance_km": dist[i],
                        "split": trace_split,
                    }

                    writer.add_trace(metadata, wf)

        if cleanup:
            shutil.rmtree(path_original)
