import seisbench
import seisbench.util
from .base import BenchmarkDataset

import h5py
from collections import defaultdict
import numpy as np
import shutil


class SCEDC(BenchmarkDataset):
    """
    SCEDC waveform archive (2000-2020).

    Splits are set using standard random sampling of :py:class: BenchmarkDataset.
    """

    def __init__(self, **kwargs):
        citation = (
            "SCEDC (2013): Southern California Earthquake Center."
            "https://doi.org/10.7909/C3WD3xH1"
        )

        seisbench.logger.warning(
            "Check available storage and memory before downloading and general use "
            "of SCEDC dataset. "
            "Dataset size: waveforms.hdf5 ~660Gb, metadata.csv ~2.2Gb"
        )

        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, **kwargs):
        # NOTE: SCEDC dataset is pre-compiled and stored in remote repository root for access
        pass


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

    def __init__(self, component_order="Z", **kwargs):
        citation = (
            "Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). "
            "P wave arrival picking and first‐motion polarity determination with deep learning. "
            "Journal of Geophysical Research: Solid Earth, 123, 5120– 5129. https://doi.org/10.1029/2017JB015251"
        )
        super().__init__(
            citation=citation,
            repository_lookup=False,
            component_order=component_order,
            **kwargs,
        )

    def _download_dataset(self, writer, cleanup=False, blocksize=2**14):
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
            # Uses callback_if_uncached only to be able to utilize the cache mechanism.
            # Concurrent accesses are anyhow already controlled
            # by the callback_if_uncached call wrapping _download_dataset.
            # It's therefore considered save to set force=True.
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

            # Write out all data from the current split
            writer.flush_hdf5()

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

    def __init__(self, component_order="Z", **kwargs):
        citation = (
            "Ross, Z. E., Meier, M.‐A., & Hauksson, E. (2018). "
            "P wave arrival picking and first‐motion polarity determination with deep learning. "
            "Journal of Geophysical Research: Solid Earth, 123, 5120– 5129. https://doi.org/10.1029/2017JB015251"
        )
        super().__init__(
            citation=citation,
            repository_lookup=False,
            component_order=component_order,
            **kwargs,
        )

    def _download_dataset(self, writer, cleanup=False, blocksize=2**14):
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
            # Uses callback_if_uncached only to be able to utilize the cache mechanism.
            # Concurrent accesses are anyhow already controlled
            # by the callback_if_uncached call wrapping _download_dataset.
            # It's therefore considered save to set force=True.
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

            # Write out all data from the current split
            writer.flush_hdf5()

        if cleanup:
            shutil.rmtree(path_original)


class Ross2018GPD(BenchmarkDataset):
    """
    Pick dataset belonging to the publication:
    Zachary E. Ross, Men‐Andrin Meier, Egill Hauksson, Thomas H. Heaton;
    Generalized Seismic Phase Detection with Deep Learning.
    Bulletin of the Seismological Society of America 2018;; 108 (5A): 2894–2901.
    https://doi.org/10.1785/0120180080
    """

    def __init__(self, **kwargs):
        citation = (
            "Ross, Z. E., Meier, M.‐A., Hauksson, E., & Heaton, T.(2018). "
            "Generalized Seismic Phase Detection with Deep Learning. "
            "Bulletin of the Seismological Society of America 2018;; 108 (5A): 2894–2901. "
            "https://doi.org/10.1785/0120180080"
        )
        super().__init__(citation=citation, repository_lookup=False, **kwargs)

    def _download_dataset(self, writer, cleanup=False, blocksize=2**14):
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

        # Download data files
        # Uses callback_if_uncached only to be able to utilize the cache mechanism
        # Concurrent accesses are anyhow already controlled by the callback_if_uncached call wrapping _download_dataset
        # It's therefore considered save to set force=True

        data_url = "https://service.scedc.caltech.edu/ftp/ross_etal_2018_bssa/scsn_ps_2000_2017_shuf.hdf5"
        filename = data_url[data_url.rfind("/") + 1 :]

        def callback_download_original(path):
            seisbench.util.download_http(
                data_url,
                path,
                desc=f"Downloading file {filename}",
            )

        seisbench.util.callback_if_uncached(
            path_original / filename, callback_download_original, force=True
        )

        writer.bucket_size = (
            4096  # Single waveforms are small so the bucket size should be larger
        )
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "sampling_rate": 100,
            "unit": "none/normalized",
            "instrument_response": "not restituted",
        }

        with h5py.File(path_original / filename, "r") as fin:
            writer.set_total(fin["X"].shape[0])
            y = fin["Y"][()]

            wf_block = None
            for i in range(fin["X"].shape[0]):
                # Preload block of waveforms
                if i % blocksize == 0:
                    wf_block = fin["X"][i : i + blocksize]
                wf = wf_block[i % blocksize].T  # Load waveforms and transpose
                wf = wf[[2, 0, 1]]  # Resort components to ZNE

                if i % 10 < 6:
                    trace_split = "train"
                elif i % 10 < 7:
                    trace_split = "dev"
                else:
                    trace_split = "test"

                metadata = {
                    "split": trace_split,
                }

                if y[i] == 0:
                    # P pick
                    metadata["trace_category"] = "earthquake"
                    metadata["trace_p_arrival_sample"] = 300
                    metadata["trace_p_status"] = "manual"
                elif y[i] == 1:
                    # S pick
                    metadata["trace_category"] = "earthquake"
                    metadata["trace_s_arrival_sample"] = 300
                    metadata["trace_s_status"] = "manual"
                else:
                    metadata["trace_category"] = "noise"

                writer.add_trace(metadata, wf)

        if cleanup:
            shutil.rmtree(path_original)


# TODO: Write Men-Andrin Meier regarding zero metadata columns, time format, split format
class Meier2019JGR(BenchmarkDataset):
    """
    Southern californian part of the dataset from Meier et al. (2019)
    Note that due to the missing Japanese data,
    there is a massive overrepresentation of noise samples.

    Meier, M.-A., Ross, Z. E., Ramachandran, A., Balakrishna, A.,
    Nair, S., Kundzicz, P., et al. (2019). Reliable real‐time
    seismic signal/noise discrimination with machine learning.
    Journal of Geophysical Research: Solid Earth, 124.
    https://doi.org/10.1029/2018JB016661
    """

    def __init__(self, **kwargs):
        citation = (
            "Meier, M.-A., Ross, Z. E., Ramachandran, A., Balakrishna, A., "
            "Nair, S., Kundzicz, P., et al. (2019). Reliable real‐time "
            "seismic signal/noise discrimination with machine learning. "
            "Journal of Geophysical Research: Solid Earth, 124. "
            "https://doi.org/10.1029/2018JB016661"
        )
        super().__init__(citation=citation, repository_lookup=False, **kwargs)

    def _download_dataset(self, writer, cleanup=False, blocksize=2**14):
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

        # Download data files
        # Uses callback_if_uncached only to be able to utilize the cache mechanism
        # Concurrent accesses are anyhow already controlled by the callback_if_uncached call wrapping _download_dataset
        # It's therefore considered save to set force=True

        data_url = "https://service.scedc.caltech.edu/ftp/meier_etal_2019_jgr/onsetWforms_meier19jgr_pub1_0_woJP.h5"
        filename = data_url[data_url.rfind("/") + 1 :]

        def callback_download_original(path):
            seisbench.util.download_http(
                data_url,
                path,
                desc=f"Downloading file {filename}",
            )

        seisbench.util.callback_if_uncached(
            path_original / filename, callback_download_original, force=True
        )

        writer.bucket_size = (
            4096  # Single waveforms are small so the bucket size should be larger
        )
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "sampling_rate": 100,
            "unit": "mps",
            "instrument_response": "gain corrected",
        }

        category_map = {
            "noise": "noise",
            "quake": "earthquake (local)",
            "tele": "earthquake (teleseismic)",
        }

        with h5py.File(path_original / filename, "r") as fin:
            total = (
                fin["quake/wforms"].shape[1]
                + fin["noise/wforms"].shape[1]
                + fin["tele/wforms"].shape[1]
            )
            writer.set_total(total)

            for group in "quake", "noise", "tele":
                gin = fin[group]
                meta_features = gin["numMeta"][()]
                category = category_map[group]

                wf_block = None
                for i in range(meta_features.shape[1]):
                    # Preload block of waveforms
                    if i % blocksize == 0:
                        wf_block = gin["wforms"][:, i : i + blocksize]
                    wf = wf_block[:, i % blocksize]  # Load waveforms
                    wf = wf[[2, 0, 1]]  # Resort components to ZNE

                    # TODO: Read/define split
                    meta_row = meta_features[:, i]

                    if group == "noise":
                        metadata = {
                            "trace_category": category,
                            # "split": trace_split,
                            "trace_snr_db": meta_row[3],
                            "trace_record_id": meta_row[4],
                            # meta_row[5] - pickIndex - ignored
                            # Data is consistently zero
                            # "station_latitude_deg": meta_row[6],
                            # "station_longitude_deg": meta_row[7],
                            # "trace_pga_mps2": meta_row[8],
                            # "trace_pgv_mps": meta_row[9],
                            # "trace_pgd_m": meta_row[10],
                            "source_origin_time": meta_row[11],  # Format unclear
                            # Data is consistently zero
                            # "path_back_azimuth_deg": meta_row[12]
                        }

                    else:
                        metadata = {
                            "trace_category": category,
                            # "split": trace_split,
                            "source_magnitude": meta_row[0],
                            "path_hyp_distance_km": meta_row[1],
                            "source_depth_km": meta_row[2],
                            "trace_snr_db": meta_row[3],
                            "trace_record_id": meta_row[4],
                            # meta_row[5] - pickIndex - ignored
                            "station_latitude_deg": meta_row[6],
                            "station_longitude_deg": meta_row[7],
                            # Data is consistently zero
                            # "trace_pga_mps2": meta_row[8],
                            # "trace_pgv_mps": meta_row[9],
                            # "trace_pgd_m": meta_row[10],
                            "source_origin_time": meta_row[11],  # Format unclear
                            # Data is consistently zero
                            # "path_back_azimuth_deg": meta_row[12]
                            "trace_p_arrival_sample": 201,
                            "trace_p_status": "manual",
                        }

                    writer.add_trace(metadata, wf)

        if cleanup:
            shutil.rmtree(path_original)
