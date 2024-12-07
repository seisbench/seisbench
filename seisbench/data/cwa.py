import seisbench
import seisbench.util
import seisbench.data as sbd

import h5py
import pandas as pd
import os
import tarfile

from seisbench.data.base import BenchmarkDataset, WaveformDataWriter
from huggingface_hub import hf_hub_download
from pathlib import Path


class CWA(BenchmarkDataset):
    def __init__(self, **kwargs):
        """
        subset      (String):           Specify the seismographic network (CWASN, TSMIP, All).
        train_year  (List):             The range of years used for the training set.
        val_year    (List):             The range of years used for the validation set.
        test_year   (List):             The range of years used for the testing set.
        merge       (Bool):             Whether to load the merged version (CWASN + TSMIP + Noise).
        """

        citation = (
            "Kuan-Wei Tang, Kuan-Yu Chen, Da-Yi Chen, Tai-Lin Chin, and Ting-Yu Hsu. (2024)"
            "The CWA Benchmark: A Seismic Dataset from Taiwan for Seismic Research."
            "Seismological Research Letters 2024."
            "doi: https://doi.org/10.1785/0220230393"
        )

        self.subset = "All"
        self.train_year = [2011, 2018]
        self.val_year = [2019]
        self.test_year = [2020, 2021]
        self.subsetYearMapping = {
            "All": [2011, 2021],
            "CWASN": [2012, 2021],
            "TSMIP": [2011, 2020],
            "Noise": [2011, 2021],
        }
        self.merge = True
        self.file = {
            "CWASN": [
                "CWASN2012_2014.tar.gz",
                "CWASN2015_2018.tar.gz",
                "CWASN2019_2021.tar.gz",
            ],
            "TSMIP": ["TSMIP.tar.gz"],
            "Noise": ["noise_chunk1.tar.gz", "noise_chunk2.tar.gz"],
            "All": [
                "merge2011_2014.tar.gz",
                "merge2015_2018.tar.gz",
                "merge2019_2021.tar.gz",
                "noise_chunk1.tar.gz",
                "noise_chunk2.tar.gz",
            ],
        }

        for key, value in kwargs.items():
            if key == "subset":
                assert value in ["CWASN", "TSMIP", "Noise", "All"], print(
                    "Subset should be one of ['CWASN', 'TSMIP', 'Noise', 'All']."
                )

                self.subset = value

            elif key == "merge":
                self.merge = value

            elif key == "train_year":
                self.train_year = value

            elif key == "dev_year":
                self.dev_year = value

            elif key == "test_year":
                self.test_year = value

        super().__init__(citation=citation, **kwargs)

    def _download_dataset(self, writer: WaveformDataWriter, basepath=None, **kwargs):
        # path to seisbench dataset cache
        path = self.path

        if basepath is None:
            seisbench.logger.warning("No cached version of CWA found. ")

        # path to the original files
        basepath = Path(basepath)

        # Data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
        }

        # Check the missing data
        missing, missing_files = self.check(
            basepath, self.subsetYearMapping[self.subset]
        )
        if len(missing) > 0:
            seisbench.logger.warning("The following are missing: ")
            seisbench.logger.warning("=" * 40)
            seisbench.logger.warning("Year -> " + ", ".join(sorted(list(missing))))
            seisbench.logger.warning(
                "Files -> " + ", ".join(sorted(list(missing_files)))
            )
            seisbench.logger.warning("=" * 40)
            seisbench.logger.warning("Start downloading...")

            # ====================== Download ====================== #
            # NLPLabNTUST/Merged-CWA
            if self.merge:
                huggingface_repo = "NLPLabNTUST/Merged-CWA"

            # NLPLabNTUST/NoMerged-CWA
            else:
                huggingface_repo = "NLPLabNTUST/NoMerged-CWA"

            seisbench.logger.warning(f"Huggingface Repo ID: {huggingface_repo}")
            seisbench.logger.warning("=" * 40)

            # Downloading & unarchiving
            for file in list(missing_files):
                res = input(f"Download -> {file}? [Y/n]: ")
                if res == "Y" or res == "y":
                    hf_hub_download(
                        repo_id=huggingface_repo,
                        filename=file,
                        repo_type="dataset",
                        local_dir=basepath,
                    )

                    print("Unarchiving...")
                    self.tar_file(basepath / file, basepath)
                    os.remove(basepath / file)

        # Loading dataset by trace (ex. Phase-picking, PGA estimation, magnitude estimation model, ...)
        print("Loading by trace...")
        writer = self.trace_load(writer, basepath, self.subsetYearMapping[self.subset])

    def tar_file(self, filepath, savepath):
        try:
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=savepath)

        except Exception as e:
            print(f"Tar file exception: {e}")

    def set_split(self, year):
        if year in range(self.dev_year[0], self.dev_year[1] + 1):
            return "dev"
        elif year in range(self.test_year[0], self.test_year[1] + 1):
            return "test"
        else:
            return "train"

    def set_split_noise(self, year_prefix):
        if (2000 + int(year_prefix)) in range(self.dev_year[0], self.dev_year[1] + 1):
            return "dev"
        elif (2000 + int(year_prefix)) in range(
            self.test_year[0], self.test_year[1] + 1
        ):
            return "test"
        else:
            return "train"

    def check(self, basepath, years):
        missing = []
        missing_files = []

        # Check the noise subset
        if self.subset == "Noise" or self.subset == "All":
            for i in range(1, 3):
                hdf5_path = "CWASN_noise_chunk" + str(i) + ".hdf5"
                meta_path = "CWASN_noise_chunk" + str(i) + ".csv"
                if (
                    not (basepath / hdf5_path).is_file()
                    or not (basepath / meta_path).is_file()
                ):
                    missing.append(f"Noise chunk{i}")
                    missing_files.append(f"noise_chunk{i}.tar.gz")

        # Check the entire dataset except noise subset
        if self.subset == "All":
            start = years[0]
            end = years[1]
            for y in range(start, end + 1):
                hdf5_path = "chunks_" + str(y) + ".hdf5"
                meta_path = "metadata_" + str(y) + ".csv"
                if (
                    not (basepath / hdf5_path).is_file()
                    or not (basepath / meta_path).is_file()
                ):
                    missing.append(str(y))

                    if int(y) in range(2011, 2015):
                        missing_files.append("merge2011_2014.tar.gz")
                    elif int(y) in range(2015, 2019):
                        missing_files.append("merge2015_2018.tar.gz")
                    else:
                        missing_files.append("merge2019_2021.tar.gz")

        elif self.subset == "CWASN":
            start = years[0]
            end = years[1]
            for y in range(start, end + 1):
                hdf5_path = "chunks_" + str(y) + ".hdf5"
                meta_path = "metadata_" + str(y) + ".csv"
                if (
                    not (basepath / hdf5_path).is_file()
                    or not (basepath / meta_path).is_file()
                ):
                    missing.append(str(y))

                    if int(y) in range(2012, 2015):
                        missing_files.append("CWASN2012_2014.tar.gz")
                    elif int(y) in range(2015, 2019):
                        missing_files.append("CWASN2015_2018.tar.gz")
                    else:
                        missing_files.append("CWASN2019_2021.tar.gz")

        elif self.subset == "TSMIP":
            start = years[0]
            end = years[1]
            for y in range(start, end + 1):
                hdf5_path = "chunks_" + str(y) + ".hdf5"
                meta_path = "metadata_" + str(y) + ".csv"
                if (
                    not (basepath / hdf5_path).is_file()
                    or not (basepath / meta_path).is_file()
                ):
                    missing.append(str(y))
                    missing_files.append("TSMIP.tar.gz")

        else:
            pass

        return set(missing), set(missing_files)

    def trace_load(self, writer, basepath, years):
        total_trace = 0

        if self.subset != "Noise":
            start = years[0]
            end = years[1]
            for y in range(start, end + 1):
                meta_path = f"metadata_{y}.csv"
                metadata = pd.read_csv(basepath / meta_path)

                metadata["split"] = self.set_split(y)
                print(f"\nyears: {y} -> {self.set_split(y)}")

                hdf5_path = f"chunks_{y}.hdf5"
                with h5py.File(basepath / hdf5_path) as f:
                    gdata = f["data"]
                    for _, row in metadata.iterrows():
                        try:
                            row = row.to_dict()

                            # Adding trace only when waveform is available
                            waveforms = gdata[row["trace_name"]][()]

                            writer.add_trace(row, waveforms)
                            total_trace += 1
                        except:
                            continue

        if self.subset == "All" or self.subset == "Noise":
            for i in range(1, 3):
                print("\nChunk", i)

                meta_path = f"CWASN_noise_chunk{i}.csv"
                hdf5_path = f"CWASN_noise_chunk{i}.hdf5"

                metadata = pd.read_csv(basepath / meta_path)
                with h5py.File(basepath / hdf5_path) as f:
                    gdata = f["data"]

                    for _, row in metadata.iterrows():
                        try:
                            row = row.to_dict()

                            row["split"] = self.set_split_noise(row["trace_name"][:2])

                            # Adding trace only when waveform is available
                            waveforms = gdata[row["trace_name"]][()]

                            writer.add_trace(row, waveforms)

                            total_trace += 1
                        except:
                            continue

        # Total number of traces
        writer.set_total(total_trace)

        return writer
