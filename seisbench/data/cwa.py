import os
import tarfile
from abc import ABC

import pandas as pd

import seisbench
import seisbench.util

from .base import WaveformBenchmarkDataset

try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None


class CWABase(WaveformBenchmarkDataset, ABC):
    """
    An abstract class for downloading datasets.
    The CWA dataset comprises data from two seismographic networks: CWASN and TSMIP.
    The dataset spans from 2011 to 2021 and primarily includes P and S wave arrivals.
    Additionally, a subset of noise data is provided.
    """

    citation = (
        "Kuan-Wei Tang, Kuan-Yu Chen, Da-Yi Chen, Tai-Lin Chin, and Ting-Yu Hsu. (2024)"
        "The CWA Benchmark: A Seismic Dataset from Taiwan for Seismic Research."
        "Seismological Research Letters 2024."
        "doi: https://doi.org/10.1785/0220230393"
    )

    chunk2file = {
        "_2011": "merge2011_2014.tar.gz",
        "_2012": "merge2011_2014.tar.gz",
        "_2013": "merge2011_2014.tar.gz",
        "_2014": "merge2011_2014.tar.gz",
        "_2015": "merge2015_2018.tar.gz",
        "_2016": "merge2015_2018.tar.gz",
        "_2017": "merge2015_2018.tar.gz",
        "_2018": "merge2015_2018.tar.gz",
        "_2019": "merge2019_2021.tar.gz",
        "_2020": "merge2019_2021.tar.gz",
        "_2021": "merge2019_2021.tar.gz",
        "_noise1": "noise_chunk1.tar.gz",
        "_noise2": "noise_chunk2.tar.gz",
    }

    src_repo_name = None

    def __init__(self, **kwargs):
        assert self.src_repo_name is not None, (
            "Subclass needs to overwrite src_repo_name"
        )
        super().__init__(citation=self.citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer, chunk, **kwargs):
        self._download_pipeline(writer, chunk)

    @staticmethod
    def _ensure_hf_hub_download_available():
        assert hf_hub_download is not None, (
            "To download this dataset, huggingface_hub must be installed. "
            "For installation instructions, "
            "see https://huggingface.co/docs/huggingface_hub/installation"
        )

    @staticmethod
    def _download_from_huggingfaceHub(path, src_file, file_type, repo_name):
        CWABase._ensure_hf_hub_download_available()
        seisbench.logger.warning(
            f"Start downloading {file_type} from Huggingface Hub: {repo_name}"
        )

        # download from huggingface hub
        hf_hub_download(
            repo_id=repo_name,
            filename=src_file,
            repo_type="dataset",
            local_dir=path,
        )

    def tar_file(self, filepath, savepath):
        with tarfile.open(filepath, "r:gz") as tar:
            seisbench.util.safe_extract_tar(tar, savepath)

    @classmethod
    def available_chunks(cls, force=False, wait_for_file=False):
        path = cls._path_internal()

        chunks_path = path / "chunks"
        if not chunks_path.is_file():
            cls._download_from_huggingfaceHub(
                path, "chunks", "chunks information", cls.src_repo_name
            )

        with open(chunks_path, "r") as f:
            chunks = [x for x in f.read().split("\n") if x.strip()]

        return chunks

    def _download_pipeline(self, writer, chunk, **kwargs):  # chunk: _2011 (for example)
        to_download = self.chunk2file[chunk]

        path = self.path

        self._download_from_huggingfaceHub(
            path, to_download, "source_file", self.src_repo_name
        )

        seisbench.logger.warning("Unarchiving. This might take a few minutes.")
        self.tar_file(path / to_download, path)

        for file in path.iterdir():
            if file.name.startswith("metadata") and file.name.endswith(".csv"):
                self._add_split(file)

        seisbench.logger.warning("Remove the source file.")
        os.remove(path / to_download)

    @staticmethod
    def _add_split(metadata_path):
        def split_by_year(trace_name):
            year = int("20" + trace_name[:2])
            if year <= 2018:
                return "train"
            elif year == 2019:
                return "dev"
            else:
                return "test"

        metadata = pd.read_csv(metadata_path)
        if "split" in metadata.columns:
            return  # No action required

        metadata["split"] = metadata["trace_name"].apply(split_by_year)
        metadata.to_csv(metadata_path, index=False)


class CWA(CWABase):
    """
    CWA dataset - Events and traces.
    """

    src_repo_name = "NLPLabNTUST/Merged-CWA"


class CWANoise(CWABase):
    """
    CWA dataset - Noise samples.
    """

    src_repo_name = "NLPLabNTUST/Merged-CWA-Noise"
