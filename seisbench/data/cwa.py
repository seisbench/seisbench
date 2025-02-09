import os
import tarfile
from abc import ABC

import pandas as pd

import seisbench

from .base import BenchmarkDataset, WaveformDataWriter


class CWABase(BenchmarkDataset, ABC):
    citation = (
        "Kuan-Wei Tang, Kuan-Yu Chen, Da-Yi Chen, Tai-Lin Chin, and Ting-Yu Hsu. (2024)"
        "The CWA Benchmark: A Seismic Dataset from Taiwan for Seismic Research."
        "Seismological Research Letters 2024."
        "doi: https://doi.org/10.1785/0220230393"
    )

    src_repoName = "NLPLabNTUST/Merged-CWA"
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

    @staticmethod
    def _download_from_huggingfaceHub(path, src_file, file_type, repo_name):
        # install the dependency package for downloading
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            seisbench.logger.exception(e)
            seisbench.logger.exception(
                "See https://huggingface.co/docs/huggingface_hub/installation."
            )

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
        try:
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=savepath)

        except Exception as e:
            seisbench.logger.exception(f"Tar file exception: {e}")

    def get_chunks(self, chunks_fileName):
        path = self.path

        chunks_path = path / chunks_fileName
        if not chunks_path.is_file():
            # install the dependency package for downloading
            self._download_from_huggingfaceHub(
                path, "chunks", "chunks information", self.src_repoName
            )

        with open(chunks_path, "r") as f:
            chunks = [x for x in f.read().split("\n") if x.strip()]

        return chunks

    def _download_pipeline(self, writer, chunk, **kwargs):  # chunk: _2011 (for example)
        toDownload = self.chunk2file[chunk]

        path = self.path

        self._download_from_huggingfaceHub(
            path, toDownload, "source_file", self.src_repoName
        )

        seisbench.logger.warning("Unarchiving. This might take a few minutes.")
        self.tar_file(path / toDownload, path)

        seisbench.logger.warning("Remove the source file.")
        os.remove(path / toDownload)


class CWA(CWABase):
    def __init__(self, **kwargs):

        chunks = self.get_chunks("chunks")
        self.src_repoName = "NLPLabNTUST/Merged-CWA"

        super().__init__(citation=self.citation, chunks=chunks, **kwargs)

    def _download_dataset(self, writer, chunk, **kwargs):

        self._download_pipeline(writer, chunk)

        self._add_split(writer.metadata_path, self.path, self.src_repoName)

    @staticmethod
    def _add_split(metadata_path, path, repo_name):
        def download_split(path, repo_name):
            # install the dependency package for downloading
            try:
                from huggingface_hub import hf_hub_download
            except Exception as e:
                seisbench.logger.exception(e)
                seisbench.logger.exception(
                    "See https://huggingface.co/docs/huggingface_hub/installation."
                )

            seisbench.logger.warning(
                f"Start downloading split.txt from Huggingface Hub: {repo_name}"
            )

            # download from huggingface hub
            hf_hub_download(
                repo_id=repo_name,
                filename="split.csv",
                repo_type="dataset",
                local_dir=path,
            )

        split_path = path / "split.csv"
        if not split_path.is_file():
            download_split(path, repo_name)

        metadata = pd.read_csv(metadata_path)
        year = metadata_path.split("metadata_")[1].split(".csv")[0]

        df = pd.read_csv(split_path)
        split = df.loc[df["year"] == year]["split"].item()
        metadata["split"] = split
        metadata.to_csv(metadata_path, index=False)


class CWANoise(CWABase):
    def __init__(self, **kwargs):
        chunks = self.get_chunks("chunks")
        self.src_repoName = "NLPLabNTUST/Merged-CWA-Noise"

        super().__init__(citation=self.citation, chunks=chunks, **kwargs)

    def _download_dataset(self, writer, chunk, **kwargs):

        self._download_pipeline(writer, chunk)

        self._add_split(writer.metadata_path, self.path, self.src_repoName)

    @staticmethod
    def _add_split(metadata_path, path, repo_name):
        def download_split(path, repo_name):
            # install the dependency package for downloading
            try:
                from huggingface_hub import hf_hub_download
            except Exception as e:
                seisbench.logger.exception(e)
                seisbench.logger.exception(
                    "See https://huggingface.co/docs/huggingface_hub/installation."
                )

            seisbench.logger.warning(
                f"Start downloading split.txt from Huggingface Hub: {repo_name}"
            )

            # download from huggingface hub
            hf_hub_download(
                repo_id=repo_name,
                filename="split.csv",
                repo_type="dataset",
                local_dir=path,
            )

        def split_by_year(trace_name, df):
            year = "20" + trace_name[:2]

            return df.loc[df["year"] == int(year)]["split"].item()

        split_path = path / "split.csv"
        if not split_path.is_file():
            download_split(path, repo_name)

        # metadata
        metadata = pd.read_csv(metadata_path)

        # split info
        df = pd.read_csv(split_path)

        # split
        metadata["split"] = metadata["trace_name"].apply(split_by_year, df=df)
        metadata.to_csv(metadata_path, index=False)
