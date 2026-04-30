from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from typing import Any
from scipy.signal import butter, sosfilt

from seisbench import logger
from .das_base import DASBenchmarkDataset, DASDataWriter


class MLSubDAS(DASBenchmarkDataset):
    chunk_count = 25
    chunk_size = 100
    min_total_labels = 500
    min_phase_labels = 250

    def __init__(self, **kwargs):
        logger.warning(
            "This dataset has been annotated semi-automatically. The annotations will be incomplete and high picking "
            "errors (>1 s) or spatially incoherent labeling occurs. Check out the original publication for details on "
            "the dataset and annotation strategy. The limitations of this dataset should be taken into account when "
            "using it for training or evaluation."
        )
        citation = (
            "Xiao, H., van den Ende, M., Tilmann, F., Rivet, D., Loureiro, A., Tsuji, T., ... & "
            "Denolle, M. (2025). DeepSubDAS: An Earthquake Phase Picker from Submarine Distributed "
            "Acoustic Sensing Data."
        )
        license = "CC BY 4.0"
        # TODO: Remove compile_from_source argument
        # TODO: Enable repository lookup
        super().__init__(
            citation=citation,
            license=license,
            compile_from_source=True,
            repository_lookup=False,
            **kwargs,
        )

    @classmethod
    def available_chunks(cls, force: bool = False, wait_for_file: bool = False):
        return [f"{i:02d}" for i in range(cls.chunk_count)]

    def _download_dataset(
        self, files: list[Path], chunk: str, base_path: Path = None, **kwargs
    ):
        """
        Converts dataset from local files.
        This function is only for reference, as it relies on a local file structure.
        """
        if base_path is None:
            raise ValueError("`base_path` must be provided for conversion from source.")
        entries = self._scan_files(base_path)
        entries = entries[entries["total_labels"] >= self.min_total_labels]

        chunk_idx = int(chunk)
        entries = entries[
            chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size
        ]

        data_key = "data"

        with DASDataWriter(
            self.path, chunk, files[0], files[1], strict=False
        ) as writer:
            for _, entry in tqdm(entries.iterrows(), total=len(entries)):
                csv_path = base_path / entry["folder"] / (entry["file"] + ".csv")
                if not csv_path.is_file():
                    csv_path = (
                        base_path / entry["folder"] / (entry["file"] + ".mat.csv")
                    )
                hdf_path = base_path / entry["folder"] / (entry["file"] + ".h5")

                metadata = pd.read_csv(csv_path)

                with h5py.File(hdf_path, "r") as f:
                    if data_key not in f:
                        print(f"Problem with file {entry['folder']}/{entry['file']}")
                        continue
                    # Data shape: (samples, channels)
                    data = f[data_key][()]

                data = self._preprocess_data(data)
                annotations = self._convert_annotations(metadata, data.shape[1], entry)

                # TODO: Get event metadata
                record_metadata = self._get_record_metadata(entry)

                writer.add_record(record_metadata, data, annotations)

    @staticmethod
    def _scan_files(base_path: Path) -> pd.DataFrame:
        entries = []

        def truncate_csv_name(x):
            x = x.name[:-4]
            if x.endswith(".mat"):
                x = x[:-4]
            return x

        for folder in base_path.iterdir():
            csv_files = [truncate_csv_name(x) for x in sorted(folder.glob("*.csv"))]
            hdf_files = set(x.name[:-3] for x in sorted(folder.glob("*.h5")))

            for file in csv_files:
                if file in hdf_files:
                    try:
                        metadata = pd.read_csv(
                            folder / f"{file}.csv",
                            dtype={
                                "p_wave_index": np.float32,
                                "s_wave_index": np.float32,
                            },
                        )
                    except FileNotFoundError:
                        metadata = pd.read_csv(
                            folder / f"{file}.mat.csv",
                            dtype={
                                "p_wave_index": np.float32,
                                "s_wave_index": np.float32,
                            },
                        )
                    if "p_wave_index" in metadata.columns:
                        p_labels = np.sum(~np.isnan(metadata["p_wave_index"]))
                        s_labels = np.sum(~np.isnan(metadata["s_wave_index"]))
                    else:
                        p_labels, s_labels = 0, 0
                else:
                    p_labels, s_labels = 0, 0

                entries.append(
                    {
                        "folder": folder.name,
                        "file": file,
                        "has_hdf5": file in hdf_files,
                        "p_labels": p_labels,
                        "s_labels": s_labels,
                    }
                )

        entries = pd.DataFrame(entries)
        entries["total_labels"] = entries["p_labels"] + entries["s_labels"]

        return entries

    @staticmethod
    def _convert_annotations(
        metadata: pd.DataFrame,
        n_channels: int,
        entry: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        # Should output P_0, P_1, ... in case of multiple P waves
        index_column = "Unnamed: 0"
        annotations = {}
        for phase in "PS":
            if entry[f"{phase.lower()}_labels"] >= MLSubDAS.min_phase_labels:
                phase_labels = metadata[
                    ~np.isnan(metadata[f"{phase.lower()}_wave_index"])
                ]
                break_points = (
                    [0]
                    + list(np.where(np.diff(phase_labels[index_column]) < 0)[0])
                    + [len(phase_labels)]
                )

                for i, (p0, p1) in enumerate(zip(break_points[:-1], break_points[1:])):
                    annotation = np.nan * np.zeros(n_channels)
                    index_vals = phase_labels[index_column].values[p0:p1]
                    pick_samples = phase_labels[f"{phase.lower()}_wave_index"].values[
                        p0:p1
                    ]
                    annotation[index_vals] = pick_samples
                    annotations[f"{phase}_{i}"] = annotation

        return annotations

    @staticmethod
    def _preprocess_data(data: np.ndarray) -> np.ndarray:
        def butter_bandpass(low, high, fs=100, order=4):
            nyq = 0.5 * fs
            low /= nyq
            high /= nyq
            sos = butter(order, [low, high], btype="band", output="sos")
            return sos

        def cosine_window(n, boundary_samples=100):
            boundary_samples = min(boundary_samples, n // 2)
            x = np.ones(n)
            x[:boundary_samples] = (
                1 - np.cos(np.linspace(0, np.pi, boundary_samples))
            ) / 2
            x[-boundary_samples:] = (
                1 + np.cos(np.linspace(0, np.pi, boundary_samples))
            ) / 2
            return x

        sos = butter_bandpass(1, 20, 100)

        window = cosine_window(data.shape[0]).reshape(-1, 1)

        data = window * (data - np.mean(data, axis=0, keepdims=True))
        data = sosfilt(sos, data, axis=0)

        return data.astype(np.float32)

    @staticmethod
    def _get_record_metadata(entry: pd.Series) -> dict[str, Any]:
        general = {
            "record_identifier": entry["file"],
            "record_sampling_rate_hz": 100.0,
            "record_p_labels": entry["p_labels"]
            if entry["p_labels"] >= MLSubDAS.min_phase_labels
            else 0,
            "record_s_labels": entry["s_labels"]
            if entry["s_labels"] >= MLSubDAS.min_phase_labels
            else 0,
        }
        cable = {}
        if entry["folder"] == "das_alaska":
            # Stated in Xiao et al. (2026)
            cable = {
                "record_channel_spacing_m": 9.57,
            }
        elif entry["folder"] == "das_chillie":
            if "CCN" in entry["file"] or "SER" in entry["file"]:
                # Stated in Xiao et al. (2026)
                cable = {
                    "record_channel_spacing_m": 10.0,
                }
            else:
                # Stated in Xiao et al. (2026)
                cable = {
                    "record_channel_spacing_m": 4.08,
                }
        elif entry["folder"] == "das_japan":
            # Stated in Xiao et al. (2026)
            cable = {
                "record_channel_spacing_m": 10.0,
            }
        elif entry["folder"] == "das_spain":
            # Stated in Xiao et al. (2026)
            cable = {
                "record_channel_spacing_m": 10.0,
            }
        elif entry["folder"] == "other_data":
            if "NyAalesund" in entry["file"]:
                # Stated in Bouffaut et al. (2022)
                cable = {
                    "record_channel_spacing_m": 4.08,
                }
            elif "maderia" in entry["file"]:
                # Stated in Loureiro et al. (2025)
                cable = {
                    "record_channel_spacing_m": 5.1,
                }

        return {**general, **cable}
