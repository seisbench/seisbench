import seisbench
from .base import WaveformDataset

from pathlib import Path
import shutil
import h5py


class STEAD(WaveformDataset):
    """
    STEAD dataset
    """

    def __init__(self, **kwargs):
        citation = (
            "Mousavi, S. M., Sheng, Y., Zhu, W., Beroza G.C., (2019). STanford EArthquake Dataset (STEAD): "
            "A Global Data Set of Seismic Signals for AI, IEEE Access, doi:10.1109/ACCESS.2019.2947848"
        )
        super().__init__(
            name=self.__class__.__name__.lower(), citation=citation, **kwargs
        )

    def _download_dataset(self, basepath=None, **kwargs):
        download_instructions = (
            "Please download STEAD following the instructions at https://github.com/smousavi05/STEAD. "
            "Provide the locations of the STEAD unpacked files (merged.csv and merged.hdf5) as parameter basepath to the class. "
            "This step is only necessary the first time STEAD is loaded."
        )

        if basepath is None:
            raise ValueError(
                "No cached version of STEAD found. " + download_instructions
            )

        basepath = Path(basepath)

        if not (basepath / "merged.csv").is_file():
            raise ValueError(
                "Basepath does not contain file merged.csv. " + download_instructions
            )
        if not (basepath / "merged.hdf5").is_file():
            raise ValueError(
                "Basepath does not contain file merged.hdf5. " + download_instructions
            )

        self._dataset_path().mkdir(parents=True, exist_ok=True)
        seisbench.logger.warning(
            "Copying STEAD files to cache. This might take a while."
        )
        shutil.copy(basepath / "merged.csv", self._dataset_path() / "metadata.csv")
        shutil.copy(basepath / "merged.hdf5", self._dataset_path() / "waveforms.hdf5")

        data_format = {
            "dimension_order": "WC",
            "component_order": "ENZ",
            "sampling_rate": 100,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        with h5py.File(self._dataset_path() / "waveforms.hdf5", "a") as fout:
            g_data_format = fout.create_group("data_format")
            for key in data_format.keys():
                g_data_format.create_dataset(key, data=data_format[key])
