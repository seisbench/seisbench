from abc import ABC

import numpy as np
import h5py
import seisbench

from .base import WaveformDataset, AbstractBenchmarkDataset


class SpectrumDataset(WaveformDataset):
    """
    This class allows to handle datasets consisting of spectra. It is very similar to the regular WaveformDataset,
    but with small adjustments for spectra. The automatic resampling has been dropped. Instead, the dataset has a
    ``frequencies`` attribute. The frequencies are read from the hdf5 file at `data/x_axis` and need to be identical
    for all samples.
    """

    def __init__(self, path, *, sampling_rate: None = None, **kwargs):
        if sampling_rate is not None:
            raise ValueError(
                "Spectrum datasets do not support sampling rates. Parameter 'sampling_rate' must be None."
            )
        super().__init__(path=path, **kwargs)

        self._load_frequencies()
        self._metadata["trace_sampling_rate_hz"] = (
            0.0  # A hack to make the backend work. Will be ignored later anyhow.
        )

    def _load_frequencies(self):
        self._frequencies = None
        for waveform_file in self._chunks_with_paths()[2]:
            with h5py.File(waveform_file, "r") as f_wave:
                if "data/x_axis" not in f_wave:
                    seisbench.logger.warning("Frequencies not specified in .hdf5 file.")

                if self._frequencies is None:
                    self._frequencies = f_wave["data/x_axis"][()]
                else:
                    if not np.allclose(self._frequencies, f_wave["data/x_axis"][()]):
                        raise ValueError(
                            "Inconsistent frequencies found between chunks."
                        )

    @property
    def frequencies(self):
        return self._frequencies

    # The following operations can all be no-ops for SpectrumDatasets
    def _unify_sampling_rate(self, *args, **kwargs):
        pass

    def _get_sample_unify_sampling_rate(self, *args, **kwargs):
        return None

    def _resample(self, waveform, *args, **kwargs):
        return waveform

    # Delete function and attributes
    get_waveforms = None
    sampling_rate = None
    resample_zerophase = None

    # Remove parameters
    def get_sample(self, idx):
        return super().get_sample(idx)

    # Renamed function
    def get_spectra(self, idx=None, mask=None):
        return super().get_waveforms(idx=idx, mask=mask)


class SpectrumBenchmarkDataset(AbstractBenchmarkDataset, SpectrumDataset, ABC):
    """
    This class is the base class for benchmark DAS datasets. For the functionality, see the superclasses.
    """

    _files = ["metadata$CHUNK.csv", "waveforms$CHUNK.hdf5"]
