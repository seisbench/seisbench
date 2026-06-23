"""
This module contains shared functions for DKPN that are required in seisbench.models and seisbench.generate.
It exists to avoid cross-imports between models and generate.
"""

from __future__ import annotations

import numpy as np
from obspy.signal.filter import bandpass
from scipy.signal import lfilter

_DKPN_ANNOTATE_DEFAULTS = {
    "*_threshold": 0.2,
    "overlap": 1500,
    "blinding": (250, 250),
}
_DKPN_FEATURE_DEFAULTS = {
    "fp_stabilization": 4,
    "t_long": 4,
    "freqmin": 0.5,
    "corner": 1,
    "perc_taper": 0.1,
    "mode": "rms",
    "clip": -999,
    "log": True,
    "normalize": False,
    "polarization_win_len": 1,
    "use_amax_only": False,
}


def _dkpn_stabilization_samples(fp_stabilization, sampling_rate):
    return int(round(float(fp_stabilization) * float(sampling_rate)))


_feature_arg_names = tuple(_DKPN_FEATURE_DEFAULTS)


class _DKPNFeatureExtractor:
    def __init__(self, **kwargs):
        expected = set(_feature_arg_names)
        missing = expected - set(kwargs)
        if missing:
            raise ValueError(f"Missing DKPN feature arguments: {sorted(missing)}")

        unknown = set(kwargs) - expected
        if unknown:
            raise ValueError(f"Unknown DKPN feature arguments: {sorted(unknown)}")

        self.__dict__.update(kwargs)
        self.eps = 1e-10

    def matrix_cfs(self, waveforms, sampling_rate):
        waveforms = waveforms.astype(np.float64, copy=False)
        waveforms -= np.mean(waveforms, axis=1, keepdims=True)
        waveforms /= np.std(waveforms, axis=1, keepdims=True) + self.eps

        cf_waveforms = []
        band_data = []
        max_bands = []
        max_values = []

        for trace_data in waveforms:
            summary = _FBSummary(
                trace_data,
                sampling_rate=float(sampling_rate),
                npts=trace_data.shape[0],
                t_long=self.t_long,
                freqmin=self.freqmin,
                corner=self.corner,
                perc_taper=self.perc_taper,
                mode=self.mode,
            )

            cf_waveforms.append(summary.summary)
            band_data.append(summary.bf)
            max_bands.append(np.argmax(summary.fc, axis=0))
            max_values.append(summary.summary)

        cf_waveforms = np.asarray(cf_waveforms)

        if self.clip > 0.0:
            cf_waveforms = np.clip(cf_waveforms, a_min=None, a_max=self.clip)
        if self.log:
            cf_waveforms = np.log10(cf_waveforms + 1.0)
        if self.normalize:
            cf_waveforms = cf_waveforms / (
                np.amax(np.abs(cf_waveforms), axis=(0, 1), keepdims=True) + self.eps
            )

        band_data = np.asarray(band_data)
        max_bands = np.asarray(max_bands)
        max_bands_amax = np.amax(max_bands, axis=0)

        band_amax = -1
        if self.use_amax_only:
            max_values = np.asarray(max_values)
            index_amax = np.unravel_index(np.argmax(max_values), max_values.shape)
            band_amax = max_bands[index_amax]

        incidence, modulus = self._polarization_fp(
            band_data, max_bands_amax, band_amax=band_amax
        )

        features = np.concatenate(
            [cf_waveforms, incidence[None, :], modulus[None, :]], axis=0
        )
        return features.astype("float32")

    def _polarization_fp(self, band_data, max_bands_amax, band_amax=-1):
        if self.use_amax_only:
            vertical = band_data[0][band_amax]
            north = band_data[1][band_amax]
            east = band_data[2][band_amax]
        else:
            trace_len = len(band_data[0][0])
            data = band_data[:, max_bands_amax, np.arange(trace_len, dtype=int)]
            vertical, north, east = data[0], data[1], data[2]

        return self._incidence_modulus(vertical, north, east)

    def _incidence_modulus(self, vertical, north, east):
        hxy = np.hypot(north, east)
        modulus = np.hypot(hxy, vertical)

        if np.max(hxy) > np.max(vertical) / 1000.0:
            incidence = np.arctan2(vertical, hxy)
            incidence = incidence / (np.pi / 2.0)
        else:
            incidence = np.zeros_like(vertical)

        if self.log:
            modulus = np.log10(modulus + 1.0)
        if self.normalize:
            modulus = modulus / (np.max(modulus + 1e-6))

        return incidence, modulus


class _FBSummary:
    def __init__(
        self,
        data,
        npts=3001,
        sampling_rate=100.0,
        t_long=5,
        freqmin=1,
        corner=1,
        perc_taper=0.1,
        mode="rms",
    ):
        self.data = data
        self.npts = npts
        self.sampling_rate = sampling_rate
        self.delta = 1 / self.sampling_rate
        self.t_long = t_long
        self.freqmin = freqmin
        self.corner = corner
        self.perc_taper = perc_taper
        self.statistics_mode = mode

        self.fc, self.bf = self._statistics_decay()
        self.summary = np.amax(self.fc, axis=0)

    def _n_bands(self):
        nyquist = self.sampling_rate / 2.0
        return int(np.log2(nyquist / 1.5 / self.freqmin)) + 1

    def filter(self):
        n_bands = self._n_bands()
        bf = np.zeros(shape=(n_bands, self.npts))

        for band in range(n_bands):
            octave_high = (self.freqmin + self.freqmin * 2.0) / 2.0 * (2**band)
            octave_low = octave_high / 2.0
            bf[band] = bandpass(
                self.data,
                octave_low,
                octave_high,
                self.sampling_rate,
                corners=self.corner,
                zerophase=False,
            )

        return bf

    def _statistics_decay(self):
        if self.t_long <= 0:
            raise ValueError("DKPN t_long must be positive.")

        decay_factor = self.delta / self.t_long
        decay_const = 1.0 - decay_factor

        bf = self.filter()
        energy = np.power(bf, 2)

        if self.statistics_mode == "rms":
            squared_energy = np.power(energy, 2)
            average_energy = lfilter(
                [decay_factor], [1.0, -decay_const], squared_energy, axis=1
            )
            sqrt_average_energy = np.sqrt(average_energy)
            rms_energy = lfilter(
                [decay_factor], [1.0, -decay_const], sqrt_average_energy, axis=1
            )
            fc = np.abs(energy) / (rms_energy + 1.0e-6)
        elif self.statistics_mode == "std":
            raise NotImplementedError("DKPN statistics mode 'std' is not implemented.")
        else:
            raise ValueError(f"Unknown DKPN statistics mode '{self.statistics_mode}'.")

        stabilization_samples = min(
            int(round(self.t_long / self.delta, 0)), fc.shape[1]
        )
        fc[:, :stabilization_samples] = 0

        return fc, bf


_raw_component_order = "ZNE"


def _raw_component_dict(flexible_horizontal_components):
    comp_dict = {"Z": 0, "N": 1, "E": 2}
    if flexible_horizontal_components:
        comp_dict["1"] = comp_dict["N"]
        comp_dict["2"] = comp_dict["E"]
    return comp_dict
