from __future__ import annotations

from typing import Any

import numpy as np
import obspy
import torch
from obspy.core import Trace
from obspy.signal.filter import bandpass
from scipy.signal import lfilter

import seisbench

from .base import GroupingHelper
from .phasenet import PhaseNet


class DKPN(PhaseNet):
    """
    Domain-Knowledge PhaseNet.

    DKPN uses the PhaseNet U-Net architecture on a five-channel
    feature representation derived from raw three-component waveforms.
    The features are three frequency-band characteristic functions,
    incidence, and modulus.
    ``annotate`` and ``classify`` accept raw three-component ObsPy
    streams and compute these features internally. Direct calls to
    ``forward`` expect precomputed five-channel DKPN features;
    generator training pipelines should use
    :py:class:`seisbench.generate.DKPNPreProcessor`.

    The DKPN reference implementation was released under the MIT
    license by Matteo Bagagli, Anthony Lomax, Sonja Gaviano, and the
    SOME project.

    .. document_args:: seisbench.models DKPN
    """

    _feature_component_order = "ZNEIM"
    _raw_component_order = "ZNE"
    _feature_arg_names = (
        "fp_stabilization",
        "t_long",
        "freqmin",
        "corner",
        "perc_taper",
        "mode",
        "clip",
        "log",
        "normalize",
        "polarization_win_len",
        "use_amax_only",
    )

    _annotate_args = PhaseNet._annotate_args.copy()
    _annotate_args["*_threshold"] = (
        PhaseNet._annotate_args["*_threshold"][0],
        0.2,
    )
    _annotate_args["blinding"] = (
        PhaseNet._annotate_args["blinding"][0],
        (250, 250),
    )
    _annotate_args["overlap"] = (
        PhaseNet._annotate_args["overlap"][0],
        1500,
    )
    _annotate_args["fp_stabilization"] = (
        "Length of the initial stabilization interval used during DKPN training",
        4,
    )
    _annotate_args["t_long"] = (
        "Long-window length in seconds for DKPN frequency-band "
        "characteristic functions",
        4,
    )
    _annotate_args["freqmin"] = (
        "Minimum frequency for DKPN octave-band characteristic functions",
        0.5,
    )
    _annotate_args["corner"] = (
        "Number of corners for the DKPN bandpass filters",
        1,
    )
    _annotate_args["perc_taper"] = (
        "Reference DKPN taper parameter, retained for metadata compatibility",
        0.1,
    )
    _annotate_args["mode"] = (
        "Statistic used for DKPN characteristic functions; currently "
        "only 'rms' is supported",
        "rms",
    )
    _annotate_args["clip"] = (
        "Upper clipping value for DKPN characteristic functions; "
        "values <= 0 disable clipping",
        -999,
    )
    _annotate_args["log"] = (
        "If true, apply log10 transform to DKPN characteristic functions and modulus",
        True,
    )
    _annotate_args["normalize"] = (
        "If true, peak-normalize DKPN characteristic functions and modulus",
        False,
    )
    _annotate_args["polarization_win_len"] = (
        "Reference DKPN metadata key for polarization window length",
        1,
    )
    _annotate_args["use_amax_only"] = (
        "If true, compute incidence/modulus from the globally strongest frequency band",
        False,
    )

    def __init__(
        self,
        in_channels=5,
        classes=3,
        phases="PSN",
        sampling_rate=100,
        component_order="ZNEIM",
        default_args=None,
        **kwargs,
    ):
        citation = (
            "A. Lomax, M. Bagagli, S. Gaviano, S. Cianetti, D. Jozinović, "
            "A. Michelini, C. Zerafa, C. Giunchi (2024). Effects on a "
            "deep-learning, seismic arrival-time picker of domain-knowledge "
            "based preprocessing of input seismograms. Seismica, 3(1). "
            "doi:10.26443/seismica.v3i1.1164"
        )

        super().__init__(
            in_channels=in_channels,
            classes=classes,
            phases=phases,
            sampling_rate=sampling_rate,
            component_order=component_order,
            default_args=default_args,
            **kwargs,
        )
        self._citation = citation

    def annotate_stream_pre(self, stream, argdict):
        super().annotate_stream_pre(stream, argdict)

        if len(stream) == 0:
            return stream

        features = self._extract_feature_stream(stream, argdict)
        if len(features) == 0:
            seisbench.logger.warning(
                "DKPN preprocessing did not find any complete "
                "3-component waveform groups."
            )

        stream.traces = features.traces
        return stream

    def _extract_feature_stream(self, stream, argdict):
        flexible = self._argdict_get_with_default(
            argdict, "flexible_horizontal_components"
        )
        raw_comp_dict = self._raw_component_dict(flexible)
        min_length_s = (self.in_samples - 1) / stream[0].stats.sampling_rate
        groups = GroupingHelper("instrument").group_stream(
            stream,
            strict=True,
            min_length_s=min_length_s,
            comp_dict=raw_comp_dict,
        )

        feature_stream = obspy.Stream()
        for group in groups:
            converted = self._convert_group_to_features(group, raw_comp_dict, argdict)
            if converted is not None:
                feature_stream += converted

        return feature_stream

    @classmethod
    def _feature_default_args(cls):
        return {key: cls._annotate_args[key][1] for key in cls._feature_arg_names}

    @classmethod
    def _feature_args(cls, argdict):
        feature_args = cls._feature_default_args()
        feature_args.update(
            {key: argdict[key] for key in cls._feature_arg_names if key in argdict}
        )
        return feature_args

    @staticmethod
    def _raw_component_dict(flexible_horizontal_components):
        comp_dict = {"Z": 0, "N": 1, "E": 2}
        if flexible_horizontal_components:
            comp_dict["1"] = comp_dict["N"]
            comp_dict["2"] = comp_dict["E"]
        return comp_dict

    def _convert_group_to_features(self, group, raw_comp_dict, argdict):
        traces_by_comp = {}
        for trace in group:
            component = trace.id[-1]
            if component not in raw_comp_dict:
                continue

            comp_idx = raw_comp_dict[component]
            if comp_idx in traces_by_comp:
                seisbench.logger.warning(
                    f"Multiple traces map to DKPN raw component "
                    f"{self._raw_component_order[comp_idx]} for "
                    f"{GroupingHelper.trace_id_without_component(trace)}. "
                    f"Using the first one."
                )
                continue

            traces_by_comp[comp_idx] = trace

        if any(comp_idx not in traces_by_comp for comp_idx in range(3)):
            return None

        waveforms, start_time, sampling_rate = self._group_to_array(traces_by_comp)
        extractor = _DKPNFeatureExtractor(**self._feature_args(argdict))
        features = extractor.matrix_cfs(waveforms, sampling_rate=sampling_rate)

        out = obspy.Stream()
        for idx, component in enumerate(self._feature_component_order):
            if idx < 3:
                stats = traces_by_comp[idx].stats.copy()
            else:
                stats = traces_by_comp[0].stats.copy()
            stats.starttime = start_time
            stats.sampling_rate = sampling_rate
            stats.channel = f"CF{component}"
            out += Trace(data=features[idx].copy(), header=stats)

        return out

    @staticmethod
    def _group_to_array(traces_by_comp):
        sampling_rate = traces_by_comp[0].stats.sampling_rate
        t_start = min(trace.stats.starttime for trace in traces_by_comp.values())
        t_end = max(trace.stats.endtime for trace in traces_by_comp.values())
        n_samples = int(round((t_end - t_start) * sampling_rate)) + 1

        data = np.zeros((3, n_samples), dtype=np.float64)
        t_offsets = []
        for comp_idx, trace in traces_by_comp.items():
            offset = int(round((trace.stats.starttime - t_start) * sampling_rate))
            end = min(offset + trace.data.size, n_samples)
            data[comp_idx, offset:end] = trace.data[: end - offset]
            t_offsets.append(trace.stats.get("t_offset", 0.0))

        start_time = t_start + float(np.mean(t_offsets))
        return data, start_time, sampling_rate

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        if batch.ndim != 3 or batch.shape[1] != self.in_channels:
            raise ValueError(
                "DKPN expects five precomputed feature channels with component "
                "order 'ZNEIM'. Use model.annotate(stream) or "
                "model.classify(stream) for raw ObsPy streams, and use "
                "seisbench.generate.DKPNPreProcessor in training generators "
                "before calling annotate_batch_pre()."
            )

        std = batch[:, 0:3, :].std(axis=-1, keepdims=True)
        batch[:, 0:3, :] = batch[:, 0:3, :] / (std + 1e-10)

        std = batch[:, 4:5, :].std(axis=-1, keepdims=True)
        batch[:, 4:5, :] = batch[:, 4:5, :] / (std + 1e-10)

        return batch

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
            "filter_args",
            "filter_kwargs",
            "grouping",
            "norm",
            "norm_amp_per_comp",
            "norm_detrend",
            "filter_factor",
        ]:
            if key in model_args:
                del model_args[key]

        model_args["component_order"] = self.component_order
        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate

        return model_args


class _DKPNFeatureExtractor:
    def __init__(self, **kwargs):
        expected = set(DKPN._feature_arg_names)
        missing = expected - set(kwargs)
        if missing:
            raise ValueError(f"Missing DKPN feature arguments: {sorted(missing)}")

        unknown = set(kwargs) - expected
        if unknown:
            raise ValueError(f"Unknown DKPN feature arguments: {sorted(unknown)}")

        self.__dict__.update(kwargs)
        self.eps = 1e-10

    def matrix_cfs(self, waveforms, sampling_rate):
        waveforms = waveforms.astype(np.float64, copy=True)
        waveforms = waveforms - np.mean(waveforms, axis=1, keepdims=True)
        waveforms = waveforms / (np.std(waveforms, axis=1, keepdims=True) + self.eps)

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
