from __future__ import annotations

from typing import Any

import numpy as np
import obspy
import torch
from obspy.core import Trace

import seisbench

from .base import GroupingHelper
from .phasenet import PhaseNet
from seisbench.util.dkpn_shared import (
    _DKPN_ANNOTATE_DEFAULTS,
    _DKPN_FEATURE_DEFAULTS,
    _dkpn_stabilization_samples,
    _feature_arg_names,
    _DKPNFeatureExtractor,
    _raw_component_order,
    _raw_component_dict,
)


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
    DKPN feature-generation parameters are model configuration and
    should be set through the constructor or weight metadata. Per-call
    ``annotate`` and ``classify`` keyword arguments are reserved for
    runtime inference options such as thresholds, overlap, blinding,
    batch size, and component handling.
    Stream inference removes the initial DKPN feature stabilization
    interval before windowing, matching training generators that use
    ``DKPNPreProcessor(output_samples=model.in_samples)``.

    The DKPN reference implementation was released under the MIT
    license by Matteo Bagagli, Anthony Lomax, Sonja Gaviano, and the
    SOME project.

    .. document_args:: seisbench.models DKPN
    """

    _feature_component_order = "ZNEIM"

    _annotate_args = PhaseNet._annotate_args.copy()
    _annotate_args["*_threshold"] = (
        PhaseNet._annotate_args["*_threshold"][0],
        _DKPN_ANNOTATE_DEFAULTS["*_threshold"],
    )
    _annotate_args["blinding"] = (
        PhaseNet._annotate_args["blinding"][0],
        _DKPN_ANNOTATE_DEFAULTS["blinding"],
    )
    _annotate_args["overlap"] = (
        PhaseNet._annotate_args["overlap"][0],
        _DKPN_ANNOTATE_DEFAULTS["overlap"],
    )

    def __init__(
        self,
        in_channels=5,
        classes=3,
        phases="PSN",
        sampling_rate=100,
        component_order="ZNEIM",
        fp_stabilization=_DKPN_FEATURE_DEFAULTS["fp_stabilization"],
        t_long=_DKPN_FEATURE_DEFAULTS["t_long"],
        freqmin=_DKPN_FEATURE_DEFAULTS["freqmin"],
        corner=_DKPN_FEATURE_DEFAULTS["corner"],
        perc_taper=_DKPN_FEATURE_DEFAULTS["perc_taper"],
        mode=_DKPN_FEATURE_DEFAULTS["mode"],
        clip=_DKPN_FEATURE_DEFAULTS["clip"],
        log=_DKPN_FEATURE_DEFAULTS["log"],
        normalize=_DKPN_FEATURE_DEFAULTS["normalize"],
        polarization_win_len=_DKPN_FEATURE_DEFAULTS["polarization_win_len"],
        use_amax_only=_DKPN_FEATURE_DEFAULTS["use_amax_only"],
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
        self._set_feature_args(
            {
                "fp_stabilization": fp_stabilization,
                "t_long": t_long,
                "freqmin": freqmin,
                "corner": corner,
                "perc_taper": perc_taper,
                "mode": mode,
                "clip": clip,
                "log": log,
                "normalize": normalize,
                "polarization_win_len": polarization_win_len,
                "use_amax_only": use_amax_only,
            }
        )

    def annotate_stream_pre(self, stream, argdict):
        super().annotate_stream_pre(stream, argdict)

        if len(stream) == 0:
            return stream

        features = self._extract_feature_stream(stream, argdict)
        if len(features) == 0:
            seisbench.logger.warning(
                "DKPN preprocessing did not find any complete 3-component "
                "waveform groups with enough samples after stabilization."
            )

        stream.traces = features.traces
        return stream

    def _extract_feature_stream(self, stream, argdict):
        flexible = self._argdict_get_with_default(
            argdict, "flexible_horizontal_components"
        )
        raw_comp_dict = _raw_component_dict(flexible)
        sampling_rate = stream[0].stats.sampling_rate
        stabilization_samples = _dkpn_stabilization_samples(
            self.fp_stabilization, sampling_rate
        )
        min_samples = self.in_samples + stabilization_samples
        min_length_s = (min_samples - 1) / sampling_rate
        groups = GroupingHelper("instrument").group_stream(
            stream,
            strict=True,
            min_length_s=min_length_s,
            comp_dict=raw_comp_dict,
        )

        feature_stream = obspy.Stream()
        for group in groups:
            converted = self._convert_group_to_features(group, raw_comp_dict)
            if converted is not None:
                feature_stream += converted

        return feature_stream

    @classmethod
    def _feature_default_args(cls):
        return _DKPN_FEATURE_DEFAULTS.copy()

    def _feature_args(self):
        return {key: getattr(self, key) for key in _feature_arg_names}

    def _set_feature_args(self, feature_args):
        unknown = set(feature_args) - set(_feature_arg_names)
        if unknown:
            raise ValueError(f"Unknown DKPN feature arguments: {sorted(unknown)}")

        for key, value in feature_args.items():
            setattr(self, key, value)

    def _convert_group_to_features(self, group, raw_comp_dict):
        traces_by_comp = {}
        for trace in group:
            component = trace.id[-1]
            if component not in raw_comp_dict:
                continue

            comp_idx = raw_comp_dict[component]
            if comp_idx in traces_by_comp:
                seisbench.logger.warning(
                    f"Multiple traces map to DKPN raw component "
                    f"{_raw_component_order[comp_idx]} for "
                    f"{GroupingHelper.trace_id_without_component(trace)}. "
                    f"Using the first one."
                )
                continue

            traces_by_comp[comp_idx] = trace

        if any(comp_idx not in traces_by_comp for comp_idx in range(3)):
            return None

        waveforms, start_time, sampling_rate = self._group_to_array(traces_by_comp)
        extractor = _DKPNFeatureExtractor(**self._feature_args())
        features = extractor.matrix_cfs(waveforms, sampling_rate=sampling_rate)
        stabilization_samples = _dkpn_stabilization_samples(
            self.fp_stabilization, sampling_rate
        )
        features = features[:, stabilization_samples:]

        if features.shape[-1] < self.in_samples:
            seisbench.logger.warning(
                "DKPN preprocessing skipped a waveform group that is too "
                "short after removing the stabilization interval."
            )
            return None

        start_time = start_time + stabilization_samples / sampling_rate

        out = obspy.Stream()
        for idx, component in enumerate(self._feature_component_order):
            if idx < 3:
                stats = traces_by_comp[idx].stats.copy()
            else:
                stats = traces_by_comp[0].stats.copy()
            stats.starttime = start_time
            stats.sampling_rate = sampling_rate
            stats.channel = f"CF{component}"
            trace = Trace(data=features[idx].copy(), header=stats)
            trace.stats.npts = features.shape[-1]
            out += trace

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
        model_args.update(self._feature_args())

        return model_args
