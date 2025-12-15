from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
import obspy

Key = tuple[float, str]


class GroupedTraceData(NamedTuple):
    data: np.ndarray
    stations: list[str]
    component_order: list[str]
    start_time: obspy.UTCDateTime
    sampling_rate: float
    grouping: Literal["instrument", "station", "full"]

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_components(self) -> int:
        return len(self.component_order)

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]


class TraceSegment(NamedTuple):
    data: np.ndarray
    key: Key
    start_time: obspy.UTCDateTime
    window_offset: int
    n_windows: int
    stations: list[str]
    in_samples: int
    pred_sample: tuple[int, int]

    @property
    def n_samples(self) -> int:
        if self.data.ndim == 1:
            return self.data.shape[0]
        return self.data.shape[1]

    @property
    def n_channels(self) -> int:
        if self.data.ndim == 1:
            return 1
        return self.data.shape[0]


class PredictionSegment(TraceSegment):
    @classmethod
    def from_trace_segment(
        cls,
        predictions: np.ndarray,
        segment: TraceSegment,
    ) -> PredictionSegment:
        return cls(
            data=predictions,
            key=segment.key,
            start_time=segment.start_time,
            window_offset=segment.window_offset,
            n_windows=segment.n_windows,
            stations=segment.stations,
            in_samples=segment.in_samples,
            pred_sample=segment.pred_sample,
        )

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]


class PredictionsStacked(NamedTuple):
    data: np.ndarray
    stations: list[str]
    start_time: obspy.UTCDateTime
    sampling_rate: float

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]
