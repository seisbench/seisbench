import pickle
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import obspy
import scipy.stats
import torch
from obspy import UTCDateTime
from obspy.geodetics import locations2degrees
from tqdm import tqdm

import seisbench

from .phasenet import PhaseNet
from .team import PhaseTEAM


class DepthPhaseModel:
    """
    Helper class implementing all tools for determining depth from depth phases
    """

    def __init__(
        self,
        time_before: float = 12.5,
        depth_levels: Optional[np.ndarray] = None,
        tt_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.time_before = time_before
        if tt_args is None:
            tt_args = {}
        self._ttlookup = TTLookup(**tt_args)
        if depth_levels is None:
            self.depth_levels = np.linspace(0, 650, 651)
        else:
            self.depth_levels = depth_levels

    def _prepare_classify_args(
        self,
        p_picks: dict[str, UTCDateTime],
        distances: Optional[dict[str, float]],
        inventory: Optional[obspy.Inventory],
        epicenter: Optional[tuple[float, float]],
    ) -> tuple[dict[str, UTCDateTime], dict[str, float]]:
        distances = self._validate_distances(distances, epicenter, inventory)

        if distances is None:
            distances = self._calculate_distances(inventory, epicenter)

        p_picks = self._validate_p_picks(p_picks, distances)

        return p_picks, distances

    @staticmethod
    def _validate_distances(distances, epicenter, inventory):
        if distances is None and (inventory is None or epicenter is None):
            raise ValueError(
                "Either distances of inventory and epicenter need to be defined."
            )
        elif distances is not None and not (inventory is None or epicenter is None):
            seisbench.logger.warning(
                "Distances and station/event positions are provided. "
                "Will ignore station/event positions."
            )

        if distances is not None:
            distances = {
                DepthPhaseModel._homogenize_station_name(k): v
                for k, v in distances.items()
            }
        return distances

    @staticmethod
    def _validate_p_picks(p_picks, distances):
        p_picks = {
            DepthPhaseModel._homogenize_station_name(k): v for k, v in p_picks.items()
        }
        del_keys = []
        for key in p_picks:
            if key not in distances:
                seisbench.logger.warning(
                    f"No distance for '{key}'. Trace will be ignored"
                )
                del_keys.append(key)
            elif distances[key] >= 100:
                seisbench.logger.debug(
                    f"Station '{key}' at distance above 100 degrees will be ignored."
                )
                del_keys.append(key)
        for key in del_keys:
            del p_picks[key]
        return p_picks

    @staticmethod
    def _calculate_distances(
        inventory: obspy.Inventory, epicenter: tuple[float, float]
    ) -> dict[str, float]:
        distances = {}
        for net in inventory:
            for sta in net:
                if len(sta) == 0:
                    # No channel details given, assume empty location code
                    code = f"{net.code}.{sta.code}."
                    distances[code] = locations2degrees(
                        sta.latitude, sta.longitude, *epicenter
                    )
                    continue

                for channel in sta:
                    code = f"{net.code}.{sta.code}.{channel.location_code}"
                    if code in distances:
                        continue
                    distances[code] = locations2degrees(
                        channel.latitude, channel.longitude, *epicenter
                    )

        return distances

    @staticmethod
    def _homogenize_station_name(trace_id: str) -> str:
        """
        Truncates station name to `NET.STA.LOC`
        """
        if trace_id.count(".") == 3:  # Channel given
            return trace_id[: trace_id.rfind(".")]
        elif trace_id.count(".") == 2:  # Correct format
            return trace_id
        elif trace_id.count(".") < 2:  # Add missing parts
            return trace_id + "." * (2 - trace_id.count("."))
        else:
            raise ValueError(f"Could not parse trace id '{trace_id}'")

    def _line_search_depth(
        self,
        annotations: obspy.Stream,
        distances: dict[str, float],
        probability_curves: bool,
    ) -> Union[float, tuple[float, np.ndarray, np.ndarray]]:
        annotations = annotations.slice(
            starttime=UTCDateTime(0)
        )  # Make sure sample 0 is at the P arrival

        probabilities = []
        for station, station_annotations in self._group_traces(annotations).items():
            probabilities.append(
                self._backproject_single_station(
                    station_annotations, distances[station]
                )
            )
        probabilities = np.stack(probabilities, axis=0)

        avg_probabilities = scipy.stats.mstats.gmean(probabilities, axis=0)
        depth = self.depth_levels[np.argmax(avg_probabilities)]

        if probability_curves:
            return depth, self.depth_levels, probabilities
        else:
            return depth

    def _backproject_single_station(
        self, station_annotations: obspy.Stream, dist: float, q_min: float = 0.5
    ):
        """
        Backproject single station

        :param q_min: Quantile to use as lower cutoff for stability
        """
        prob = np.ones_like(self.depth_levels)
        for i, depth in enumerate(self.depth_levels):
            arrivals = self._ttlookup.get_traveltimes(dist, depth)
            for phase in ["pP", "sP"]:
                j = self._ttlookup.phases.index(phase)
                trace = station_annotations.select(channel=f"*_{phase}")[0]
                y_trace = trace.data

                y_trace = self._smooth_curve(y_trace)
                y_trace = self._norm_label(y_trace)

                if not np.isnan(arrivals[j]):
                    sample = int(arrivals[j] * trace.stats.sampling_rate)
                    if sample < y_trace.shape[0]:
                        prob[i] *= max(y_trace[sample], np.quantile(y_trace, q_min))
                    else:
                        prob[i] *= np.quantile(y_trace, q_min)
                else:
                    prob[i] *= np.quantile(y_trace, q_min)

        return prob

    @staticmethod
    def _norm_label(y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Norm label trace to sum to 1.
        """
        y = y + eps / y.shape[-1]
        return y / np.sum(y, axis=-1, keepdims=True)

    @staticmethod
    def _smooth_curve(y: np.ndarray, smoothing: float = 10) -> np.ndarray:
        """
        Smooth curve with Gaussian kernel
        """
        if smoothing == 0:
            return y

        kernel = np.arange(-7 * smoothing, 7 * smoothing + 1, 1)
        kernel = np.exp(-0.5 * (kernel / (2 * smoothing)) ** 2)
        kernel /= np.sum(kernel)

        if y.ndim == 1:
            return np.convolve(y, kernel, "same")
        else:
            y_new = np.zeros_like(y)
            for i in range(y.shape[0]):
                y_new[i] = np.convolve(y[i], kernel, "same")
            return y_new

    @staticmethod
    def _group_traces(annotations: obspy.Stream) -> dict[str, obspy.Stream]:
        grouping = defaultdict(obspy.Stream)
        for trace in annotations:
            key = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"
            grouping[key].append(trace)

        return grouping

    def _rebase_streams_for_picks(
        self, stream: obspy.Stream, p_picks: dict[str, UTCDateTime], in_samples: int
    ) -> obspy.Stream:
        """
        Cuts appropriate segments from the stream. Rebases each trace to have the pick at 0.
        """
        selected_stream = obspy.Stream()

        for trace_id, pick in p_picks.items():
            for trace in stream.select(*trace_id.split(".")):
                trace = trace.slice(starttime=pick - self.time_before).copy()
                trace.data = trace.data[:in_samples]

                if (
                    abs(trace.stats.starttime - pick + self.time_before)
                    > trace.stats.delta
                ) or trace.stats.npts != in_samples:
                    # Trace does not start at correct time or is too short
                    continue

                trace.stats.starttime -= pick
                # trace.stats.starttime = UTCDateTime(0) - self.time_before
                selected_stream.append(trace)

        return selected_stream


class TTLookup:
    def __init__(
        self,
        dists: np.ndarray = np.linspace(1, 100, 50),  # In degrees
        depths: np.ndarray = np.linspace(5, 660, 50),  # In kilometers
        model: str = "iasp91",
        phases: tuple[str] = ("P", "pP", "sP"),
    ):
        assert self._linspaced(dists), "TTLookup requires linearly spaced distances"
        assert self._linspaced(depths), "TTLookup requires linearly spaced depths"

        self.model = model

        self.dists = dists
        self.depths = depths
        self.phases = phases

        self._grid = None

        cache = seisbench.cache_aux_root / "ttlookup" / f"{model}.pkl"

        write_cache = False
        if cache.is_file():
            with open(cache, "rb") as f:
                dists, depths, phases, grid = pickle.load(f)

            if not (
                dists.shape == self.dists.shape
                and depths.shape == self.depths.shape
                and np.allclose(dists, self.dists)
                and np.allclose(depths, self.depths)
                and phases == self.phases
            ):
                seisbench.logger.warning("Traveltime cache invalid. Recalculating.")
                write_cache = True
            else:
                self._grid = grid
        else:
            write_cache = True

        if write_cache:
            self._precalculate()
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump((self.dists, self.depths, self.phases, self._grid), f)

    def _precalculate(self):
        seisbench.logger.warning(
            "Precalculating travel times. This will take a moment. "
            "Results will be cached for future applications."
        )
        from obspy.taup import TauPyModel

        model = TauPyModel(model=self.model)

        self._grid = (
            np.zeros((len(self.depths), len(self.dists), len(self.phases))) * np.nan
        )

        for i, depth in enumerate(tqdm(self.depths, desc="Precalculating traveltimes")):
            for j, dist in enumerate(self.dists):
                arrivals = model.get_travel_times(
                    source_depth_in_km=depth,
                    distance_in_degree=dist,
                    phase_list=self.phases + ("Pdiff",),
                )

                if len(arrivals) == 0:
                    continue

                t0 = arrivals[0].time

                if arrivals[0].phase.name not in ["P", "Pdiff"]:
                    t0 = np.nan

                for k, phase in enumerate(self.phases):
                    for arrival in arrivals:
                        if arrival.phase.name == phase:
                            self._grid[i, j, k] = arrival.time - t0
                            break

    def get_traveltimes(self, dist: float, depth: float):
        frac_dist = (dist - self.dists[0]) / (self.dists[1] - self.dists[0])
        frac_depth = (depth - self.depths[0]) / (self.depths[1] - self.depths[0])

        idx_dist = int(frac_dist)
        idx_depth = int(frac_depth)

        alpha_dist = frac_dist - idx_dist
        alpha_depth = frac_depth - idx_depth

        tt = (
            self._grid[idx_depth, idx_dist] * (1 - alpha_depth) * (1 - alpha_dist)
            + self._grid[idx_depth + 1, idx_dist] * alpha_depth * (1 - alpha_dist)
            + self._grid[idx_depth, idx_dist + 1] * (1 - alpha_depth) * alpha_dist
            + self._grid[idx_depth + 1, idx_dist + 1] * alpha_depth * alpha_dist
        )
        return tt

    @staticmethod
    def _linspaced(x: np.ndarray):
        return np.allclose(x[1] - x[0], x[1:] - x[:-1])


class DepthPhaseNet(PhaseNet, DepthPhaseModel):
    """
    .. document_args:: seisbench.models DepthPhaseNet
    """

    def __init__(
        self,
        phases: str = ("P", "pP", "sP"),
        sampling_rate: float = 20.0,
        depth_phase_args: Optional[dict] = None,
        norm="peak",
        **kwargs,
    ) -> None:
        if depth_phase_args is None:
            depth_phase_args = {}
        PhaseNet.__init__(
            self,
            phases=phases,
            sampling_rate=sampling_rate,
            norm=norm,
            **kwargs,
        )
        DepthPhaseModel.__init__(self, *depth_phase_args)

    def forward(self, x: torch.tensor, logits=False) -> torch.tensor:
        y = super().forward(x, logits=True)
        if logits:
            return y
        else:
            return torch.sigmoid(y)

    def annotate(
        self,
        stream: obspy.Stream,
        parallelism: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "DepthPhaseNet does not implement an annotate function. "
            "Please use the classify function instead."
        )

    def classify(
        self,
        stream: obspy.Stream,
        p_picks: dict[str, UTCDateTime],
        distances: Optional[dict[str, float]] = None,
        inventory: Optional[obspy.Inventory] = None,
        epicenter: Optional[tuple[float, float]] = None,
        probability_curves: bool = False,
        **kwargs,
    ) -> Union[float, tuple[float, np.ndarray, np.ndarray]]:
        """
        Calculate depth of an event using depth phase picking and a line search over the depth axis.
        Can only handle one event at a time.

        For the line search, the epicentral distances of the stations to the event is required.
        These can either be provided directly or through an inventory and the event epicenter.

        :param stream: Obspy stream to classify
        :param p_picks: Dictionary of P pick times. Station codes will be truncated to `NET.STA.LOC`.
        :param distances: Dictionary of epicentral distances for the stations in degrees
        :param inventory: Inventory for the stations
        :param epicenter: (latitude, longitude) of the event epicenter
        :param probability_curves: If true, returns depth_levels and probability curves/otherwise only the depth
        """
        p_picks, distances = self._prepare_classify_args(
            p_picks, distances, inventory, epicenter
        )

        argdict = self.default_args.copy()
        argdict.update(kwargs)

        # Ensure all traces are at the right sampling rate and filtering causes no boundary artifacts
        self.annotate_stream_pre(stream, argdict)
        selected_stream = self._rebase_streams_for_picks(
            stream, p_picks, self.in_samples
        )

        annotations = super().annotate(selected_stream, **kwargs)

        return self._line_search_depth(
            annotations,
            distances,
            probability_curves,
        )


class DepthPhaseTEAM(PhaseTEAM, DepthPhaseModel):
    """
    .. document_args:: seisbench.models DepthPhaseNet
    """

    def __init__(
        self,
        phases: str = ("P", "pP", "sP"),
        classes: int = 3,
        sampling_rate: float = 20.0,
        depth_phase_args: Optional[dict] = None,
        norm="peak",
        **kwargs,
    ) -> None:
        if depth_phase_args is None:
            depth_phase_args = {}
        PhaseTEAM.__init__(
            self,
            phases=phases,
            classes=classes,
            sampling_rate=sampling_rate,
            norm=norm,
            **kwargs,
        )
        DepthPhaseModel.__init__(self, *depth_phase_args)

    def annotate(
        self,
        stream: obspy.Stream,
        parallelism: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "DepthPhaseTEAM does not implement an annotate function. "
            "Please use the classify function instead."
        )

    def classify(
        self,
        stream: obspy.Stream,
        p_picks: dict[str, UTCDateTime],
        distances: Optional[dict[str, float]] = None,
        inventory: Optional[obspy.Inventory] = None,
        epicenter: Optional[tuple[float, float]] = None,
        probability_curves: bool = False,
        **kwargs,
    ) -> Union[float, tuple[float, np.ndarray, np.ndarray]]:
        """
        Calculate depth of an event using depth phase picking and a line search over the depth axis.
        Can only handle one event at a time.

        For the line search, the epicentral distances of the stations to the event is required.
        These can either be provided directly or through an inventory and the event epicenter.

        :param stream: Obspy stream to classify
        :param p_picks: Dictionary of P pick times. Station codes will be truncated to `NET.STA.LOC`.
        :param distances: Dictionary of epicentral distances for the stations in degrees
        :param inventory: Inventory for the stations
        :param epicenter: (latitude, longitude) of the event epicenter
        :param probability_curves: If true, returns depth_levels and probability curves/otherwise only the depth
        """
        p_picks, distances = self._prepare_classify_args(
            p_picks, distances, inventory, epicenter
        )

        argdict = self.default_args.copy()
        argdict.update(kwargs)

        # Ensure all traces are at the right sampling rate and filtering causes no boundary artifacts
        self.annotate_stream_pre(stream, argdict)
        selected_stream = self._rebase_streams_for_picks(
            stream, p_picks, self.in_samples
        )

        annotations = super().annotate(selected_stream, **kwargs)

        return self._line_search_depth(
            annotations,
            distances,
            probability_curves,
        )
