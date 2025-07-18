import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import obspy
import scipy.stats
import torch
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from tqdm import tqdm

import seisbench
import seisbench.util as sbu

from .base import WaveformModel
from .phasenet import PhaseNet
from .team import PhaseTEAM


class DepthPhaseModel:
    """
    Helper class implementing all tools for determining depth from depth phases

    :param time_before: Time included before the P pick in seconds
    :param depth_levels: Array of depth levels to search for
    :param tt_args: Arguments for the :py:class:`TTLookup`
    :param qc_std: Maximum standard deviation to pass quality control.
                   If None, no quality control is applied.
    :param qc_depth: Quality control is only applied to predictions shallower than this depth.
                     If None, quality control is applied to all depth levels.
    """

    def __init__(
        self,
        time_before: float = 12.5,
        depth_levels: Optional[np.ndarray] = None,
        tt_args: Optional[dict[str, Any]] = None,
        qc_std: Optional[float] = None,
        qc_depth: Optional[float] = None,
    ) -> None:
        self.time_before = time_before
        if tt_args is None:
            tt_args = {}
        self._ttlookup = TTLookup(**tt_args)

        self.qc_std = qc_std
        self.qc_depth = qc_depth

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
            elif not (15 < distances[key] < 100):
                seisbench.logger.debug(
                    f"Station '{key}' at distance outside 15 to 100 degrees will be ignored."
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
        self, annotations: obspy.Stream, distances: dict[str, float], caller_name: str
    ) -> sbu.ClassifyOutput:
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

        if len(probabilities) == 0:
            sbu.ClassifyOutput(
                caller_name,
                depth=np.nan,
                depth_levels=self.depth_levels,
                probabilities=np.nan * self.depth_levels,
                avg_probabilities=np.nan * self.depth_levels,
            )

        probabilities = np.stack(probabilities, axis=0)

        avg_probabilities = scipy.stats.mstats.gmean(
            probabilities, nan_policy="omit", axis=0
        )

        depth = self.depth_levels[np.argmax(avg_probabilities)]

        depth = self._qc_prediction(avg_probabilities, depth)

        return sbu.ClassifyOutput(
            caller_name,
            depth=depth,
            depth_levels=self.depth_levels,
            probabilities=probabilities,
            avg_probabilities=avg_probabilities,
        )

    def _qc_prediction(self, prob: np.ndarray, depth: float) -> float:
        normed_prob = self._norm_label(prob, eps=1e-12)
        mean = np.sum(normed_prob * self.depth_levels)
        var = np.sum(normed_prob * ((self.depth_levels - mean) ** 2))
        std = np.sqrt(var)

        if self.qc_std is not None:
            if std > self.qc_std and (self.qc_depth is None or depth < self.qc_depth):
                seisbench.logger.warning(
                    f"Standard deviation ({std:.1f} km) above quality control "
                    f"limit ({self.qc_std:.1f} km). Returning NaN. You can increase "
                    f"qc_std to get a depth estimate nonetheless, but the result is "
                    f"likely unreliable."
                )
                return np.nan

        return depth

    def _backproject_single_station(
        self,
        station_annotations: obspy.Stream,
        dist: float,
        q_min: float = 0.5,
        truncate: int = 100,
    ):
        """
        Backproject single station

        :param q_min: Quantile to use as lower cutoff for stability
        :param truncate: Number of samples truncated at the end for stability
        """
        prob = np.ones_like(self.depth_levels)
        has_phases = np.zeros(
            len(self.depth_levels), dtype=bool
        )  # Log where at least one phase value was available
        for i, depth in enumerate(self.depth_levels):
            arrivals = self._ttlookup.get_traveltimes(dist, depth)
            for phase in ["pP", "sP"]:
                j = self._ttlookup.phases.index(phase)
                trace = station_annotations.select(channel=f"*_{phase}")[0]
                y_trace = trace.data[:-truncate]

                y_trace = self._smooth_curve(y_trace)
                y_trace = self._norm_label(y_trace)

                if not np.isnan(arrivals[j]):
                    sample = int(arrivals[j] * trace.stats.sampling_rate)
                    if sample < y_trace.shape[0]:
                        prob[i] *= max(y_trace[sample], np.quantile(y_trace, q_min))
                        has_phases[i] = True
                    else:
                        prob[i] *= np.quantile(y_trace, q_min)
                else:
                    prob[i] *= np.quantile(y_trace, q_min)

        prob[~has_phases] = np.nan  # Set all values without any phase to nan

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


class DepthFinder:
    """
    This class is a high-level interface to the depth phase models.
    It determines event depth at teleseismic distances based on a preliminary location.
    In contrast to the depth phase models, it is not provided with waveforms,
    but automatically downloads data through FDSN.
    Furthermore, it automatically determines first P arrivals using predicted travel times
    and a deep learning picker.

    The processing consists of several steps:

    - determine available station at the time of the event
    - predict P arrivals
    - download waveforms through FDSN
    - repick P arrivals with a deep learning model
    - determine depth with deep learning based depth model

    If waveforms and P wave picks are already available, it is highly recommended to directly use
    the underlying depth phase model instead of this helper.

    .. code-block:: python
        :caption: Example application

        networks = {"GFZ": ["GE"], "IRIS": ["II", "IU"]}  # FDSN providers and networks
        depth_model = sbm.DepthPhaseTEAM.from_pretrained("original")  # A depth phase model
        phase_model = sbm.PhaseNet.from_pretrained("geofon")  # A teleseismic picking model
        depth_finder = DepthFinder(networks, depth_model, phase_model)

    :param networks: Dictionary of FDSN providers and seismic network codes to query
    :param depth_model: The depth phase model to use
    :param phase_model: The phase picking model to use for pick refinement
    :param p_window: Seconds around the predicted P arrival to search for actual arrival
    :param p_threshold: Minimum detection confidence for the primary P phase to include a record
    """

    def __init__(
        self,
        networks: dict[str, list[str]],
        depth_model: DepthPhaseModel,
        phase_model: WaveformModel,
        p_window: float = 10,
        p_threshold: float = 0.15,
    ):
        self.networks = networks
        self.depth_model = depth_model
        self.phase_model = phase_model
        self.p_window = p_window
        self.p_threshold = p_threshold
        self.cache: Optional[Path] = (
            None  # If set, cache waveforms at this path and try loading them here
        )

        self.tt_model = TauPyModel(model="iasp91")

        self._clients = {provider: Client(provider) for provider in networks.keys()}
        self._network_to_provider = {
            net: provider for provider, nets in networks.items() for net in nets
        }

        self._setup_inventories()

    def _setup_inventories(self):
        self._inventories = {}
        for provider, networks in self.networks.items():
            for network in networks:
                seisbench.logger.debug(
                    f"Querying inventory for {network} from {provider}"
                )
                self._inventories[network] = self._clients[provider].get_stations(
                    network=network, channel="BH?", level="CHANNEL"
                )

    def _get_stations(self, time: UTCDateTime) -> obspy.Inventory:
        stations = obspy.Inventory()
        for network, inv in self._inventories.items():
            stations += inv.select(time=time)

        return stations

    def get_depth(
        self,
        lat: float,
        lon: float,
        depth: float,
        origin_time: UTCDateTime,
    ) -> sbu.ClassifyOutput:
        """
        Get the depth of an event based on its preliminary latitude, longitude, depth and origin time.
        A depth estimate needs to be input, as it is required to predict preliminary P arrivals.
        This is not a circular reasoning, as depth and origin_time trade off against each other.

        :param lat: Latitude of the event
        :param lon: Longitude of the event
        :param depth: Preliminary depth of the event
        :param origin_time: Preliminary origin time of the event
        """
        stations = self._get_stations(origin_time)

        distances = self.depth_model._calculate_distances(stations, (lat, lon))
        distances = {key: val for key, val in distances.items() if 15 < val < 100}

        p_picks_tt = self._get_picks_tt(origin_time, depth, distances)

        stream = self._get_cache(lat, lon, depth, origin_time)
        if stream is None:
            stream = self._get_waveforms(p_picks_tt)
        self._set_cache(stream, lat, lon, depth, origin_time)

        p_picks = self._repick_dl(p_picks_tt, stream)

        seisbench.logger.debug("Calculating depth")
        classify_outputs = self.depth_model.classify(stream, p_picks, distances)

        return sbu.ClassifyOutput(
            self.__class__.__name__,
            depth=classify_outputs.depth,
            depth_levels=classify_outputs.depth_levels,
            probabilities=classify_outputs.probabilities,
            avg_probabilities=classify_outputs.avg_probabilities,
            p_picks=p_picks,
            p_picks_tt=p_picks_tt,
            distances=distances,
            stream=stream,
        )

    def _get_picks_tt(
        self, origin_time: UTCDateTime, depth: float, distances: dict[str, float]
    ):
        seisbench.logger.debug("Calculating traveltimes")
        p_picks = {}
        for station, dist in distances.items():
            # Assume all station are at 0 km elevation. Error is small enough to be fixed by repicker.
            tt = self._get_traveltime(dist, depth)
            if not np.isnan(tt):
                p_picks[station] = origin_time + tt
        return p_picks

    def _get_traveltime(self, dist_deg: float, source_depth_km: float) -> float:
        arrivals = self.tt_model.get_travel_times(
            source_depth_in_km=source_depth_km,
            distance_in_degree=dist_deg,
            phase_list=["p", "P"],
        )

        if len(arrivals) > 0:
            return arrivals[0].time
        else:
            return np.nan

    def _get_cache(
        self, lat: float, lon: float, depth: float, origin_time: UTCDateTime
    ) -> Optional[obspy.Stream]:
        if self.cache is None:
            return

        ev_cache = self.cache / self._get_event_key(lat, lon, depth, origin_time)
        if ev_cache.is_file():
            if ev_cache.stat().st_size == 0:  # Empty token file
                return obspy.Stream()
            else:
                return obspy.read(ev_cache)

    def _set_cache(
        self,
        stream: obspy.Stream,
        lat: float,
        lon: float,
        depth: float,
        origin_time: UTCDateTime,
    ) -> None:
        if self.cache is None:
            return

        ev_cache = self.cache / self._get_event_key(lat, lon, depth, origin_time)
        if not ev_cache.is_file():
            if len(stream) > 0:
                stream.write(ev_cache)
            else:
                with open(ev_cache, "w"):
                    pass  # Create empty token file

    def _get_event_key(
        self, lat: float, lon: float, depth: float, origin_time: UTCDateTime
    ) -> str:
        return f"{origin_time}__{lat:.3f}__{lon:.3f}__{depth:.2f}.mseed"

    def _get_waveforms(
        self,
        p_picks: dict[str, UTCDateTime],
        time_before: float = 100,
        time_after: float = 300,
    ) -> obspy.Stream:
        bulks = {provider: [] for provider in self.networks.keys()}
        for station, pick in p_picks.items():
            net, sta, loc = station.split(".")
            provider = self._network_to_provider[net]
            bulks[provider].append(
                (net, sta, loc, "BH?", pick - time_before, pick + time_after)
            )

        stream = obspy.Stream()
        for provider, bulk in bulks.items():
            seisbench.logger.debug(f"Querying {provider}")
            stream += sbu.fdsn_get_bulk_safe(self._clients[provider], bulk)

        return stream

    def _repick_dl(
        self, p_picks: dict[str, UTCDateTime], stream: obspy.Stream
    ) -> dict[str, UTCDateTime]:
        seisbench.logger.debug("Repicking")
        ann = self.phase_model.annotate(stream).select(channel="*_P")

        refined_p_picks = {}

        for station, pick in p_picks.items():
            net, sta, loc = station.split(".")
            station_ann = ann.select(network=net, station=sta, location=loc)
            station_ann = station_ann.slice(pick - self.p_window, pick + self.p_window)

            if len(station_ann) != 1:
                continue
            station_ann = station_ann[0]

            if np.max(station_ann.data) < self.p_threshold:
                continue

            refined_p_picks[station] = (
                station_ann.stats.starttime
                + station_ann.times()[np.argmax(station_ann.data)]
            )

        return refined_p_picks


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
        DepthPhaseModel.__init__(self, **depth_phase_args)

        self._citation = (
            "Münchmeyer, J., Saul, J. & Tilmann, F. (2023) "
            "Learning the deep and the shallow: Deep learning "
            "based depth phase picking and earthquake depth estimation."
            "Seismological Research Letters (in revision)."
        )

    def forward(self, x: torch.tensor, logits=False) -> torch.tensor:
        y = super().forward(x, logits=True)
        if logits:
            return y
        else:
            return torch.sigmoid(y)

    def annotate(
        self,
        stream: obspy.Stream,
        p_picks: dict[str, UTCDateTime],
        **kwargs,
    ):
        """
        Get depth phase probabilities curves for one event.
        Note that the annotations are aligned to have the P arrival at UTCDateTime(0), i.e.,
        1970-01-01 00:00:00.
        The probability curves are not normalized, there absolute value is therefore meaningless.

        .. warning::
            This class does not expose an 'annotate_async` function directly.

        :param stream: Obspy stream to annotate
        :param p_picks: Dictionary of P pick times. Station codes will be truncated to `NET.STA.LOC`.
        :param kwargs: All kwargs are passed to the annotate function of the superclass.
        """
        p_picks = {
            DepthPhaseModel._homogenize_station_name(k): v for k, v in p_picks.items()
        }

        argdict = self.default_args.copy()
        argdict.update(kwargs)

        # Ensure all traces are at the right sampling rate and filtering causes no boundary artifacts
        self.annotate_stream_pre(stream, argdict)
        selected_stream = self._rebase_streams_for_picks(
            stream, p_picks, self.in_samples
        )

        return super().annotate(selected_stream, **kwargs)

    def classify(
        self,
        stream: obspy.Stream,
        p_picks: dict[str, UTCDateTime],
        distances: Optional[dict[str, float]] = None,
        inventory: Optional[obspy.Inventory] = None,
        epicenter: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> sbu.ClassifyOutput:
        """
        Calculate depth of an event using depth phase picking and a line search over the depth axis.
        Can only handle one event at a time.

        For the line search, the epicentral distances of the stations to the event is required.
        These can either be provided directly or through an inventory and the event epicenter.

        .. warning::
            This class does not expose an 'classify_async` function directly.

        :param stream: Obspy stream to classify
        :param p_picks: Dictionary of P pick times. Station codes will be truncated to `NET.STA.LOC`.
        :param distances: Dictionary of epicentral distances for the stations in degrees
        :param inventory: Inventory for the stations
        :param epicenter: (latitude, longitude) of the event epicenter
        """
        p_picks, distances = self._prepare_classify_args(
            p_picks, distances, inventory, epicenter
        )

        annotations = self.annotate(stream, p_picks, **kwargs)

        return self._line_search_depth(annotations, distances, self.name)


class DepthPhaseTEAM(PhaseTEAM, DepthPhaseModel):
    """
    .. document_args:: seisbench.models DepthPhaseTEAM
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
        DepthPhaseModel.__init__(self, **depth_phase_args)

        self._citation = (
            "Münchmeyer, J., Saul, J. & Tilmann, F. (2023) "
            "Learning the deep and the shallow: Deep learning "
            "based depth phase picking and earthquake depth estimation."
            "Seismological Research Letters (in revision)."
        )

    def annotate(
        self,
        stream: obspy.Stream,
        p_picks: dict[str, UTCDateTime],
        **kwargs,
    ):
        """
        Get depth phase probabilities curves for one event.
        Note that the annotations are aligned to have the P arrival at UTCDateTime(0), i.e.,
        1970-01-01 00:00:00.
        The probability curves are not normalized, there absolute value is therefore meaningless.

        :param stream: Obspy stream to annotate
        :param p_picks: Dictionary of P pick times. Station codes will be truncated to `NET.STA.LOC`.
        :param kwargs: All kwargs are passed to the annotate function of the superclass.
        """
        p_picks = {
            DepthPhaseModel._homogenize_station_name(k): v for k, v in p_picks.items()
        }

        argdict = self.default_args.copy()
        argdict.update(kwargs)

        # Ensure all traces are at the right sampling rate and filtering causes no boundary artifacts
        self.annotate_stream_pre(stream, argdict)
        selected_stream = self._rebase_streams_for_picks(
            stream, p_picks, self.in_samples
        )

        return super().annotate(selected_stream, **kwargs)

    def classify(
        self,
        stream: obspy.Stream,
        p_picks: dict[str, UTCDateTime],
        distances: Optional[dict[str, float]] = None,
        inventory: Optional[obspy.Inventory] = None,
        epicenter: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> sbu.ClassifyOutput:
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
        """
        p_picks, distances = self._prepare_classify_args(
            p_picks, distances, inventory, epicenter
        )

        annotations = self.annotate(stream, p_picks, **kwargs)

        return self._line_search_depth(annotations, distances, self.name)
