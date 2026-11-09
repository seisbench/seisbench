import asyncio

from obspy import UTCDateTime
from obspy.core.event import (
    Comment,
    Event,
    EventDescription,
    Magnitude,
    Origin,
    ResourceIdentifier,
)

import numpy as np
import pytest

from seisbench.data.dlf import (
    DownloadStats,
    NotQuietError,
    Outcome,
    check_quiet,
    get_event_params,
    is_away_from_events,
    split_events,
)
from seisbench.data.utils import FDSNSource


def test_is_quiet():
    origins = [UTCDateTime(2020, 1, 1, 0, 10).timestamp]
    event = UTCDateTime(2020, 1, 1, 0, 10)

    # Ends exactly 120 s before the event
    assert is_away_from_events(origins, event - 180, event - 120)
    assert not is_away_from_events(origins, event - 179, event - 119)
    # Starts exactly 180 s after the event
    assert is_away_from_events(origins, event + 180, event + 240)
    assert not is_away_from_events(origins, event + 179, event + 239)
    assert not is_away_from_events(origins, event - 30, event + 30)
    assert is_away_from_events([], event, event + 60)


def test_get_event_params():
    origin = Origin(
        time=UTCDateTime(2015, 3, 1),
        latitude=19.2,
        longitude=-155.4,
        depth=40e3,
    )
    magnitude = Magnitude(mag=1.2)
    event = Event(
        resource_id=ResourceIdentifier(
            "smi:local/phasenet_dlf_v4b/hawaii/hawaii2_60234378"
        ),
        origins=[origin],
        magnitudes=[magnitude],
        comments=[Comment(text="split: test"), Comment(text="event_class: DLP")],
        event_descriptions=[
            EventDescription(text="Pahala deep swarm", type="region name")
        ],
    )
    event.preferred_origin_id = origin.resource_id
    event.preferred_magnitude_id = magnitude.resource_id

    params = get_event_params(event, "hawaii")
    assert params["source_id"] == "hawaii:hawaii2_60234378"
    # The split is drawn per catalog, not taken from the comment
    assert "split" not in params
    assert params["source_event_class"] == "DLP"
    assert params["source_volcano"] == "Pahala deep swarm"
    assert params["source_region"] == "hawaii"
    assert params["source_depth_km"] == 40.0
    assert params["source_magnitude"] == 1.2


def test_fdsn_source_networks():
    source = FDSNSource("https://example.org", networks=["NC", "BK"])

    async def get_stations(networks):
        # Returns before any request if the source serves none of the networks
        return await source.get_stations(
            None, "STA", UTCDateTime(0), UTCDateTime(1), "station", networks
        )

    assert len(asyncio.run(get_stations({"AV", "HV"}))) == 0


def test_split_events():
    rng = np.random.default_rng(0)
    # 100 isolated events, one per day, and a swarm of 20 events within an hour
    times = [i * 86400.0 for i in range(100)] + [
        50.0 * 86400 + 100 + i for i in range(20)
    ]
    splits = split_events(times, (0.7, 0.1, 0.2), rng)

    assert len(set(splits[100:])) == 1  # The swarm stays in one split
    counts = {split: splits.count(split) for split in ("train", "dev", "test")}
    assert abs(counts["train"] - 84) <= 20
    assert abs(counts["dev"] - 12) <= 1
    assert abs(counts["test"] - 24) <= 1


def test_download_stats(tmp_path):
    outcomes = [
        Outcome("event", "hawaii", "hawaii:1", "a", "downloaded"),
        Outcome("event", "hawaii", "hawaii:1", "b", "failed", "FDSNNoDataException"),
        Outcome("noise", "hawaii", "hawaii:1", "noise_a", "near_event"),
        Outcome("event", "eifel_qseek", "eifel_qseek:2", "c", "no_source"),
    ]
    stats = DownloadStats.from_outcomes({"hawaii": 1, "eifel_qseek": 1}, outcomes)

    hawaii, eifel = stats.regions
    assert (hawaii.events_downloaded, hawaii.traces, hawaii.traces_failed) == (1, 2, 1)
    assert hawaii.noise_near_event == 1
    assert hawaii.failure_reasons == {"FDSNNoDataException": 1}
    assert (eifel.events_downloaded, eifel.traces_no_source) == (0, 1)
    assert (stats.total.events, stats.total.traces) == (2, 3)
    assert len(stats.failures) == 3
    assert "hawaii: 1/1 events, 1/2 traces downloaded" in str(stats)

    stats.write(tmp_path)
    assert (tmp_path / "download_stats.json").exists()
    assert len((tmp_path / "download_failures.csv").read_text().splitlines()) == 4


def test_check_quiet():
    rng = np.random.default_rng(0)
    sampling_rate = 100.0
    window_start = 3000  # NOISE_LEAD of 30 s
    noise = rng.normal(size=(3, 9001))

    check_quiet(noise, sampling_rate, 1.0, window_start)

    # A component without data, e.g., a vertical-only station, is fine
    z_only = noise.copy()
    z_only[1:] = 0
    check_quiet(z_only, sampling_rate, 1 / 3, window_start)

    # An arrival in the window
    burst = noise.copy()
    burst[0, 6000:6200] *= 20
    with pytest.raises(NotQuietError, match="STA/LTA"):
        check_quiet(burst, sampling_rate, 1.0, window_start)

    # An arrival before the window is not checked
    early = noise.copy()
    early[0, 500:700] *= 20
    check_quiet(early, sampling_rate, 1.0, window_start)

    with pytest.raises(NotQuietError, match="gaps"):
        check_quiet(noise, sampling_rate, 0.9, window_start)
    with pytest.raises(NotQuietError, match="too late"):
        check_quiet(noise, sampling_rate, 1.0, 1000)
