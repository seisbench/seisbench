import logging
from unittest.mock import patch

import numpy as np
import obspy
import pytest
from obspy import UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Station

import seisbench.models


def test_homogenize_station_name():
    assert (
        seisbench.models.depthphase.DepthPhaseModel._homogenize_station_name(
            "NET.STA.LOC.CHA"
        )
        == "NET.STA.LOC"
    )
    assert (
        seisbench.models.depthphase.DepthPhaseModel._homogenize_station_name(
            "NET.STA.LOC"
        )
        == "NET.STA.LOC"
    )
    assert (
        seisbench.models.depthphase.DepthPhaseModel._homogenize_station_name("NET.STA")
        == "NET.STA."
    )
    assert (
        seisbench.models.depthphase.DepthPhaseModel._homogenize_station_name("NET")
        == "NET.."
    )

    with pytest.raises(ValueError):
        seisbench.models.depthphase.DepthPhaseModel._homogenize_station_name(
            "NET.STA.LOC.CHA."
        )


def test_validate_distances(caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        distances = seisbench.models.depthphase.DepthPhaseModel._validate_distances(
            {"A.B.C.D": 0}, None, None
        )
    assert "Distances and station/event positions are provided." not in caplog.text
    assert distances == {"A.B.C": 0}
    with pytest.raises(ValueError):
        seisbench.models.depthphase.DepthPhaseModel._validate_distances(
            None, None, None
        )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        seisbench.models.depthphase.DepthPhaseModel._validate_distances({}, (0, 0), [])
    assert "Distances and station/event positions are provided." in caplog.text


def test_validate_p_picks(caplog):
    p_picks = {"A.B.C.D": 0, "X.Y.Z": 0}
    distances = {"A.B.C": 0}

    with caplog.at_level(logging.WARNING):
        p_picks = seisbench.models.depthphase.DepthPhaseModel._validate_p_picks(
            p_picks, distances
        )

    assert "X.Y.Z" in caplog.text

    assert p_picks == {"A.B.C": 0}


def test_calculate_distances():
    channels = [
        Channel("BHZ", "", latitude=1, longitude=1, elevation=0, depth=0),
        Channel("BHZ", "01", latitude=2, longitude=2, elevation=0, depth=0),
    ]
    stations = [
        Station("A", latitude=1, longitude=1, elevation=0, channels=channels),
        Station("B", latitude=0, longitude=0, elevation=0),
    ]
    net = Network("NE", stations)
    inv = Inventory([net])

    distances = seisbench.models.depthphase.DepthPhaseModel._calculate_distances(
        inv, (0, 0)
    )
    assert sorted(list(distances.keys())) == ["NE.A.", "NE.A.01", "NE.B."]
    assert np.isclose(distances["NE.A."], 1.414177)
    assert np.isclose(distances["NE.A.01"], 2.828140)
    assert np.isclose(distances["NE.B."], 0)


def test_rebase_streams_for_picks():
    t0 = UTCDateTime("2000-01-01")
    picks = {"NE.A.": t0 + 30, "XY.B.01": t0 + 20, "XY.D.01": t0 + 20}

    stream = obspy.Stream()
    for c in "ZNE":
        stream.append(
            obspy.Trace(
                np.ones(10000),
                header={
                    "network": "NE",
                    "station": "A",
                    "location": "",
                    "channel": f"HH{c}",
                    "sampling_rate": 100,
                    "starttime": t0,
                },
            )
        )
        stream.append(
            obspy.Trace(
                np.ones(10000),
                header={
                    "network": "XY",
                    "station": "B",
                    "location": "01",
                    "channel": f"HH{c}",
                    "sampling_rate": 100,
                    "starttime": t0 + 10,
                },
            )
        )
        stream.append(
            obspy.Trace(
                np.ones(10000),
                header={
                    "network": "XY",
                    "station": "C",
                    "location": "01",
                    "channel": f"HH{c}",
                    "sampling_rate": 100,
                    "starttime": t0 + 50,
                },
            )
        )
        stream.append(
            obspy.Trace(
                np.ones(10000),
                header={
                    "network": "XY",
                    "station": "D",
                    "location": "01",
                    "channel": f"HH{c}",
                    "sampling_rate": 100,
                    "starttime": t0 - 1000,
                },
            )
        )

    model = seisbench.models.depthphase.DepthPhaseModel(time_before=10)
    substream = model._rebase_streams_for_picks(stream, picks, in_samples=1500)
    assert len(substream) == 6
    for trace in substream:
        assert trace.stats.npts == 1500


def test_ttlookup(caplog, tmp_path):
    with patch("seisbench.cache_aux_root", tmp_path):
        with caplog.at_level(logging.WARNING):
            model = seisbench.models.depthphase.TTLookup(
                dists=np.linspace(30, 100, 2), depths=np.linspace(5, 660, 2)
            )
        assert "Precalculating travel times." in caplog.text
        model.get_traveltimes(50, 100)

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            model = seisbench.models.depthphase.TTLookup(
                dists=np.linspace(30, 100, 2), depths=np.linspace(5, 660, 2)
            )
        assert "Precalculating travel times." not in caplog.text
        model.get_traveltimes(50, 100)
