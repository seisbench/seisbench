import logging
from unittest.mock import patch

import numpy as np
import obspy
import pytest
import torch
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
    distances = {"A.B.C": 20}

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


def test_rebase_streams_for_picks(tmp_path):
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

    with patch("seisbench.cache_aux_root", tmp_path):
        model = seisbench.models.depthphase.DepthPhaseModel(
            time_before=10,
            tt_args=dict(dists=np.linspace(30, 100, 2), depths=np.linspace(5, 660, 2)),
        )
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


def test_group_traces():
    t0 = UTCDateTime("2000-01-01")
    stream = obspy.Stream()
    for c in "ZNE":
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

    for c in "ZN":
        stream.append(
            obspy.Trace(
                np.ones(10000),
                header={
                    "network": "XY",
                    "station": "E",
                    "location": "",
                    "channel": f"HH{c}",
                    "sampling_rate": 100,
                    "starttime": t0 - 1000,
                },
            )
        )

    grouped = seisbench.models.depthphase.DepthPhaseModel._group_traces(stream)
    assert len(grouped) == 2
    assert len(grouped["XY.D.01"]) == 3
    assert len(grouped["XY.E."]) == 2


def test_smooth_curve():
    for x in [np.random.rand(1000), np.random.rand(3, 1000)]:
        # Avoid boundary artifacts
        x[..., :150] = 0
        x[..., -150:] = 0

        y = seisbench.models.depthphase.DepthPhaseModel._smooth_curve(x, smoothing=10)
        assert np.allclose(
            np.sum(x, axis=-1), np.sum(y, axis=-1)
        )  # Integral stays constant
        assert not np.allclose(x, y)  # Values are changed


def test_norm_label():
    x = np.random.rand(3, 1000)
    y = seisbench.models.depthphase.DepthPhaseModel._norm_label(x, eps=1e-10)

    assert np.allclose(np.sum(y, axis=-1), 1)


def test_backproject_single_station(tmp_path):
    with patch("seisbench.cache_aux_root", tmp_path):
        model = seisbench.models.depthphase.DepthPhaseModel(
            depth_levels=np.linspace(10, 500, 20),
            tt_args=dict(dists=np.linspace(30, 100, 2), depths=np.linspace(5, 660, 2)),
        )
        t0 = UTCDateTime(0)

        annotations = obspy.Stream(
            [
                obspy.Trace(
                    np.ones(10000),
                    header={
                        "network": "XY",
                        "station": "E",
                        "location": "",
                        "channel": "model_pP",
                        "sampling_rate": 100,
                        "starttime": t0 - 10,
                    },
                ),
                obspy.Trace(
                    np.ones(10000),
                    header={
                        "network": "XY",
                        "station": "E",
                        "location": "",
                        "channel": "model_sP",
                        "sampling_rate": 100,
                        "starttime": t0 - 10,
                    },
                ),
            ]
        )

        pred = model._backproject_single_station(annotations, dist=50)
        assert len(pred) == len(model.depth_levels)


def test_line_search_depth(tmp_path):
    with patch("seisbench.cache_aux_root", tmp_path):
        model = seisbench.models.depthphase.DepthPhaseModel(
            depth_levels=np.linspace(10, 500, 20),
            tt_args=dict(dists=np.linspace(30, 100, 2), depths=np.linspace(5, 660, 2)),
        )
        t0 = UTCDateTime(0)

        annotations = obspy.Stream()
        for station in "DE":
            annotations += obspy.Stream(
                [
                    obspy.Trace(
                        np.random.rand(10000),
                        header={
                            "network": "XY",
                            "station": station,
                            "location": "",
                            "channel": "model_pP",
                            "sampling_rate": 100,
                            "starttime": t0 - 10,
                        },
                    ),
                    obspy.Trace(
                        np.random.rand(10000),
                        header={
                            "network": "XY",
                            "station": station,
                            "location": "",
                            "channel": "model_sP",
                            "sampling_rate": 100,
                            "starttime": t0 - 10,
                        },
                    ),
                ]
            )

        distances = {
            "XY.D.": 50,
            "XY.E.": 10,
        }

        output = model._line_search_depth(annotations, distances, "")
        assert output.probabilities.shape == (2, output.depth_levels.shape[0])
        assert output.depth == output.depth_levels[np.argmax(output.avg_probabilities)]


def test_depthphaseteam():
    model = seisbench.models.DepthPhaseTEAM(classes=4)

    x = torch.rand(2, 10, 3, 3001)
    y = model(x)
    assert y.shape == (2, 10, 4, 3001)


def test_depthphasemodel_qc(tmp_path):
    with patch("seisbench.cache_aux_root", tmp_path):
        model = seisbench.models.depthphase.DepthPhaseModel(
            time_before=10,
            tt_args=dict(dists=np.linspace(30, 100, 2), depths=np.linspace(5, 660, 2)),
            qc_std=20,
            qc_depth=200,
        )
        prob = np.zeros_like(model.depth_levels)
        prob[0] = 1
        assert model._qc_prediction(prob, 100.0) == 100.0
        prob[-1] = 1  # Make sure STD is high
        assert model._qc_prediction(prob, 300.0) == 300.0  # Deeper than qc_depth
        assert np.isnan(model._qc_prediction(prob, 100.0))


# @pytest.mark.slow  # Test is slow and depends on SeisBench repository and FDSN web service
@pytest.mark.skip(reason="The dependency on the GFZ webservice causes flaky tests.")
@patch("seisbench.__version__", "0.5.0")  #
def test_depth_finder():
    networks = {"GFZ": ["GE"]}
    depth_model = seisbench.models.DepthPhaseTEAM.from_pretrained("original")
    phase_model = seisbench.models.PhaseNet.from_pretrained("geofon")
    depth_finder = seisbench.models.DepthFinder(networks, depth_model, phase_model)

    lat, lon, depth, org_time = (
        24.631,
        121.703,
        52.8,
        UTCDateTime("1995-12-01T03:17:04.490000Z"),
    )

    output = depth_finder.get_depth(lat, lon, depth, org_time)
    assert isinstance(output.depth, float)
