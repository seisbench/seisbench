import seisbench
import seisbench.models
from seisbench.models.base import ActivationLSTMCell, CustomLSTM

import numpy as np
import obspy
from obspy import UTCDateTime
import torch
from unittest.mock import patch
import logging
import pytest
import inspect
from collections import defaultdict
from pathlib import Path


def get_input_args(obj):
    signature = inspect.signature(obj)
    args = {k: v._default for k, v in signature.parameters.items()}
    if "kwargs" in args.keys():
        del args["kwargs"]
    return args


def test_weights_docstring():
    model = seisbench.models.GPD()
    assert model.weights_docstring is None

    model = seisbench.models.GPD.from_pretrained("dummy")
    assert isinstance(model.weights_docstring, str)


def test_sanitize_mismatching_records(caplog):
    t0 = UTCDateTime(0)
    stats = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHZ",
        "sampling_rate": 20,
        "starttime": t0,
    }

    trace0 = obspy.Trace(np.zeros(1000), stats)
    trace1 = obspy.Trace(np.ones(1000), stats)
    trace2 = obspy.Trace(2 * np.ones(1000), stats)

    # Overlapping matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 10), trace0.slice(t0 + 5, t0 + 15)])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 1
        )
    assert caplog.text == ""

    # Overlapping not matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 10), trace1.slice(t0 + 5, t0 + 15)])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 0
        )
    assert "All mismatching traces will be ignored." in caplog.text

    # Full intersection matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 15), trace0.slice(t0 + 5, t0 + 10)])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 1
        )
    assert caplog.text == ""

    # Full intersection not matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 15), trace1.slice(t0 + 5, t0 + 10)])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 0
        )
    assert "All mismatching traces will be ignored." in caplog.text

    # No intersection same values
    stream = obspy.Stream([trace0.slice(t0, t0 + 10), trace0.slice(t0 + 15, t0 + 20)])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 2
        )
    assert caplog.text == ""

    # No intersection different values
    stream = obspy.Stream([trace0.slice(t0, t0 + 10), trace1.slice(t0 + 15, t0 + 20)])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 2
        )
    assert caplog.text == ""

    # Three traces matching
    stream = obspy.Stream(
        [
            trace0.slice(t0, t0 + 10),
            trace0.slice(t0 + 5, t0 + 15),
            trace0.slice(t0 + 7, t0 + 11),
        ]
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 1
        )
    assert caplog.text == ""

    # Three traces not matching
    stream = obspy.Stream(
        [
            trace0.slice(t0, t0 + 10),
            trace0.slice(t0 + 5, t0 + 15),
            trace1.slice(t0 + 7, t0 + 11),
        ]
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 0
        )
    assert "All mismatching traces will be ignored." in caplog.text

    # Three traces not matching
    stream = obspy.Stream(
        [
            trace0.slice(t0, t0 + 10),
            trace1.slice(t0 + 2, t0 + 4),
            trace2.slice(t0 + 5, t0 + 11),
        ]
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 0
        )
    assert "All mismatching traces will be ignored." in caplog.text

    # Three traces - two mismatching, one independent
    stream = obspy.Stream(
        [
            trace0.slice(t0, t0 + 10),
            trace1.slice(t0 + 2, t0 + 4),
            trace2.slice(t0 + 11, t0 + 15),
        ]
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            len(
                seisbench.models.WaveformModel.sanitize_mismatching_overlapping_records(
                    stream
                )
            )
            == 1
        )
    assert "All mismatching traces will be ignored." in caplog.text


class DummyWaveformModel(seisbench.models.WaveformModel):
    def annotate(self, stream, *args, **kwargs):
        pass

    def classify(self, stream, *args, **kwargs):
        pass

    @property
    def device(self):
        return "cpu"


def test_stream_to_arrays_instrument():
    t0 = UTCDateTime(0)
    stats_z = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHZ",
        "sampling_rate": 100,
        "starttime": t0,
    }
    stats_n = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHN",
        "sampling_rate": 100,
        "starttime": t0,
    }
    stats_e = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHE",
        "sampling_rate": 100,
        "starttime": t0,
    }
    dummy = DummyWaveformModel(component_order="ZNE")

    trace_z = obspy.Trace(np.ones(1000), stats_z)
    trace_n = obspy.Trace(2 * np.ones(1000), stats_n)
    trace_e = obspy.Trace(3 * np.ones(1000), stats_e)

    # Aligned strict
    stream = obspy.Stream([trace_z, trace_n, trace_e])
    times, data = dummy.stream_to_arrays(stream, strict=True)
    assert len(times) == len(data) == 1
    assert times[0] == t0
    assert data[0].shape == (3, len(trace_z.data))
    assert (data[0][0] == trace_z.data).all()
    assert (data[0][1] == trace_n.data).all()
    assert (data[0][2] == trace_e.data).all()

    # Aligned non strict
    stream = obspy.Stream([trace_z, trace_n, trace_e])
    times, data = dummy.stream_to_arrays(stream, strict=False)
    assert len(times) == len(data) == 1
    assert times[0] == t0
    assert data[0].shape == (3, len(trace_z.data))
    assert (data[0][0] == trace_z.data).all()
    assert (data[0][1] == trace_n.data).all()
    assert (data[0][2] == trace_e.data).all()

    # Covering strict
    stream = obspy.Stream([trace_z, trace_n, trace_e.slice(t0 + 1, t0 + 5)])
    times, data = dummy.stream_to_arrays(stream, strict=True)
    assert len(times) == len(data) == 1
    assert times[0] == t0 + 1
    assert data[0].shape == (3, 401)
    assert (data[0][0] == trace_z.data[0]).all()
    assert (data[0][1] == trace_n.data[0]).all()
    assert (data[0][2] == trace_e.data[0]).all()

    # Covering non-strict
    stream = obspy.Stream([trace_z, trace_n, trace_e.slice(t0 + 1, t0 + 5)])
    times, data = dummy.stream_to_arrays(stream, strict=False)
    assert len(times) == len(data) == 1
    assert times[0] == t0
    assert data[0].shape == (3, len(trace_z.data))
    assert (data[0][0] == trace_z.data[0]).all()
    assert (data[0][1] == trace_n.data[0]).all()
    assert (data[0][2, 100:501] == trace_e.data[0]).all()
    assert (data[0][2, :100] == 0).all()
    assert (data[0][2, 501:] == 0).all()

    # Double covering strict
    stream = obspy.Stream(
        [trace_z, trace_n, trace_e.slice(t0 + 1, t0 + 5), trace_e.slice(t0 + 6, t0 + 8)]
    )
    times, data = dummy.stream_to_arrays(stream, strict=True)
    assert len(times) == len(data) == 2
    assert times[0] == t0 + 1
    assert times[1] == t0 + 6
    assert data[0].shape == (3, 401)
    assert data[1].shape == (3, 201)
    for i in range(2):
        assert (data[i][0] == trace_z.data[0]).all()
        assert (data[i][1] == trace_n.data[0]).all()
        assert (data[i][2] == trace_e.data[0]).all()

    # Double covering non strict
    stream = obspy.Stream(
        [trace_z, trace_n, trace_e.slice(t0 + 1, t0 + 5), trace_e.slice(t0 + 6, t0 + 8)]
    )
    times, data = dummy.stream_to_arrays(stream, strict=False)
    assert len(times) == len(data) == 1
    assert times[0] == t0
    assert data[0].shape == (3, len(trace_z.data))
    assert (data[0][0] == trace_z.data[0]).all()
    assert (data[0][1] == trace_n.data[0]).all()
    assert (data[0][2, 100:501] == trace_e.data[0]).all()
    assert (data[0][2, 600:801] == trace_e.data[0]).all()
    assert (data[0][2, :100] == 0).all()
    assert (data[0][2, 501:600] == 0).all()
    assert (data[0][2, 801:] == 0).all()

    # Intersecting strict
    stream = obspy.Stream(
        [trace_z, trace_n.slice(t0 + 1, t0 + 5), trace_e.slice(t0 + 3, t0 + 7)]
    )
    times, data = dummy.stream_to_arrays(stream, strict=True)
    assert len(times) == len(data) == 1
    assert times[0] == t0 + 3
    assert data[0].shape == (3, 201)
    assert (data[0][0] == trace_z.data[0]).all()
    assert (data[0][1] == trace_n.data[0]).all()
    assert (data[0][2] == trace_e.data[0]).all()

    # Intersecting non strict
    stream = obspy.Stream(
        [trace_z, trace_n.slice(t0 + 1, t0 + 5), trace_e.slice(t0 + 3, t0 + 7)]
    )
    times, data = dummy.stream_to_arrays(stream, strict=False)
    assert len(times) == len(data) == 1
    assert times[0] == t0
    assert data[0].shape == (3, len(trace_z.data))
    assert (data[0][0] == trace_z.data[0]).all()
    assert (data[0][1, 100:501] == trace_n.data[0]).all()
    assert (data[0][1, :100] == 0).all()
    assert (data[0][1, 501:] == 0).all()
    assert (data[0][2, 300:701] == trace_e.data[0]).all()
    assert (data[0][2, :300] == 0).all()
    assert (data[0][2, 701:] == 0).all()

    # No overlap strict
    stream = obspy.Stream(
        [
            trace_z.slice(t0, t0 + 1),
            trace_n.slice(t0 + 1, t0 + 2),
            trace_e.slice(t0 + 1, t0 + 2),
        ]
    )
    times, data = dummy.stream_to_arrays(stream, strict=True)
    assert len(times) == len(data) == 0

    # No overlap non strict
    stream = obspy.Stream(
        [
            trace_z.slice(t0, t0 + 1),
            trace_n.slice(t0 + 1, t0 + 2),
            trace_e.slice(t0 + 1, t0 + 2),
        ]
    )
    times, data = dummy.stream_to_arrays(stream, strict=False)
    assert len(times) == len(data) == 1
    assert times[0] == t0
    assert data[0].shape == (3, 201)
    assert (data[0][0, :101] == trace_z.data[0]).all()
    assert (data[0][0, 101:] == 0).all()
    assert (data[0][1, :100] == 0).all()
    assert (data[0][1, 100:] == trace_n.data[0]).all()
    assert (data[0][2, :100] == 0).all()
    assert (data[0][2, 100:] == trace_e.data[0]).all()

    # Separate fragments strict
    stream = obspy.Stream(
        [
            trace_z.slice(t0, t0 + 1),
            trace_n.slice(t0 + 0, t0 + 1),
            trace_e.slice(t0 + 0, t0 + 1),
            trace_z.slice(t0 + 2, t0 + 3),
            trace_n.slice(t0 + 2, t0 + 3),
            trace_e.slice(t0 + 2, t0 + 3),
        ]
    )
    times, data = dummy.stream_to_arrays(stream, strict=True)
    assert len(times) == len(data) == 2
    for i in range(2):
        assert times[i] == t0 + 2 * i
        assert data[i].shape == (3, 101)
        assert (data[i][0] == trace_z.data[0]).all()
        assert (data[i][1] == trace_n.data[0]).all()
        assert (data[i][2] == trace_e.data[0]).all()

    # Separate fragments non-strict
    stream = obspy.Stream(
        [
            trace_z.slice(t0, t0 + 1),
            trace_n.slice(t0 + 0, t0 + 1),
            trace_e.slice(t0 + 0, t0 + 1),
            trace_z.slice(t0 + 2, t0 + 3),
            trace_n.slice(t0 + 2, t0 + 3),
            trace_e.slice(t0 + 2, t0 + 3),
        ]
    )
    times, data = dummy.stream_to_arrays(stream, strict=False)
    assert len(times) == len(data) == 2
    for i in range(2):
        assert times[i] == t0 + 2 * i
        assert data[i].shape == (3, 101)
        assert (data[i][0] == trace_z.data[0]).all()
        assert (data[i][1] == trace_n.data[0]).all()
        assert (data[i][2] == trace_e.data[0]).all()


def test_stream_to_arrays_channel():
    t0 = UTCDateTime(0)
    stats_z = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHZ",
        "sampling_rate": 100,
        "starttime": t0,
    }
    dummy = DummyWaveformModel(grouping="channel")

    trace_z = obspy.Trace(np.ones(1000), stats_z)

    # Simple
    stream = obspy.Stream([trace_z])
    times, data = dummy.stream_to_arrays(
        stream,
    )
    assert len(times) == len(data) == 1
    assert times[0] == t0
    assert data[0].shape == (len(trace_z.data),)
    assert (data[0] == trace_z.data).all()

    # Separate fragments
    stream = obspy.Stream([trace_z.slice(t0, t0 + 1), trace_z.slice(t0 + 2, t0 + 3)])
    times, data = dummy.stream_to_arrays(stream, strict=True)
    assert len(times) == len(data) == 2
    for i in range(2):
        assert times[i] == t0 + 2 * i
        assert data[i].shape == (101,)


def test_flexible_horizontal_components(caplog):
    t0 = UTCDateTime(0)
    stats_z = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHZ",
        "sampling_rate": 100,
        "starttime": t0,
    }
    stats_n = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHN",
        "sampling_rate": 100,
        "starttime": t0,
    }
    stats_e = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHE",
        "sampling_rate": 100,
        "starttime": t0,
    }
    stats_1 = {
        "network": "SB",
        "station": "TEST",
        "channel": "HH1",
        "sampling_rate": 100,
        "starttime": t0,
    }
    stats_2 = {
        "network": "SB",
        "station": "TEST",
        "channel": "HH2",
        "sampling_rate": 100,
        "starttime": t0,
    }
    stats_2_test2 = {
        "network": "SB",
        "station": "TEST2",
        "channel": "HH2",
        "sampling_rate": 100,
        "starttime": t0,
    }
    dummy = DummyWaveformModel(component_order="ZNE")

    trace_z = obspy.Trace(np.ones(1000), stats_z)
    trace_n = obspy.Trace(2 * np.ones(1000), stats_n)
    trace_e = obspy.Trace(3 * np.ones(1000), stats_e)
    trace_1 = obspy.Trace(4 * np.ones(1000), stats_1)
    trace_2 = obspy.Trace(5 * np.ones(1000), stats_2)
    trace_2_test2 = obspy.Trace(5 * np.ones(1000), stats_2_test2)

    # flexible_horizontal_components=False
    stream = obspy.Stream([trace_z, trace_1, trace_2])
    times, data = dummy.stream_to_arrays(
        stream, strict=True, flexible_horizontal_components=False
    )
    assert len(times) == len(data) == 0

    # flexible_horizontal_components=True
    stream = obspy.Stream([trace_z, trace_1, trace_2])
    times, data = dummy.stream_to_arrays(
        stream, strict=True, flexible_horizontal_components=True
    )
    assert len(times) == len(data) == 1

    # Warning for mixed component names
    caplog.clear()
    stream = obspy.Stream([trace_z, trace_n, trace_e, trace_1, trace_2])
    with caplog.at_level(logging.WARNING):
        times, data = dummy.stream_to_arrays(
            stream, strict=True, flexible_horizontal_components=True
        )
    assert "This might lead to undefined behavior." in caplog.text
    assert len(times) == len(data) == 1

    # No warning for mixed component names on different stations
    caplog.clear()
    stream = obspy.Stream([trace_z, trace_n, trace_e, trace_2_test2])
    with caplog.at_level(logging.WARNING):
        dummy.stream_to_arrays(stream, strict=True, flexible_horizontal_components=True)
    assert "This might lead to undefined behavior." not in caplog.text


def test_group_stream_by_instrument():
    # The first 3 should be grouped together, the last 3 should each be separate
    stream = obspy.Stream(
        [
            obspy.Trace(header={"network": "SB", "station": "ABC1", "channel": "BHZ"}),
            obspy.Trace(header={"network": "SB", "station": "ABC1", "channel": "BHN"}),
            obspy.Trace(header={"network": "SB", "station": "ABC1", "channel": "BHE"}),
            obspy.Trace(header={"network": "SB", "station": "ABC1", "channel": "HHZ"}),
            obspy.Trace(header={"network": "HB", "station": "ABC1", "channel": "BHZ"}),
            obspy.Trace(header={"network": "SB", "station": "ABC2", "channel": "BHZ"}),
        ]
    )

    dummy = DummyWaveformModel(component_order="ZNE")

    dummy._grouping = "instrument"
    groups = dummy.group_stream(stream)

    assert len(groups) == 4
    assert list(sorted([len(x) for x in groups])) == [1, 1, 1, 3]

    dummy._grouping = "channel"
    groups = dummy.group_stream(stream)

    assert len(groups) == 6
    assert list(sorted([len(x) for x in groups])) == [1, 1, 1, 1, 1, 1]

    dummy._grouping = "invalid"
    with pytest.raises(ValueError):
        dummy.group_stream(stream)


def test_recursive_torch_to_numpy():
    dummy = DummyWaveformModel(component_order="ZNE")

    x = np.random.rand(5, 3)
    y = dummy._recursive_torch_to_numpy(torch.tensor(x))
    assert (x == y).all()

    xt = torch.tensor(x)
    bigtest = [[xt, xt, xt], (xt, xt, xt, xt)]
    bigresult = dummy._recursive_torch_to_numpy(bigtest)
    assert len(bigresult) == 2
    assert isinstance(bigresult, list)
    assert len(bigresult[0]) == 3
    assert isinstance(bigresult[0], list)
    assert len(bigresult[1]) == 4
    assert isinstance(bigresult[1], tuple)


def test_recursive_slice_pred():
    dummy = DummyWaveformModel(component_order="ZNE")

    x1 = np.random.rand(5, 3)
    x2 = np.random.rand(5, 3)
    x3 = np.random.rand(5, 3)
    x4 = np.random.rand(5, 3)
    x5 = np.random.rand(5, 3)

    x = [[x1, x2], (x3, x4, x5)]
    y = dummy._recursive_slice_pred(x)

    for i in range(x1.shape[0]):
        assert isinstance(y[i], list)
        assert isinstance(y[i][0], list)
        assert isinstance(y[i][1], tuple)

        [px1, px2], (px3, px4, px5) = y[i]
        assert (x1[i] == px1).all()
        assert (x2[i] == px2).all()
        assert (x3[i] == px3).all()
        assert (x4[i] == px4).all()
        assert (x5[i] == px5).all()


def test_trim_nan():
    dummy = DummyWaveformModel(component_order="ZNE")

    x = np.random.rand(100)
    y, f, b = dummy._trim_nan(x)
    assert (y == x).all()
    assert (f, b) == (0, 0)

    x[:10] = np.nan
    y, f, b = dummy._trim_nan(x)
    assert (y == x[10:]).all()
    assert (f, b) == (10, 0)

    x[95:] = np.nan
    y, f, b = dummy._trim_nan(x)
    assert (y == x[10:95]).all()
    assert (f, b) == (10, 5)

    x[30:40] = np.nan
    y, f, b = dummy._trim_nan(x)
    # Needs an extra check as nan==nan is defined as false
    double_nan = np.logical_and(np.isnan(y), np.isnan(x[10:95]))
    assert (np.logical_or(y == x[10:95], double_nan)).all()
    assert (f, b) == (10, 5)


def test_predictions_to_stream():
    dummy = DummyWaveformModel(component_order="ZNE")

    pred_rates = [100, 50]
    pred_times = [UTCDateTime(), UTCDateTime()]
    preds = [np.random.rand(1000, 3), np.random.rand(2000, 3)]
    preds[0][:100] = np.nan  # Test proper shift
    trace = obspy.Trace(np.zeros(100), header={"network": "SB", "station": "ABC1"})

    stream = dummy._predictions_to_stream(
        pred_rates[0], pred_times[0], preds[0], trace.stats
    )
    stream += dummy._predictions_to_stream(
        pred_rates[1], pred_times[1], preds[1], trace.stats
    )

    assert stream[0].stats.starttime == pred_times[0] + 1
    assert stream[1].stats.starttime == pred_times[0] + 1
    assert stream[2].stats.starttime == pred_times[0] + 1
    assert stream[3].stats.starttime == pred_times[1]
    assert stream[4].stats.starttime == pred_times[1]
    assert stream[5].stats.starttime == pred_times[1]

    assert stream[0].id == "SB.ABC1..DummyWaveformModel_0"
    assert stream[1].id == "SB.ABC1..DummyWaveformModel_1"
    assert stream[2].id == "SB.ABC1..DummyWaveformModel_2"
    assert stream[3].id == "SB.ABC1..DummyWaveformModel_0"
    assert stream[4].id == "SB.ABC1..DummyWaveformModel_1"
    assert stream[5].id == "SB.ABC1..DummyWaveformModel_2"

    assert stream[0].stats.sampling_rate == 100
    assert stream[1].stats.sampling_rate == 100
    assert stream[2].stats.sampling_rate == 100
    assert stream[3].stats.sampling_rate == 50
    assert stream[4].stats.sampling_rate == 50
    assert stream[5].stats.sampling_rate == 50


def test_cut_fragments_point():
    dummy = DummyWaveformModel(
        component_order="ZNE", in_samples=1000, sampling_rate=100
    )
    data = [np.ones((3, 10000))]

    out = [
        x[0]
        for x in dummy._cut_fragments_point(
            0, data[0], None, {"stride": 100, "sampling_rate": 100}
        )
    ]

    assert len(out) == 91
    assert out[0].shape == (3, 1000)


def test_reassemble_blocks_point():
    dummy = DummyWaveformModel(
        component_order="ZNE", in_samples=1000, sampling_rate=100
    )

    trace_stats = obspy.read()[0].stats

    out = []
    argdict = {"stride": 100, "sampling_rate": 100}
    buffer = defaultdict(list)
    for i in range(100):
        elem = ([0], (0, i, 100, trace_stats, 0))
        out_elem = dummy._reassemble_blocks_point(elem, buffer, argdict)
        if out_elem is not None:
            out += [out_elem[0]]

    assert len(out) == 1
    assert out[0][0] == 1
    assert out[0][2].shape == (100, 1)


def test_cut_fragments_array():
    dummy = DummyWaveformModel(
        component_order="ZNE", in_samples=1000, sampling_rate=100, pred_sample=(0, 1000)
    )
    data = [np.ones((3, 10001))]
    argdict = {"overlap": 100, "sampling_rate": 100}
    elem = (0, data[0], None)

    out = [x[0] for x in dummy._cut_fragments_array(elem, argdict)]

    assert len(out) == 12
    assert out[0].shape == (3, 1000)


def test_reassemble_blocks_array():
    dummy = DummyWaveformModel(
        component_order="ZNE", in_samples=1000, sampling_rate=100, pred_sample=(0, 1000)
    )

    trace_stats = obspy.read()[0].stats

    out = []
    argdict = {"stride": 100, "sampling_rate": 100}
    buffer = defaultdict(list)

    starts = [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9000, 9001]

    for i in range(12):
        elem = (np.ones((1000, 3)), (0, starts[i], 12, trace_stats, 0))
        out_elem = dummy._reassemble_blocks_array(elem, buffer, argdict)
        if out_elem is not None:
            out += [out_elem[0]]

    assert len(out) == 1
    assert out[0][0] == 100.0
    assert out[0][2].shape == (10001, 3)


def test_reassemble_blocks_array_stack_options():
    # Check that the maximum is taken when stacking
    dummy = DummyWaveformModel(
        component_order="ZNE", in_samples=1000, sampling_rate=100, pred_sample=(0, 1000)
    )

    trace_stats = obspy.read()[0].stats

    for stacking in {"max", "avg"}:
        out = []
        argdict = {"overlap": 100, "stacking": stacking, "sampling_rate": 100}
        buffer = defaultdict(list)

        starts = [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9000, 9001]

        for i in range(12):
            elem = (np.ones((1000, 3)) + i, (0, starts[i], 12, trace_stats, 0))
            out_elem = dummy._reassemble_blocks_array(elem, buffer, argdict)
            if out_elem is not None:
                out += [out_elem[0]]

        assert len(out) == 1
        assert out[0][0] == 100.0
        assert out[0][2].shape == (10001, 3)
        assert out[0][2].min() == 1
        assert out[0][2].max() == 12
        if stacking == "max":
            # In the max stacking scheme the last window length samples should be equal to the max
            assert np.all(out[0][2][-1000:] == 12)
        elif stacking == "avg":
            # In the avg stacking scheme the samples from 9100 to 10000 should be equal to the mean of 11, 12
            assert np.all(out[0][2][9100:10000] == 11.5)


def test_picks_from_annotations():
    data = np.zeros(1000)
    data[100:201] = 0.7
    data[300:401] = 0.1
    data[500:701] = 0.5
    data[600] = 0.6  # Different peak
    t0 = UTCDateTime()
    trace = obspy.Trace(
        data,
        header={
            "network": "SB",
            "station": "ABC1",
            "starttime": t0,
            "sampling_rate": 100,
        },
    )

    picks = seisbench.models.WaveformModel.picks_from_annotations(
        obspy.Stream([trace]), 0.3, "P"
    )
    picks = sorted(picks)

    assert len(picks) == 2

    assert picks[0].trace_id == "SB.ABC1."
    assert picks[0].start_time == t0 + 1
    assert picks[0].end_time == t0 + 2
    assert picks[0].peak_value == 0.7
    assert picks[0].phase == "P"

    assert picks[1].trace_id == "SB.ABC1."
    assert picks[1].start_time == t0 + 5
    assert picks[1].end_time == t0 + 7
    assert picks[1].peak_time == t0 + 6
    assert picks[1].peak_value == 0.6
    assert picks[1].phase == "P"


def test_detections_from_annotations():
    data = np.zeros(1000)
    data[100:201] = 0.7
    data[300:401] = 0.1
    data[500:701] = 0.5
    data[600] = 0.6  # Different peak
    t0 = UTCDateTime()
    trace = obspy.Trace(
        data,
        header={
            "network": "SB",
            "station": "ABC1",
            "starttime": t0,
            "sampling_rate": 100,
        },
    )

    detections = seisbench.models.WaveformModel.detections_from_annotations(
        obspy.Stream([trace]), 0.3
    )
    detections = sorted(detections)

    assert len(detections) == 2

    assert detections[0].trace_id == "SB.ABC1."
    assert detections[0].start_time == t0 + 1
    assert detections[0].end_time == t0 + 2
    assert detections[0].peak_value == 0.7

    assert detections[1].trace_id == "SB.ABC1."
    assert detections[1].start_time == t0 + 5
    assert detections[1].end_time == t0 + 7
    assert detections[1].peak_value == 0.6


def test_waveform_pipeline_instantiation():
    with pytest.raises(TypeError):
        seisbench.models.WaveformPipeline({})

    class MyPipeline(seisbench.models.WaveformPipeline):
        def annotate(self, stream, **kwargs):
            pass

        def classify(self, stream, **kwargs):
            pass

        @classmethod
        def component_classes(cls):
            return {}

    MyPipeline({})


def test_waveform_pipeline_list_pretrained(tmp_path):
    class MyPipeline(seisbench.models.WaveformPipeline):
        def annotate(self, stream, **kwargs):
            pass

        def classify(self, stream, **kwargs):
            pass

        @classmethod
        def component_classes(cls):
            return {}

    def write_config(*arg, **kwargs):
        path = tmp_path / "pipelines" / "mypipeline" / "myconfig.json"
        with open(path, "w") as f:
            f.write('{"docstring": "test docstring"}\n')

    with patch(
        "seisbench.cache_root", tmp_path
    ):  # Ensure test does not modify SeisBench cache
        with patch("seisbench.util.ls_webdav") as ls_webdav:
            ls_webdav.return_value = ["myconfig.json"]
            with patch("seisbench.util.download_http") as download_http:
                download_http.side_effect = write_config

                configurations = MyPipeline.list_pretrained(details=True)
                assert configurations == {"myconfig": "test docstring"}


def test_waveform_pipeline_from_pretrained(tmp_path):
    class MyPipeline(seisbench.models.WaveformPipeline):
        def annotate(self, stream, **kwargs):
            pass

        def classify(self, stream, **kwargs):
            pass

        @classmethod
        def component_classes(cls):
            return {"gpd": seisbench.models.GPD}

    def write_config(*arg, **kwargs):
        path = tmp_path / "pipelines" / "mypipeline" / "myconfig.json"
        with open(path, "w") as f:
            f.write('{"components": {"gpd": "dummy"}}\n')

    with patch(
        "seisbench.cache_root", tmp_path
    ):  # Ensure test does not modify SeisBench cache
        with patch("seisbench.util.download_http") as download_http:
            download_http.side_effect = write_config

            with patch("seisbench.models.GPD.from_pretrained") as gpd_from_pretrained:
                gpd_from_pretrained.return_value = (
                    seisbench.models.GPD()
                )  # Return random GPD instance

                MyPipeline.from_pretrained("myconfig")

                gpd_from_pretrained.assert_called_once_with(
                    "dummy", force=False, wait_for_file=False
                )


def test_recurrent_dropout():
    lstm = CustomLSTM(
        ActivationLSTMCell, 1, 100, bidirectional=True, recurrent_dropout=0.25
    )
    x = torch.rand(100, 5, 1)

    with torch.no_grad():
        y = lstm(x)[0]

    assert y.shape == (100, 5, 200)


def test_dpp_detector():
    model = seisbench.models.DPPDetector(input_channels=3, nclasses=3)

    x = torch.rand(16, 3, 500)
    y = model(x)

    assert y.shape == (16, 3)


def test_dpp_ps():
    model = seisbench.models.DPPPicker("P")

    x = torch.rand(16, 1, 500)
    y = model(x)

    assert y.shape == (16, 500)

    model = seisbench.models.DPPPicker("S")

    x = torch.rand(16, 2, 500)
    y = model(x)

    assert y.shape == (16, 500)


def test_resample():
    # Only tests for correct target frequencies and number of samples, but not for correct content
    t1 = obspy.Trace(np.random.rand(1000), header={"sampling_rate": 100})  # Decimate
    t2 = obspy.Trace(np.random.rand(1000), header={"sampling_rate": 25})  # Unchanged
    t3 = obspy.Trace(np.random.rand(1000), header={"sampling_rate": 40})  # Resample

    s = obspy.Stream([t1, t2, t3])
    seisbench.models.WaveformModel.resample(s, 25)
    assert s[0].stats.sampling_rate == 25
    assert len(s[0]) == 250
    assert s[1].stats.sampling_rate == 25
    assert len(s[1]) == 1000
    assert s[1].stats.sampling_rate == 25
    assert len(s[2]) == 625


def test_parse_seisbench_requirements():
    model = seisbench.models.GPD()

    # Minimum version
    model._weights_metadata = {"seisbench_requirement": seisbench.__version__}
    model._parse_metadata()

    # Newer version
    model._weights_metadata = {"seisbench_requirement": seisbench.__version__ + "1"}
    with pytest.raises(ValueError):
        model._parse_metadata()

    # Older version
    version = seisbench.__version__
    version = version[:-1] + chr(ord(version[-1]) - 1)
    model._weights_metadata = {"seisbench_requirement": version}
    model._parse_metadata()


def test_parse_default_args():
    model = seisbench.models.GPD()

    # Minimum version
    default_args = {"dummy": 1, "stride": 100}
    model._weights_metadata = {"default_args": default_args}
    model._parse_metadata()

    for key in default_args:
        assert key in model.default_args
        assert model.default_args[key] == default_args[key]


def test_default_labels():
    model = seisbench.models.PhaseNet(
        sampling_rate=200, phases=None
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    assert model.labels is None

    model.classify(stream)  # Ensures classify succeeds even though labels are unknown

    assert model.labels == [0, 1, 2]


@pytest.mark.parametrize(
    "parallelism",
    [None, 1],
)
def test_annotate_cred(parallelism):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.CRED(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream, parallelism=parallelism)
    assert len(annotations) > 0
    model.classify(
        stream, parallelism=parallelism
    )  # Ensures classify succeeds even though labels are unknown


@pytest.mark.parametrize(
    "parallelism",
    [None, 1],
)
def test_annotate_eqtransformer(parallelism):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.EQTransformer(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream, parallelism=parallelism)
    assert len(annotations) > 0
    model.classify(
        stream, parallelism=parallelism
    )  # Ensures classify succeeds even though labels are unknown


@pytest.mark.parametrize(
    "parallelism",
    [None, 1],
)
def test_annotate_gpd(parallelism):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.GPD(
        sampling_rate=100
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream, parallelism=parallelism)
    assert len(annotations) > 0
    model.classify(
        stream, parallelism=parallelism
    )  # Ensures classify succeeds even though labels are unknown


@pytest.mark.parametrize(
    "parallelism",
    [None, 1],
)
def test_annotate_phasenet(parallelism):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.PhaseNet(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream, parallelism=parallelism)
    assert len(annotations) > 0
    model.classify(
        stream, parallelism=parallelism
    )  # Ensures classify succeeds even though labels are unknown


@pytest.mark.parametrize(
    "parallelism",
    [None, 1],
)
def test_annotate_basicphaseae(parallelism):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.BasicPhaseAE(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream, parallelism=parallelism)
    assert len(annotations) > 0
    model.classify(
        stream, parallelism=parallelism
    )  # Ensures classify succeeds even though labels are unknown


def test_short_traces(caplog):
    # Test that on both point and array models short traces do not cause an infinite loop, but a warning
    stream = obspy.read()
    t0 = stream[0].stats.starttime
    stream = stream.slice(t0, t0 + 1)  # Cut very short trace

    model = seisbench.models.GPD()
    with caplog.at_level(logging.WARNING):
        ann = model.annotate(stream)
    assert "Output might be empty." in caplog.text
    assert len(ann) == 0

    caplog.clear()

    model = seisbench.models.EQTransformer()
    with caplog.at_level(logging.WARNING):
        ann = model.annotate(stream)
    assert "Output might be empty." in caplog.text
    assert len(ann) == 0


def test_deep_denoiser():
    seisbench.models.DeepDenoiser()


@pytest.mark.parametrize(
    "parallelism",
    [None, 1],
)
def test_annotate_deep_denoiser(parallelism):
    stream = obspy.read()

    model = seisbench.models.DeepDenoiser()
    annotations = model.annotate(stream, parallelism=parallelism)

    assert len(annotations) == 3
    for i in range(3):
        assert annotations[i].data.shape == (3000,)


def test_eqtransformer_annotate_window_post():
    model = seisbench.models.EQTransformer()

    pred = 3 * [np.ones(1000)]

    # Default: No blinding
    blinded = model.annotate_window_post(pred, argdict={})
    assert (blinded == 1).all()

    # No blinding
    blinded = model.annotate_window_post(pred, argdict={"blinding": (0, 0)})
    assert (blinded == 1).all()

    # Front blinding
    blinded = model.annotate_window_post(pred, argdict={"blinding": (100, 0)})
    assert np.isnan(blinded[:100]).all()
    assert (blinded[100:] == 1).all()

    # End blinding
    blinded = model.annotate_window_post(pred, argdict={"blinding": (0, 100)})
    assert (blinded[:900] == 1).all()
    assert np.isnan(blinded[900:]).all()

    # Two sided blinding
    blinded = model.annotate_window_post(pred, argdict={"blinding": (100, 100)})
    assert np.isnan(blinded[:100]).all()
    assert (blinded[100:900] == 1).all()
    assert np.isnan(blinded[900:]).all()


def test_save_load_gpd(tmp_path):
    model_orig = seisbench.models.GPD()
    model_orig_args = get_input_args(model_orig.__class__)

    # Test model saving
    model_orig.save(tmp_path / "gpd")
    assert (tmp_path / "gpd.json").exists()
    assert (tmp_path / "gpd.pt").exists()

    stream = obspy.read()

    # Test model loading
    model_load = seisbench.models.GPD.load(tmp_path / "gpd")
    model_load_args = get_input_args(model_orig.__class__)

    # Test no changes to weights
    pred_orig = model_orig.annotate(stream, sampling_rate=400)
    pred_load = model_load.annotate(stream, sampling_rate=400)

    for i in range(len(pred_orig)):
        assert np.allclose(pred_orig[i].data, pred_load[i].data)
    assert model_orig_args == model_load_args


def test_save_load_basicphaseae(tmp_path, caplog):
    model_orig = seisbench.models.BasicPhaseAE()
    model_orig_args = get_input_args(model_orig.__class__)

    # Test model saving
    with caplog.at_level(logging.WARNING):
        model_orig.save(tmp_path / "basicphaseae")

    assert (tmp_path / "basicphaseae.json").exists()
    assert (tmp_path / "basicphaseae.pt").exists()

    stream = obspy.read()

    # Test model loading
    model_load = seisbench.models.BasicPhaseAE.load(tmp_path / "basicphaseae")
    model_load_args = get_input_args(model_orig.__class__)

    # Test no changes to weights
    pred_orig = model_orig.annotate(stream, sampling_rate=400)
    pred_load = model_load.annotate(stream, sampling_rate=400)

    # TODO: Find out why there is a single nan value in the reconstruction.
    #     for i in range(len(pred_orig)):
    #         assert np.allclose(pred_orig[i].data, pred_load[i].data)
    assert model_orig_args == model_load_args


def test_save_load_phasenet(tmp_path):
    model_orig = seisbench.models.PhaseNet()
    model_orig_args = get_input_args(model_orig.__class__)

    # Test model saving
    model_orig.save(tmp_path / "phasenet")
    assert (tmp_path / "phasenet.json").exists()
    assert (tmp_path / "phasenet.pt").exists()

    stream = obspy.read()

    # Test model loading
    model_load = seisbench.models.PhaseNet.load(tmp_path / "phasenet")
    model_load_args = get_input_args(model_orig.__class__)

    # Test no changes to weights
    pred_orig = model_orig.annotate(stream, sampling_rate=400)
    pred_load = model_load.annotate(stream, sampling_rate=400)

    for i in range(len(pred_orig)):
        assert np.allclose(pred_orig[i].data, pred_load[i].data)
    assert model_orig_args == model_load_args


def test_save_load_eqtransformer(tmp_path):
    model_orig = seisbench.models.EQTransformer()
    model_orig_args = get_input_args(model_orig.__class__)

    # Test model saving
    model_orig.save(tmp_path / "eqtransformer")
    assert (tmp_path / "eqtransformer.json").exists()
    assert (tmp_path / "eqtransformer.pt").exists()

    stream = obspy.read()

    # Test model loading
    model_load = seisbench.models.EQTransformer.load(tmp_path / "eqtransformer")
    model_load_args = get_input_args(model_orig.__class__)

    # Test no changes to weights
    pred_orig = model_orig.annotate(stream, sampling_rate=400)
    pred_load = model_load.annotate(stream, sampling_rate=400)

    for i in range(len(pred_orig)):
        assert np.allclose(pred_orig[i].data, pred_load[i].data)
    assert model_orig_args == model_load_args


def test_save_load_cred(tmp_path):
    model_orig = seisbench.models.CRED()
    model_orig_args = get_input_args(model_orig.__class__)

    # Test model saving
    model_orig.save(tmp_path / "cred")
    assert (tmp_path / "cred.json").exists()
    assert (tmp_path / "cred.pt").exists()

    stream = obspy.read()

    # Test model loading
    model_load = seisbench.models.CRED.load(tmp_path / "cred")
    model_load_args = get_input_args(model_orig.__class__)

    # Test no changes to weights
    pred_orig = model_orig.annotate(stream, sampling_rate=400)
    pred_load = model_load.annotate(stream, sampling_rate=400)

    for i in range(len(pred_orig)):
        assert np.allclose(pred_orig[i].data, pred_load[i].data)
    assert model_orig_args == model_load_args


def test_save_load_deepdenoiser(tmp_path):
    model_orig = seisbench.models.DeepDenoiser()
    model_orig_args = get_input_args(model_orig.__class__)

    # Test model saving
    model_orig.save(tmp_path / "deepdenoiser")
    assert (tmp_path / "deepdenoiser.json").exists()
    assert (tmp_path / "deepdenoiser.pt").exists()

    stream = obspy.read()

    # Test model loading
    model_load = seisbench.models.DeepDenoiser.load(tmp_path / "deepdenoiser")
    model_load_args = get_input_args(model_orig.__class__)

    # Test no changes to weights
    pred_orig = model_orig.annotate(stream, sampling_rate=400)
    pred_load = model_load.annotate(stream, sampling_rate=400)

    for i in range(len(pred_orig)):
        assert np.allclose(pred_orig[i].data, pred_load[i].data)
    assert model_orig_args == model_load_args


def test_save_load_model_updated_after_construction(tmp_path):
    model_orig = seisbench.models.EQTransformer()
    model_orig_state = model_orig.__dict__

    # Check obtaining of original arguments for object works
    model_orig_args = get_input_args(model_orig.__class__)
    assert model_orig_state["in_samples"] == model_orig_args["in_samples"]
    assert model_orig_state["sampling_rate"] == model_orig_args["sampling_rate"]

    # Test model saving w. updated params
    model_orig.in_samples = 10_000
    model_orig.sampling_rate = 500

    model_orig.save(tmp_path / "eqtransformer_changed")
    assert (tmp_path / "eqtransformer_changed.json").exists()
    assert (tmp_path / "eqtransformer_changed.pt").exists()

    # Test model loading w. updated params
    model_load = seisbench.models.EQTransformer.load(tmp_path / "eqtransformer_changed")
    model_load_state = model_load.__dict__

    assert model_load_state["in_samples"] == 10_000
    assert model_load_state["sampling_rate"] == 500


def test_save_load_model_updated_after_construction_inheritence_compatible(tmp_path):
    model_orig = seisbench.models.GPD(component_order="NZE")

    model_orig.save(tmp_path / "gpd_changed")
    model_loaded = model_orig.load(tmp_path / "gpd_changed")

    assert model_orig.component_order == model_loaded.component_order


def test_check_parallelism_annotate(caplog):
    t0 = UTCDateTime(0)
    stats = {
        "network": "SB",
        "station": "TEST",
        "channel": "HHZ",
        "sampling_rate": 100,
        "starttime": t0,
    }

    trace0 = obspy.Trace(np.zeros(1000), stats)
    trace1 = obspy.Trace(np.ones(1000), stats)
    stream_small = obspy.Stream([trace0, trace1])

    trace0 = obspy.Trace(np.zeros(int(3e7)), stats)
    trace1 = obspy.Trace(np.zeros(int(3e7)), stats)
    stream_large = obspy.Stream([trace0, trace1])

    with caplog.at_level(logging.WARNING):
        seisbench.models.WaveformModel._check_parallelism_annotate(
            stream_small, parallelism=None
        )
    assert "Consider activating parallelisation. " not in caplog.text

    with caplog.at_level(logging.WARNING):
        seisbench.models.WaveformModel._check_parallelism_annotate(
            stream_large, parallelism=None
        )
    assert "Consider activating parallelisation. " in caplog.text

    caplog.clear()

    with caplog.at_level(logging.WARNING):
        seisbench.models.WaveformModel._check_parallelism_annotate(
            stream_large, parallelism=1
        )
    assert "Consider using the sequential asyncio implementation. " not in caplog.text

    with caplog.at_level(logging.WARNING):
        seisbench.models.WaveformModel._check_parallelism_annotate(
            stream_small, parallelism=1
        )
    assert "Consider using the sequential asyncio implementation. " in caplog.text


def test_get_versions_from_files():
    # Only latest
    files = ["original.pt", "original.json"]
    versions = seisbench.models.EQTransformer._get_versions_from_files(
        "original", files=files
    )
    assert versions == [""]

    # No weights at all
    files = []
    versions = seisbench.models.EQTransformer._get_versions_from_files(
        "something", files=files
    )
    assert versions == []

    # No weights for target name
    files = ["original.pt", "original.json"]
    versions = seisbench.models.EQTransformer._get_versions_from_files(
        "something", files=files
    )
    assert versions == []

    # Multiple versions
    files = ["original.pt", "original.json", "original.pt.v1", "original.json.v1"]
    versions = seisbench.models.EQTransformer._get_versions_from_files(
        "original", files=files
    )
    assert versions == ["", "1"]

    # Multiple versions - mixed names
    files = [
        "original.pt",
        "original.json",
        "original.pt.v1",
        "original.json.v1",
        "something.pt.v1",
        "something.json.v1",
    ]
    versions = seisbench.models.EQTransformer._get_versions_from_files(
        "original", files=files
    )
    assert versions == ["", "1"]
    versions = seisbench.models.EQTransformer._get_versions_from_files(
        "something", files=files
    )
    assert versions == ["1"]


def test_cleanup_local_repository(tmp_path):
    with patch("seisbench.models.GPD._model_path") as model_path:
        model_path.return_value = tmp_path

        # Create dummies
        def create_versions(name, versions, content):
            for x in versions:
                for suffix in ["json", "pt"]:
                    if x == "":
                        filename = f"{name}.{suffix}"
                    else:
                        filename = f"{name}.{suffix}.v{x}"
                    with open(tmp_path / filename, "w") as f:
                        f.write(content)

        create_versions("test", ["", "2"], "{}\n")
        create_versions("original", [""], '{"version": "1.2.3"}\n')

        seisbench.models.GPD._cleanup_local_repository()

        assert sorted([x.name for x in tmp_path.iterdir()]) == [
            "original.json.v1.2.3",
            "original.pt.v1.2.3",
            "test.json.v1",
            "test.json.v2",
            "test.pt.v1",
            "test.pt.v2",
        ]

        model_path.return_value = tmp_path / "not_existent"
        seisbench.models.GPD._cleanup_local_repository()  # Just checking that this does not crash


def test_list_versions(tmp_path):
    with patch("seisbench.models.GPD._model_path") as model_path:
        model_path.return_value = tmp_path

        # Create dummies
        def create_versions(name, versions, content):
            for x in versions:
                for suffix in ["json", "pt"]:
                    if x == "":
                        filename = f"{name}.{suffix}"
                    else:
                        filename = f"{name}.{suffix}.v{x}"
                    with open(tmp_path / filename, "w") as f:
                        f.write(content)

        create_versions("test", ["", "2"], "{}\n")
        assert seisbench.models.GPD.list_versions("test", remote=False) == ["1", "2"]

        with patch("seisbench.util.download_http") as download, patch(
            "seisbench.util.ls_webdav"
        ) as ls_webdav:

            def side_effect(remote_metadata_path, metadata_path, progress_bar=False):
                # Checks correct url and writes out dummy
                assert remote_metadata_path.endswith("test.json")
                assert remote_metadata_path.startswith(
                    seisbench.models.GPD._remote_path()
                )
                with open(metadata_path, "w") as f:
                    f.write('{"version": "3"}\n')

            download.side_effect = side_effect
            ls_webdav.return_value = ["test.json"]

            assert seisbench.models.GPD.list_versions("test", remote=True) == [
                "1",
                "2",
                "3",
            ]

        with patch("seisbench.util.ls_webdav") as ls_webdav:
            ls_webdav.return_value = []
            (tmp_path / "a").mkdir()
            model_path.return_value = tmp_path / "a"

            assert seisbench.models.GPD.list_versions("test") == []

            model_path.return_value = tmp_path / "not_existent"
            assert (
                seisbench.models.GPD.list_versions("test") == []
            )  # Just check that this does not crash


def test_ensure_weight_files(tmp_path):
    # Files available
    tmp_path1 = tmp_path / "1"
    tmp_path1.mkdir()
    with open(tmp_path1 / "test.pt.v1", "w"), open(tmp_path1 / "test.json.v1", "w"):
        pass
    seisbench.models.GPD._ensure_weight_files(
        "test", "1", tmp_path1 / "test.pt.v1", tmp_path1 / "test.json.v1", False, False
    )

    # File available remote with version suffix
    tmp_path2 = tmp_path / "2"
    tmp_path2.mkdir()

    with patch("seisbench.util.download_http") as download, patch(
        "seisbench.util.ls_webdav"
    ) as ls_webdav:

        def side_effect(remote_path, local_path, progress_bar=False):
            # Checks correct url and writes out dummy
            assert remote_path.endswith("test.json.v1") or remote_path.endswith(
                "test.pt.v1"
            )
            assert remote_path.startswith(seisbench.models.GPD._remote_path())
            with open(local_path, "w") as f:
                f.write('{"version": "3"}\n')

        download.side_effect = side_effect
        ls_webdav.return_value = ["test.json.v1", "test.pt.v1"]

        seisbench.models.GPD._ensure_weight_files(
            "test",
            "1",
            tmp_path2 / "test.pt.v1",
            tmp_path2 / "test.json.v1",
            False,
            False,
        )

        assert (tmp_path2 / "test.pt.v1").is_file()
        assert (tmp_path2 / "test.json.v1").is_file()

    # File available remote without version suffix
    tmp_path3 = tmp_path / "3"
    tmp_path3.mkdir()

    with patch("seisbench.util.download_http") as download, patch(
        "seisbench.util.ls_webdav"
    ) as ls_webdav:

        def side_effect(remote_path, local_path, progress_bar=False):
            # Checks correct url and writes out dummy
            assert remote_path.endswith("test.json") or remote_path.endswith("test.pt")
            assert remote_path.startswith(seisbench.models.GPD._remote_path())
            with open(local_path, "w") as f:
                f.write('{"version": "1"}\n')

        download.side_effect = side_effect
        ls_webdav.return_value = ["test.json", "test.pt"]

        seisbench.models.GPD._ensure_weight_files(
            "test",
            "1",
            tmp_path3 / "test.pt.v1",
            tmp_path3 / "test.json.v1",
            False,
            False,
        )

        assert (tmp_path3 / "test.pt.v1").is_file()
        assert (tmp_path3 / "test.json.v1").is_file()

    # Version not available in remote - no file without suffix
    tmp_path4 = tmp_path / "4"
    tmp_path4.mkdir()

    with patch("seisbench.util.download_http") as download, patch(
        "seisbench.util.ls_webdav"
    ) as ls_webdav:

        def side_effect(remote_path, local_path, progress_bar=False):
            with open(local_path, "w") as f:
                f.write('{"version": "1"}\n')

        download.side_effect = side_effect
        ls_webdav.return_value = ["test.json.v3", "test.pt.v3"]

        with pytest.raises(ValueError):
            seisbench.models.GPD._ensure_weight_files(
                "test",
                "1",
                tmp_path4 / "test.pt.v1",
                tmp_path4 / "test.json.v1",
                False,
                False,
            )

    # Version not available in remote - file without suffix
    tmp_path5 = tmp_path / "5"
    tmp_path5.mkdir()

    with patch("seisbench.util.download_http") as download, patch(
        "seisbench.util.ls_webdav"
    ) as ls_webdav:

        def side_effect(remote_path, local_path, progress_bar=False):
            # Checks correct url and writes out dummy
            assert remote_path.endswith("test.json")
            assert remote_path.startswith(seisbench.models.GPD._remote_path())
            with open(local_path, "w") as f:
                f.write('{"version": "2"}\n')

        download.side_effect = side_effect
        ls_webdav.return_value = ["test.json", "test.pt"]

        with pytest.raises(ValueError):
            seisbench.models.GPD._ensure_weight_files(
                "test",
                "1",
                tmp_path5 / "test.pt.v1",
                tmp_path5 / "test.json.v1",
                False,
                False,
            )


def test_parse_weight_filename():
    assert seisbench.models.GPD._parse_weight_filename("test.json") == (
        "test",
        "json",
        None,
    )
    assert seisbench.models.GPD._parse_weight_filename("test.json.v1") == (
        "test",
        "json",
        "1",
    )
    assert seisbench.models.GPD._parse_weight_filename("test.pt") == (
        "test",
        "pt",
        None,
    )
    assert seisbench.models.GPD._parse_weight_filename("test.pt.v1") == (
        "test",
        "pt",
        "1",
    )
    assert seisbench.models.GPD._parse_weight_filename("test.jasd") == (
        None,
        None,
        None,
    )


def test_list_pretrained(tmp_path):
    with patch("seisbench.models.GPD._model_path") as model_path:
        model_path.return_value = tmp_path

        # Create dummies
        def create_versions(name, versions, content):
            for x, c in zip(versions, content):
                for suffix in ["json", "pt"]:
                    if x == "":
                        filename = f"{name}.{suffix}"
                    else:
                        filename = f"{name}.{suffix}.v{x}"
                    with open(tmp_path / filename, "w") as f:
                        f.write(c)

        create_versions(
            "test", ["1", "2"], ['{"docstring": "d1"}\n', '{"docstring": "d2"}\n']
        )
        create_versions(
            "bla", ["1", "2"], ['{"docstring": "b1"}\n', '{"docstring": "b2"}\n']
        )

        assert seisbench.models.GPD.list_pretrained(details=False, remote=False) == [
            "bla",
            "test",
        ]
        assert seisbench.models.GPD.list_pretrained(details=True, remote=False) == {
            "bla": "b2",
            "test": "d2",
        }

        with patch("seisbench.util.ls_webdav") as ls_webdav:
            ls_webdav.return_value = ["foo.json", "foo.pt"]
            assert seisbench.models.GPD.list_pretrained(details=False, remote=True) == [
                "bla",
                "foo",
                "test",
            ]

            with patch(
                "seisbench.models.GPD._get_latest_docstring"
            ) as get_latest_docstring:
                get_latest_docstring.return_value = "123"
                assert seisbench.models.GPD.list_pretrained(
                    details=True, remote=True
                ) == {"bla": "123", "test": "123", "foo": "123"}


def test_get_latest_docstring(tmp_path):
    with patch("seisbench.models.GPD._model_path") as model_path:
        model_path.return_value = tmp_path

        # Create dummies
        def create_versions(name, versions, content):
            for x, c in zip(versions, content):
                for suffix in ["json", "pt"]:
                    if x == "":
                        filename = f"{name}.{suffix}"
                    else:
                        filename = f"{name}.{suffix}.v{x}"
                    with open(tmp_path / filename, "w") as f:
                        f.write(c)

        create_versions(
            "test", ["1", "2"], ['{"docstring": "d1"}\n', '{"docstring": "d2"}\n']
        )

        assert seisbench.models.GPD._get_latest_docstring("test", remote=False) == "d2"

        with patch("seisbench.util.download_http") as download, patch(
            "seisbench.util.ls_webdav"
        ) as ls_webdav:

            def side_effect(remote_path, local_path, progress_bar=False):
                # Checks correct url and writes out dummy
                assert remote_path.endswith("test.json")
                assert remote_path.startswith(seisbench.models.GPD._remote_path())
                with open(local_path, "w") as f:
                    f.write('{"docstring": "d3", "version": "3"}\n')

            download.side_effect = side_effect
            ls_webdav.return_value = ["test.json", "test.pt"]

            assert (
                seisbench.models.GPD._get_latest_docstring("test", remote=True) == "d3"
            )


@pytest.mark.parametrize(
    "prefill_cache, update",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_from_pretrained(tmp_path, prefill_cache, update):
    with patch("seisbench.models.GPD._model_path") as model_path:
        model_path.return_value = tmp_path

        if prefill_cache:
            with open(tmp_path / "test.pt.v1", "wb") as fw, open(
                tmp_path / "test.json.v1", "w"
            ) as f:
                torch.save({}, fw)
                f.write("{}\n")

        with patch("seisbench.util.ls_webdav") as ls_webdav, patch(
            "seisbench.models.GPD._ensure_weight_files"
        ) as ensure_weight_files, patch("seisbench.models.GPD.load_state_dict") as _:
            ls_webdav.return_value = ["test.json.v2", "test.pt.v2"]

            def write_weights(
                name, version_str, weight_path, metadata_path, force, wait_for_file
            ):
                with open(weight_path, "wb") as fw, open(metadata_path, "w") as f:
                    torch.save({}, fw)
                    f.write("{}\n")

            ensure_weight_files.side_effect = write_weights

            seisbench.models.GPD.from_pretrained("test", update=update)


def test_save_load_with_version(tmp_path):
    model = seisbench.models.GPD()
    model.save(tmp_path / "mymodel", version_str="1")

    assert (tmp_path / "mymodel.json.v1").is_file()
    assert (tmp_path / "mymodel.pt.v1").is_file()
    assert not (tmp_path / "mymodel.json").is_file()
    assert not (tmp_path / "mymodel.pt").is_file()

    with pytest.raises(FileNotFoundError):
        # No version specified
        seisbench.models.GPD.load(tmp_path / "mymodel")

    # Call just passes
    seisbench.models.GPD.load(tmp_path / "mymodel", version_str="1")


def test_get_weights_file_paths():
    # Without suffix
    path_json, path_pt = seisbench.models.GPD._get_weights_file_paths(
        "path", version_str=None
    )
    assert path_json == Path("path.json")
    assert path_pt == Path("path.pt")

    # With suffix
    path_json, path_pt = seisbench.models.GPD._get_weights_file_paths(
        "path", version_str="3rc2"
    )
    assert path_json == Path("path.json.v3rc2")
    assert path_pt == Path("path.pt.v3rc2")


def test_list_pretrained_version_empty_cache(tmp_path):
    with patch(
        "seisbench.cache_root", tmp_path / "list_pretrained"
    ):  # Ensure SeisBench cache is empty
        seisbench.models.GPD.list_pretrained(details=True, remote=False)

    with patch(
        "seisbench.cache_root", tmp_path / "list_versions"
    ):  # Ensure SeisBench cache is empty
        seisbench.models.GPD.list_versions("original", remote=False)
