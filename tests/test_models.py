import asyncio
import inspect
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import obspy
import pytest
import torch
from obspy import UTCDateTime

import seisbench
import seisbench.models
import seisbench.util as sbu
from seisbench.models.base import ActivationLSTMCell, CustomLSTM
from seisbench.models.team import AlphabeticFullGroupingHelper


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
    t0_out, data, stations = dummy.stream_to_array(stream, {})
    assert t0_out == t0
    assert stations == ["SB.TEST."]
    assert data.shape == (3, len(trace_z.data))
    assert (data[0] == trace_z.data).all()
    assert (data[1] == trace_n.data).all()
    assert (data[2] == trace_e.data).all()


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

    stream = obspy.Stream([trace_z])
    t0_out, data, stations = dummy.stream_to_array(stream, {})
    assert t0_out == t0
    print(stations)
    assert stations == ["SB.TEST..HHZ"]
    assert data.shape == (len(trace_z.data),)
    assert (data == trace_z.data).all()


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
    _, data, stations = dummy.stream_to_array(
        stream, {"flexible_horizontal_components": False}
    )
    assert np.allclose(data[0, :], 1)
    assert np.allclose(data[1, :], 0)
    assert np.allclose(data[2, :], 0)

    # flexible_horizontal_components=True
    stream = obspy.Stream([trace_z, trace_1, trace_2])
    _, data, stations = dummy.stream_to_array(
        stream, {"flexible_horizontal_components": True}
    )
    assert np.allclose(data[0, :], 1)
    assert np.allclose(data[1, :], 4)
    assert np.allclose(data[2, :], 5)

    # Warning for mixed component names
    caplog.clear()
    stream = obspy.Stream([trace_z, trace_n, trace_e, trace_1, trace_2])
    with caplog.at_level(logging.WARNING):
        dummy.stream_to_array(stream, {"flexible_horizontal_components": True})
    assert "This might lead to undefined behavior." in caplog.text

    # No warning for mixed component names on different stations
    caplog.clear()
    stream = obspy.Stream([trace_z, trace_n, trace_e, trace_2_test2])
    with caplog.at_level(logging.WARNING):
        dummy.stream_to_array(stream, {"flexible_horizontal_components": True})
    assert "This might lead to undefined behavior." not in caplog.text


def test_group_stream_by_instrument():
    # The first 4 should be grouped together, the last 2 should each be separate
    stream = obspy.Stream(
        [
            obspy.Trace(
                np.ones(100),
                header={"network": "SB", "station": "ABC1", "channel": "BHZ"},
            ),
            obspy.Trace(
                np.ones(100),
                header={"network": "SB", "station": "ABC1", "channel": "BHN"},
            ),
            obspy.Trace(
                np.ones(100),
                header={"network": "SB", "station": "ABC1", "channel": "BHE"},
            ),
            obspy.Trace(
                np.ones(100),
                header={"network": "SB", "station": "ABC1", "channel": "HHZ"},
            ),
            obspy.Trace(
                np.ones(100),
                header={"network": "HB", "station": "ABC1", "channel": "BHZ"},
            ),
            obspy.Trace(
                np.ones(100),
                header={"network": "SB", "station": "ABC2", "channel": "BHZ"},
            ),
        ]
    )

    helper = seisbench.models.base.GroupingHelper("instrument")

    comp_dict = {"Z": 0, "N": 1, "E": 2}

    groups = helper.group_stream(stream, False, 0, comp_dict)

    assert len(groups) == 3
    assert list(sorted([len(x) for x in groups])) == [1, 1, 4]

    helper = seisbench.models.base.GroupingHelper("channel")
    groups = helper.group_stream(stream, False, 0, comp_dict)

    assert len(groups) == 6
    assert list(sorted([len(x) for x in groups])) == [1, 1, 1, 1, 1, 1]

    with pytest.raises(ValueError):
        seisbench.models.base.GroupingHelper("invalid")


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
    stations = ["SB.ABC1."]

    stream = dummy._predictions_to_stream(
        pred_rates[0], pred_times[0], preds[0], stations
    )
    stream += dummy._predictions_to_stream(
        pred_rates[1], pred_times[1], preds[1], stations
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
    elem = (UTCDateTime(), data[0], ["A", "B"])

    out = [x[0] for x in dummy._cut_fragments_array(elem, argdict)]

    assert len(out) == 12
    assert out[0].shape == (3, 1000)


def test_reassemble_blocks_array():
    dummy = DummyWaveformModel(
        component_order="ZNE", in_samples=1000, sampling_rate=100, pred_sample=(0, 1000)
    )

    trace_stats = obspy.read()[0].stats

    t0 = UTCDateTime()
    out = []
    argdict = {"stride": 100, "sampling_rate": 100}
    buffer = defaultdict(list)

    starts = [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9000, 9001]

    for i in range(12):
        elem = (
            np.ones((1000, 3)),
            ("key", t0, starts[i], 12, trace_stats, 0, 1000, (0, 1000)),
        )
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

    t0 = UTCDateTime()
    trace_stats = obspy.read()[0].stats

    for stacking in {"max", "avg"}:
        out = []
        argdict = {"overlap": 100, "stacking": stacking, "sampling_rate": 100}
        buffer = defaultdict(list)

        starts = [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9000, 9001]

        for i in range(12):
            elem = (
                np.ones((1000, 3)) + i,
                ("key", t0, starts[i], 12, trace_stats, 0, 1000, (0, 1000)),
            )
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

    with patch("seisbench.__version__", "1.2.3"):
        # Minimum version
        weights_metadata = {"seisbench_requirement": seisbench.__version__}
        model._check_version_requirement(weights_metadata)

        # Newer version
        weights_metadata = {"seisbench_requirement": seisbench.__version__ + "1"}
        with pytest.raises(ValueError):
            model._check_version_requirement(weights_metadata)

        # Older version
        version = seisbench.__version__
        version = version[:-1] + chr(ord(version[-1]) - 1)
        weights_metadata = {"seisbench_requirement": version}
        model._check_version_requirement(weights_metadata)


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


def test_annotate_cred():
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.CRED(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.detections, sbu.DetectionList)
    assert output.creator == model.name


def test_annotate_eqtransformer():
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.EQTransformer(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert isinstance(output.detections, sbu.DetectionList)
    assert output.creator == model.name


@pytest.mark.parametrize(
    "model",
    [
        "phasenet",
        "eqtransformer",
    ],
)
def test_annotate_pickblue(model):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    with patch(
        "seisbench.models.SeisBenchModel._check_version_requirement"
    ):  # Ignore version requirement
        model = seisbench.models.PickBlue(base=model)

    model.sampling_rate = 400  # Higher sampling rate ensures trace is long enough

    stream = obspy.read("./tests/examples/OBS*")
    annotations = model.annotate(stream)
    assert len(annotations) == 3
    model.classify(
        stream,
    )  # Ensures classify succeeds even though labels are unknown


def test_annotate_gpd():
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.GPD(
        sampling_rate=100
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


def test_annotate_phasenetlight():
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.PhaseNetLight(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


@pytest.mark.parametrize("filter_factor", [1, 2])
def test_annotate_phasenet(filter_factor):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.PhaseNet(
        sampling_rate=400,
        filter_factor=filter_factor,
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


def test_annotate_overlap():
    # Tests that the annotate function works the same with fractional and sample overlap
    model = seisbench.models.PhaseNet(
        sampling_rate=400,
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations1 = model.annotate(stream, overlap=0.5)
    annotations2 = model.annotate(stream, overlap=int(0.5 * model.in_samples))
    assert len(annotations1) == len(annotations2)
    for t1, t2 in zip(annotations1, annotations2):
        assert (t1.data == t2.data).all()


@pytest.mark.parametrize("output_activation", ["sigmoid", "softmax"])
@pytest.mark.parametrize("norm", ["std", "peak"])
@pytest.mark.parametrize("in_samples", [1337, 3001, 6000])
def test_annotate_variablelengthphasenet(in_samples, norm, output_activation):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.VariableLengthPhaseNet(
        sampling_rate=400,
        in_samples=in_samples,
        norm=norm,
        output_activation=output_activation,
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


def test_annotate_basicphaseae():
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.BasicPhaseAE(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


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


def test_annotate_deep_denoiser():
    stream = obspy.read()

    model = seisbench.models.DeepDenoiser()
    annotations = model.annotate(stream)

    assert len(annotations) == 3
    for i in range(3):
        assert annotations[i].data.shape == (3000,)


def test_annotate_seisdae():
    stream = obspy.read()
    model = seisbench.models.SeisDAE(in_samples=3000)
    annotations = model.annotate(stream)

    assert len(annotations) == 3
    for i in range(3):
        assert annotations[i].data.shape == (3000,)


def test_eqtransformer_annotate_batch_post():
    model = seisbench.models.EQTransformer()

    pred = 3 * [torch.ones(5, 1000)]

    # No blinding
    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (0, 0)}
    ).numpy()
    assert (blinded == 1).all()

    # Front blinding
    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (100, 0)}
    ).numpy()
    assert np.isnan(blinded[:, :100]).all()
    assert (blinded[:, 100:] == 1).all()

    # End blinding
    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (0, 100)}
    ).numpy()
    assert (blinded[:, :900] == 1).all()
    assert np.isnan(blinded[:, 900:]).all()

    # Two sided blinding
    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (100, 100)}
    ).numpy()
    assert np.isnan(blinded[:, :100]).all()
    assert (blinded[:, 100:900] == 1).all()
    assert np.isnan(blinded[:, 900:]).all()


def test_phasenet_annotate_batch_post():
    model = seisbench.models.PhaseNet()

    # Default: No blinding
    pred = torch.ones((5, 3, 1000))
    blinded = model.annotate_batch_post(pred, None, argdict={}).numpy()
    assert (blinded == 1).all()

    # No blinding
    pred = torch.ones((5, 3, 1000))
    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (0, 0)}
    ).numpy()
    assert (blinded == 1).all()

    # Front blinding
    pred = torch.ones((5, 3, 1000))
    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (100, 0)}
    ).numpy()
    assert np.isnan(blinded[:, :100]).all()
    assert (blinded[:, 100:] == 1).all()

    # End blinding
    pred = torch.ones((5, 3, 1000))
    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (0, 100)}
    ).numpy()
    assert (blinded[:, :900] == 1).all()
    assert np.isnan(blinded[:, 900:]).all()

    # Two sided blinding
    pred = torch.ones((5, 3, 1000))
    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (100, 100)}
    ).numpy()
    assert np.isnan(blinded[:, :100]).all()
    assert (blinded[:, 100:900] == 1).all()
    assert np.isnan(blinded[:, 900:]).all()


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
    model_orig = seisbench.models.PhaseNet(norm="peak", sampling_rate=400)
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
    pred_orig = model_orig.annotate(stream)
    pred_load = model_load.annotate(stream)

    for i in range(len(pred_orig)):
        assert np.allclose(pred_orig[i].data, pred_load[i].data)
    assert model_orig_args == model_load_args


def test_save_load_phasenetlight(tmp_path):
    model_orig = seisbench.models.PhaseNetLight(norm="peak", sampling_rate=400)
    model_orig_args = get_input_args(model_orig.__class__)

    # Test model saving
    model_orig.save(tmp_path / "phasenet")
    assert (tmp_path / "phasenet.json").exists()
    assert (tmp_path / "phasenet.pt").exists()

    stream = obspy.read()

    # Test model loading
    model_load = seisbench.models.PhaseNetLight.load(tmp_path / "phasenet")
    model_load_args = get_input_args(model_orig.__class__)

    # Test no changes to weights
    pred_orig = model_orig.annotate(stream)
    pred_load = model_load.annotate(stream)

    for i in range(len(pred_orig)):
        assert np.allclose(pred_orig[i].data, pred_load[i].data)
    assert model_orig_args == model_load_args


def test_save_load_eqtransformer(tmp_path):
    model_orig = seisbench.models.EQTransformer(norm="peak", sampling_rate=400)
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
    pred_orig = model_orig.annotate(stream)
    pred_load = model_load.annotate(stream)

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
    model_orig.filter_kwargs = {"freqmin": 1, "freqmax": 10, "zerophase": True}

    model_orig.save(tmp_path / "eqtransformer_changed")
    assert (tmp_path / "eqtransformer_changed.json").exists()
    assert (tmp_path / "eqtransformer_changed.pt").exists()

    # Test model loading w. updated params
    model_load = seisbench.models.EQTransformer.load(tmp_path / "eqtransformer_changed")
    model_load_state = model_load.__dict__

    assert model_load_state["in_samples"] == 10_000
    assert model_load_state["sampling_rate"] == 500
    assert model_load_state["filter_kwargs"] == {
        "freqmin": 1,
        "freqmax": 10,
        "zerophase": True,
    }


def test_save_load_model_updated_after_construction_inheritence_compatible(tmp_path):
    model_orig = seisbench.models.GPD(component_order="NZE")

    model_orig.save(tmp_path / "gpd_changed")
    model_loaded = model_orig.load(tmp_path / "gpd_changed")

    assert model_orig.component_order == model_loaded.component_order


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


def test_list_pretrained404(tmp_path):
    with patch("seisbench.models.GPD._model_path") as model_path:
        model_path.return_value = tmp_path

        def raise404(*args, **kwargs):
            raise ValueError(f"Invalid URL. Request returned status code 404.")

        with patch("seisbench.util.ls_webdav") as ls_webdav:
            ls_webdav.side_effect = raise404

            assert (
                seisbench.models.GPD.list_pretrained(details=False, remote=True) == []
            )

        def raise505(*args, **kwargs):
            raise ValueError(f"Invalid URL. Request returned status code 505.")

        with patch("seisbench.util.ls_webdav") as ls_webdav:
            ls_webdav.side_effect = raise505

            with pytest.raises(ValueError):
                seisbench.models.GPD.list_pretrained(details=False, remote=True)


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
        "seisbench.cache_model_root", tmp_path / "list_pretrained"
    ):  # Ensure SeisBench cache is empty
        seisbench.models.GPD.list_pretrained(details=True, remote=False)

    with patch(
        "seisbench.cache_model_root", tmp_path / "list_versions"
    ):  # Ensure SeisBench cache is empty
        seisbench.models.GPD.list_versions("original", remote=False)


def test_verify_argdict(caplog):
    model = seisbench.models.GPD()

    # No wildcard - Matching
    model._annotate_args = {"param": ("", 0)}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model._verify_argdict({"param": 3})
    assert caplog.text == ""

    # Wildcard - Matching
    model._annotate_args = {"*_param": ("", 0)}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model._verify_argdict({"my_param": 3})
    assert caplog.text == ""

    # No wildcard - Not matching
    model._annotate_args = {"param": ("", 0)}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model._verify_argdict({"not_param": 3})
    assert "Unknown argument" in caplog.text

    # Wildcard - Not matching
    model._annotate_args = {"*_param": ("", 0)}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model._verify_argdict({"my_var": 3})
    assert "Unknown argument" in caplog.text


def test_annotate_filter():
    model = seisbench.models.GPD()

    # Nothing happens
    stream_org = obspy.read()
    stream = stream_org.copy()
    model.filter_args = None
    model.filter_kwargs = None
    model._filter_stream(stream)

    for trace_a, trace_b in zip(stream_org, stream):
        assert np.allclose(trace_a.data, trace_b.data)

    # Filter all
    stream_org = obspy.read()
    stream = stream_org.copy()
    model.filter_args = ["highpass"]
    model.filter_kwargs = {"freq": 1}
    model._filter_stream(stream)

    for trace_a, trace_b in zip(stream_org, stream):
        assert not np.allclose(trace_a.data, trace_b.data)

    # Filter Z only
    stream_org = obspy.read()
    stream = stream_org.copy()
    model.filter_args = {"??Z": ["highpass"]}
    model.filter_kwargs = {"??Z": {"freq": 1}}
    model._filter_stream(stream)

    for trace_a, trace_b in zip(stream_org, stream):
        if trace_a.stats.channel[-1] == "Z":
            assert not np.allclose(trace_a.data, trace_b.data)
        else:
            assert np.allclose(trace_a.data, trace_b.data)

    # Invalid filter
    model.filter_args = {"??Z": ["highpass"]}
    model.filter_kwargs = {"??Y": {"freq": 1}}

    with pytest.raises(ValueError) as e:
        model._filter_stream(stream)

    assert "Invalid filter definition" in str(e)


def test_phasenet_forward():
    model = seisbench.models.PhaseNet()
    x = torch.rand((2, 3, 3001))

    with torch.no_grad():
        pred = model(x).numpy()
    assert np.allclose(np.sum(pred, axis=1), 1)

    with torch.no_grad():
        pred = model(x, logits=True).numpy()
    assert not np.allclose(np.sum(pred, axis=1), 1)


def test_phasenetlight_forward():
    model = seisbench.models.PhaseNetLight()
    x = torch.rand((2, 3, 3001))

    with torch.no_grad():
        pred = model(x).numpy()
    assert np.allclose(np.sum(pred, axis=1), 1)

    with torch.no_grad():
        pred = model(x, logits=True).numpy()
    assert not np.allclose(np.sum(pred, axis=1), 1)


def test_basicphaseae_forward():
    model = seisbench.models.BasicPhaseAE()
    x = torch.rand((2, 3, 600))

    with torch.no_grad():
        pred = model(x).numpy()
    assert np.allclose(np.sum(pred, axis=1), 1)

    with torch.no_grad():
        pred = model(x, logits=True).numpy()
    assert not np.allclose(np.sum(pred, axis=1), 1)


def test_gpd_forward():
    model = seisbench.models.GPD()
    x = torch.rand((2, 3, 400))

    with torch.no_grad():
        pred = model(x).numpy()
    assert np.allclose(np.sum(pred, axis=1), 1)

    with torch.no_grad():
        pred = model(x, logits=True).numpy()
    assert not np.allclose(np.sum(pred, axis=1), 1)


def test_dpp_forward():
    model = seisbench.models.DPPDetector()
    x = torch.rand((2, 3, 500))

    with torch.no_grad():
        pred = model(x).numpy()
    assert np.allclose(np.sum(pred, axis=1), 1)

    with torch.no_grad():
        pred = model(x, logits=True).numpy()
    assert not np.allclose(np.sum(pred, axis=1), 1)


def test_eqtransformer_forward():
    model = seisbench.models.EQTransformer()
    x = torch.rand((2, 3, 6000))

    with torch.no_grad():
        pred = [p.numpy() for p in model(x)]

    for p in pred:
        assert np.all(np.logical_and(0 <= p, p <= 1))

    with torch.no_grad():
        pred_logit = [p.numpy() for p in model(x, logits=True)]

    for p, pl in zip(pred, pred_logit):
        assert not np.allclose(p, pl)


def test_cred_forward():
    model = seisbench.models.CRED()
    x = np.random.rand(2, 3, 3000)
    x = model.waveforms_to_spectrogram(torch.tensor(x, dtype=torch.float32))

    with torch.no_grad():
        pred = model(x).numpy()

    assert np.all(np.logical_and(0 <= pred, pred <= 1))

    with torch.no_grad():
        pred_logit = model(x, logits=True).numpy()

    assert not np.allclose(pred, pred_logit)


def test_argdict_get_with_default():
    model = seisbench.models.PhaseNet()
    model._annotate_args["testarg"] = ("Test docstr", 1)

    assert model._argdict_get_with_default({"testarg": 2}, "testarg") == 2
    assert model._argdict_get_with_default({"not_testarg": 2}, "testarg") == 1


@pytest.mark.parametrize(
    "cls",
    [
        seisbench.models.PhaseNet,
        seisbench.models.EQTransformer,
        seisbench.models.PhaseNetLight,
    ],
)
def test_model_normalize(cls):
    # Tolerance is set rather high to mitigate EQTransformer taper
    model = cls()

    model.norm = "std"
    x = np.random.rand(5, 3, 100000)
    x_norm = model.annotate_batch_pre(torch.tensor(x), {}).numpy()
    assert np.allclose(np.std(x_norm, axis=-1), 1, atol=1e-2, rtol=1e-2)

    model.norm = "peak"
    x = np.random.rand(5, 3, 100000)
    x_norm = model.annotate_batch_pre(torch.tensor(x), {}).numpy()
    assert np.allclose(np.max(np.abs(x_norm), axis=-1), 1, atol=1e-2, rtol=1e-2)


def test_version_warnings(caplog):
    class MockModel(seisbench.models.WaveformModel):
        _weight_warnings = [
            ("abc|def", "2", "MYWARN"),  # Problematic
            ("xyz", "3", "MYWARN"),  # Fine
        ]

    def check_version(name, version_str, warns):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            MockModel._version_warnings(name, version_str)

        if warns:
            assert "MYWARN" in caplog.text
        else:
            assert "MYWARN" not in caplog.text

    check_version("abc", "2", True)
    check_version("def", "2", True)
    check_version("xyz", "3", True)

    check_version("abc", "1", False)
    check_version("def", "3", False)
    check_version("xyz", "2", False)


def test_phaseteam():
    model = seisbench.models.PhaseTEAM(classes=4)

    x = torch.rand(2, 10, 3, 3001)
    # Make sure there is some padding
    x[0, 5:] = 0.0
    x[1, 6:] = 0.0

    y = model(x)

    assert y.shape == (2, 10, 4, 3001)


def test_get_intervals():
    helper = seisbench.models.GroupingHelper("full")

    comp_dict = {"Z": 0, "N": 1, "E": 2}

    t0 = UTCDateTime("2000-01-01")

    # Easy example, none-intersecting traces, single component
    stream = obspy.Stream(
        [
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC1",
                    "channel": "BHZ",
                    "sampling_rate": 20,
                    "starttime": t0,
                },
            ),
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC2",
                    "channel": "HHE",
                    "sampling_rate": 20,
                    "starttime": t0 + 100,
                },
            ),
        ]
    )

    intervals = helper._get_intervals(
        stream, strict=False, comp_dict=comp_dict, min_length_s=0
    )
    assert len(intervals) == 2
    assert intervals[0] == (
        ["SB.ABC1."],
        stream[0].stats.starttime,
        stream[0].stats.endtime,
    )
    assert intervals[1] == (
        ["SB.ABC2."],
        stream[1].stats.starttime,
        stream[1].stats.endtime,
    )
    intervals = helper._get_intervals(
        stream, strict=True, comp_dict=comp_dict, min_length_s=0
    )
    assert len(intervals) == 0

    # Easy example, none-intersecting traces, three components
    stream = obspy.Stream(
        [
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC1",
                    "channel": "BHZ",
                    "sampling_rate": 20,
                    "starttime": t0,
                },
            ),
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC1",
                    "channel": "BHN",
                    "sampling_rate": 20,
                    "starttime": t0,
                },
            ),
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC1",
                    "channel": "BHE",
                    "sampling_rate": 20,
                    "starttime": t0,
                },
            ),
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC2",
                    "channel": "HHZ",
                    "sampling_rate": 20,
                    "starttime": t0 + 100,
                },
            ),
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC2",
                    "channel": "HHE",
                    "sampling_rate": 20,
                    "starttime": t0 + 100,
                },
            ),
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC2",
                    "channel": "HHN",
                    "sampling_rate": 20,
                    "starttime": t0 + 100,
                },
            ),
        ]
    )

    for strict in [True, False]:
        intervals = helper._get_intervals(
            stream, strict=strict, comp_dict=comp_dict, min_length_s=0
        )
        assert len(intervals) == 2
        assert intervals[0] == (
            ["SB.ABC1."],
            stream[0].stats.starttime,
            stream[0].stats.endtime,
        )
        assert intervals[1] == (
            ["SB.ABC2."],
            stream[3].stats.starttime,
            stream[3].stats.endtime,
        )

    # Example with intersections
    stream = obspy.Stream(
        [
            obspy.Trace(
                np.zeros(2000),
                header={
                    "network": "SB",
                    "station": "ABC1",
                    "channel": "BHZ",
                    "sampling_rate": 20,
                    "starttime": t0,
                },
            ),
            obspy.Trace(
                np.zeros(100),
                header={
                    "network": "SB",
                    "station": "ABC2",
                    "channel": "HHE",
                    "sampling_rate": 20,
                    "starttime": t0 + 10,
                },
            ),
        ]
    )

    intervals = helper._get_intervals(
        stream, strict=False, comp_dict=comp_dict, min_length_s=0
    )
    assert len(intervals) == 3
    assert intervals[0] == (
        ["SB.ABC1."],
        stream[0].stats.starttime,
        stream[1].stats.starttime,
    )
    assert intervals[1] == (
        ["SB.ABC1.", "SB.ABC2."],
        stream[1].stats.starttime,
        stream[1].stats.endtime,
    )
    assert intervals[2] == (
        ["SB.ABC1."],
        stream[1].stats.endtime,
        stream[0].stats.endtime,
    )


def test_assemble_groups():
    helper = seisbench.models.GroupingHelper("full")

    t0 = UTCDateTime("2000-01-01")
    stream = obspy.Stream(
        [
            obspy.Trace(
                np.zeros(1000),
                header={
                    "network": "SB",
                    "station": "ABC1",
                    "channel": "BHZ",
                    "sampling_rate": 20,
                    "starttime": t0,
                },
            ),
            obspy.Trace(
                np.zeros(1000),
                header={
                    "network": "SB",
                    "station": "ABC1",
                    "channel": "BHN",
                    "sampling_rate": 20,
                    "starttime": t0,
                },
            ),
            obspy.Trace(
                np.zeros(1000),
                header={
                    "network": "SB",
                    "station": "ABC1",
                    "channel": "BHE",
                    "sampling_rate": 20,
                    "starttime": t0,
                },
            ),
            obspy.Trace(
                np.zeros(1000),
                header={
                    "network": "SB",
                    "station": "ABC2",
                    "channel": "HHZ",
                    "sampling_rate": 20,
                    "starttime": t0 + 2,
                },
            ),
            obspy.Trace(
                np.zeros(1000),
                header={
                    "network": "SB",
                    "station": "ABC2",
                    "channel": "HHE",
                    "sampling_rate": 20,
                    "starttime": t0 + 2,
                },
            ),
            obspy.Trace(
                np.zeros(1000),
                header={
                    "network": "SB",
                    "station": "ABC2",
                    "channel": "HHN",
                    "sampling_rate": 20,
                    "starttime": t0 + 2,
                },
            ),
        ]
    )

    intervals = [
        (["SB.ABC1..BH"], t0, t0 + 5),
        (["SB.ABC1..BH", "SB.ABC2..HH"], t0 + 10, t0 + 20),
    ]
    groups = helper._assemble_groups(stream, intervals)
    print(groups)

    assert len(groups[0]) == 3
    for trace in groups[0]:
        assert trace.stats.starttime == t0
        assert trace.stats.endtime == t0 + 5
    assert len(groups[1]) == 6
    for trace in groups[1]:
        assert trace.stats.starttime == t0 + 10
        assert trace.stats.endtime == t0 + 20


def test_merge_intervals():
    helper = seisbench.models.GroupingHelper("full")
    intervals = [
        (0, -5, 1),
        (0, 5, 10),
        (1, 0.5, 3),
        (1, 5, 10),
        (2, 2, 6),
        (2, 20, 25),
    ]

    t_root = obspy.UTCDateTime(0)

    comp_dict = {"Z": 0}
    sampling_rate = 100

    stream = obspy.Stream()
    for sta, t0, t1 in intervals:
        samples = int((t1 - t0) * sampling_rate) + 1
        stream.append(
            obspy.Trace(
                np.zeros(samples),
                header={
                    "network": "SB",
                    "station": f"ABC{sta}",
                    "channel": "BHZ",
                    "sampling_rate": sampling_rate,
                    "starttime": t_root + t0,
                },
            )
        )

    selected = helper._get_intervals(stream, False, 2, comp_dict)
    assert selected == [
        (["SB.ABC0."], t_root - 5, t_root + 1),
        (["SB.ABC1."], t_root + 1, t_root + 3),
        (["SB.ABC2."], t_root + 3, t_root + 5),
        (["SB.ABC0.", "SB.ABC1."], t_root + 5, t_root + 10),
        (["SB.ABC2."], t_root + 20, t_root + 25),
    ]

    selected = helper._get_intervals(stream, False, 4, comp_dict)
    assert selected == [
        (["SB.ABC0."], t_root - 5, t_root + 1),
        (["SB.ABC2."], t_root + 2, t_root + 6),
        (["SB.ABC0.", "SB.ABC1."], t_root + 6, t_root + 10),
        (["SB.ABC2."], t_root + 20, t_root + 25),
    ]


def test_split_groups():
    helper = AlphabeticFullGroupingHelper(max_stations=10)

    stations = [f"SB.AB{i:02d}" for i in range(15)]
    np.random.shuffle(stations)
    t0 = UTCDateTime()
    intervals = [(stations, t0, t0 + 10)]

    intervals = helper._split_groups(intervals)

    assert len(intervals) == 2
    assert len(intervals[0][0]) == 8
    assert intervals[0][0] == [f"SB.AB{i:02d}" for i in range(8)]
    assert len(intervals[1][0]) == 7
    assert intervals[1][0] == [f"SB.AB{i:02d}" for i in range(8, 15)]
    assert intervals[0][1] == intervals[1][1] == t0
    assert intervals[0][2] == intervals[1][2] == t0 + 10


def test_align_fractional_samples():
    helper = seisbench.models.GroupingHelper("full")

    stream = obspy.Stream(
        [
            obspy.Trace(header={"starttime": UTCDateTime(0.01), "sampling_rate": 100}),
            obspy.Trace(header={"starttime": UTCDateTime(0.014), "sampling_rate": 100}),
            obspy.Trace(header={"starttime": UTCDateTime(0.006), "sampling_rate": 100}),
        ]
    )

    helper._align_fractional_samples(stream)

    for trace in stream:
        assert trace.stats.starttime == UTCDateTime(0.01)


def test_get_intervals_empty():
    helper = seisbench.models.GroupingHelper("instrument")

    helper._get_intervals(
        obspy.Stream(), strict=False, min_length_s=1, comp_dict={"Z": 0}
    )

    trace = obspy.Trace(np.zeros(0))
    helper._get_intervals(
        obspy.Stream([trace]), strict=False, min_length_s=1, comp_dict={"Z": 0}
    )


def test_phasenet_async():
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.PhaseNet(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = asyncio.run(model.annotate_async(stream))
    assert len(annotations) > 0
    output = asyncio.run(
        model.classify_async(stream)
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


def test_annotate_empty():
    model = seisbench.models.PhaseNet()
    ann = model.annotate(obspy.Stream())
    assert len(ann) == 0


def test_overlap_mismatching_records_empty():
    model = seisbench.models.PhaseNet()

    header = {
        "network": "XX",
        "station": "STA",
        "location": "00",
        "channel": f"HHZ",
        "sampling_rate": 100.0,
    }

    trace1 = obspy.Trace(np.ones(10000), header=header)
    trace2 = obspy.Trace(np.zeros(10000), header=header)

    ann = model.annotate(obspy.Stream([trace1, trace2]))

    assert len(ann) == 0


def test_predict_buffer_padding():
    # Note: Normally, for PhaseNet we don't want the model to do padding.
    model = seisbench.models.PhaseNet()

    buffer = [np.random.rand(i, 3001) for i in range(1, 4)]

    model.allow_padding = False
    with pytest.raises(ValueError):
        model._predict_buffer(buffer, {})

    model.allow_padding = True
    model._predict_buffer(buffer, {})


def test_annotate_obstransformer():
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = seisbench.models.OBSTransformer(
        sampling_rate=400
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


def test_fractional_sample_handling():
    t0 = UTCDateTime("2000-01-01 00:00:00")
    common_metadata = {
        "network": "XX",
        "station": "YYY",
        "location": "",
        "sampling_rate": 100,
    }
    stream = obspy.Stream(
        [
            obspy.Trace(
                np.ones(10000),
                header={
                    "starttime": t0 + i / 1000.0,
                    "channel": f"HH{c}",
                    **common_metadata,
                },
            )
            for i, c in enumerate("ZNE")
        ]
    )

    model = seisbench.models.PhaseNet(sampling_rate=100)

    ann = model.annotate(stream, blinding=(0, 0))
    for trace in ann:
        assert trace.stats.starttime == t0 + 1.0 / 1000.0


def test_dynamic_samples():
    class DynamicWaveformModel(seisbench.models.WaveformModel):
        def __init__(self):
            super().__init__(
                component_order="ZNE",
                output_type="array",
                in_samples=1024,
                pred_sample=(0, 1024),
                labels="PSN",
                sampling_rate=100,
            )

            self.shape_log = []

            # Required for device function
            self.layer = torch.nn.Linear(5, 5)

        def forward(self, x):
            return torch.ones((x.shape[0], 3, x.shape[-1]))

        def _get_in_pred_samples(
            self, block: np.ndarray
        ) -> tuple[int, tuple[int, int]]:
            in_samples = 2 ** int(
                np.log2(block.shape[-1])
            )  # The largest power of 2 below the block shape
            in_samples = min(
                max(in_samples, 2**10), 2**20
            )  # Enforce upper and lower bounds
            pred_sample = (0, in_samples)
            self.shape_log.append((in_samples, pred_sample))
            return in_samples, pred_sample

        def annotate_batch_post(
            self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
        ) -> torch.Tensor:
            # Transpose predictions to correct shape
            return torch.transpose(batch, -1, -2)

    model = DynamicWaveformModel()

    for n, target in zip(
        [1000, 1024, 5000, 2**20, 2**22], [1024, 1024, 4096, 2**20, 2**20]
    ):
        header_base = {
            "network": "XX",
            "station": "YY",
            "location": "",
            "sampling_rate": model.sampling_rate,
        }
        stream = obspy.Stream(
            [
                obspy.Trace(
                    np.random.randn(n), header={**header_base, "channel": f"HH{c}"}
                )
                for c in model.component_order
            ]
        )

        ann = model.annotate(stream)
        if n < 1024:  # Trace too short
            assert len(ann) == 0
        else:
            assert len(ann) == 3

            assert model.shape_log[-1] == (target, (0, target))

    # Annotate mixed stream, i.e., mixed window sizes
    header_base = {
        "network": "XX",
        "station": "YY",
        "location": "",
        "sampling_rate": model.sampling_rate,
    }
    stream1 = obspy.Stream(
        [
            obspy.Trace(
                np.random.randn(2000), header={**header_base, "channel": f"HH{c}"}
            )
            for c in model.component_order
        ]
    )
    header_base = {
        "network": "XX",
        "station": "ZZ",
        "location": "",
        "sampling_rate": model.sampling_rate,
    }
    stream2 = obspy.Stream(
        [
            obspy.Trace(
                np.random.randn(10000), header={**header_base, "channel": f"HH{c}"}
            )
            for c in model.component_order
        ]
    )
    assert len(model.annotate(stream1 + stream2)) == 6
