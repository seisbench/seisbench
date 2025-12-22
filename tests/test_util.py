import logging
import os
import pickle
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import obspy
import pandas as pd
import pytest
import requests
import scipy.signal
import torch
from obspy.clients.fdsn.header import FDSNException

import seisbench
import seisbench.util
from seisbench.util.trace_ops import (
    _round_py2,
    fdsn_get_bulk_safe,
    stream_slice,
    waveform_id_to_network_station_location,
)


def test_callback_if_uncached(tmp_path: Path):
    def callback(file):
        with open(file, "w") as f:
            f.write("test")

    # File is not cached and is created
    seisbench.util.callback_if_uncached(tmp_path / "dummy", callback)
    assert (tmp_path / "dummy").is_file()
    assert not (tmp_path / "dummy.partial").is_file()

    # Partial file exists, fail
    open(tmp_path / "dummy2.partial", "a").close()
    with pytest.raises(ValueError):
        seisbench.util.callback_if_uncached(tmp_path / "dummy2", callback)

    # Partial file exists, cleanup and callback
    open(tmp_path / "dummy3.partial", "a").close()
    seisbench.util.callback_if_uncached(tmp_path / "dummy3", callback, force=True)
    assert (tmp_path / "dummy3").is_file()
    assert not (tmp_path / "dummy3.partial").is_file()

    # Wait for file is called
    with patch("time.sleep") as time_sleep:

        def side_effect_raise(_):
            raise ValueError("Sleeping")

        time_sleep.side_effect = side_effect_raise
        open(tmp_path / "dummy4.partial", "a").close()
        with pytest.raises(ValueError) as e:
            seisbench.util.callback_if_uncached(
                tmp_path / "dummy4", callback, wait_for_file=True
            )
            assert str(e).find("Sleeping") != -1

    # Wait for file is called, downloads if partial is removed
    with patch("time.sleep") as time_sleep:

        def side_effect_remove(_):
            os.remove(tmp_path / "dummy5.partial")

        time_sleep.side_effect = side_effect_remove
        open(tmp_path / "dummy5.partial", "a").close()
        seisbench.util.callback_if_uncached(
            tmp_path / "dummy5", callback, wait_for_file=True
        )
        assert (tmp_path / "dummy5").is_file()
        assert not (tmp_path / "dummy5.partial").is_file()

    # Wait for file is called, returns if target file is created
    with patch("time.sleep") as time_sleep:

        def side_effect_write(_):
            with open(tmp_path / "dummy6", "w") as f:
                f.write("test")

        time_sleep.side_effect = side_effect_write
        open(tmp_path / "dummy6.partial", "a").close()
        seisbench.util.callback_if_uncached(
            tmp_path / "dummy6", callback, wait_for_file=True
        )
        assert (tmp_path / "dummy6").is_file()


def test_waveform_id_to_network_station_location():
    assert waveform_id_to_network_station_location("NET.STA.LOC.CHA") == "NET.STA.LOC"
    assert waveform_id_to_network_station_location("NET.STA..CHA") == "NET.STA."
    assert waveform_id_to_network_station_location("invalid") == "invalid"


def test_precheck_url(caplog):
    # Timeout
    with patch("requests.head") as head_mock:

        def side_effect_raise(*args, **kwargs):
            raise requests.Timeout()

        head_mock.side_effect = side_effect_raise

        with caplog.at_level(logging.WARNING):
            seisbench.util.precheck_url(seisbench.remote_root, timeout=5)
        assert "timeout" in caplog.text

    caplog.clear()

    # ConnectionError
    with patch("requests.head") as head_mock:

        def side_effect_raise(*args, **kwargs):
            raise requests.ConnectionError()

        head_mock.side_effect = side_effect_raise

        with caplog.at_level(logging.WARNING):
            seisbench.util.precheck_url(seisbench.remote_root, timeout=5)
        assert "connection error" in caplog.text

    caplog.clear()

    # 400+ response code
    with patch("requests.head") as head_mock:
        response_mock = MagicMock()
        response_mock.status_code = 400
        head_mock.return_value = response_mock

        with caplog.at_level(logging.WARNING):
            seisbench.util.precheck_url(seisbench.remote_root, timeout=5)
        assert "status code 400" in caplog.text


def test_log_lifecycle(caplog):
    @seisbench.util.log_lifecycle(logging.DEBUG)
    def test_func():
        pass

    with caplog.at_level(logging.DEBUG):
        test_func()

    assert "Starting test_func" in caplog.text
    assert "Stopping test_func" in caplog.text


def test_classify_outputs():
    output = seisbench.util.ClassifyOutput("model", picks=[])
    assert output.creator == "model"
    assert len(output.picks) == 0
    with pytest.raises(AttributeError):
        output.missing_key


def test_repr_entries():
    picks = seisbench.util.PickList(3 * [seisbench.util.Pick("CX.PB01.", None)])
    assert len(picks._rep_entries().split("\n")) == 3

    picks = seisbench.util.PickList(100 * [seisbench.util.Pick("CX.PB01.", None)])
    assert len(picks._rep_entries().split("\n")) == 7


def test_pick_list_select():
    picks = seisbench.util.PickList(
        [
            seisbench.util.Pick("CX.PB01.", None, peak_value=0.5, phase="P"),
            seisbench.util.Pick("CX.PB02.", None, peak_value=0.3, phase="S"),
            seisbench.util.Pick("CX.PB03.", None, peak_value=None, phase=None),
        ]
    )

    assert len(picks.select(phase="P")) == 1
    assert len(picks.select(phase="S")) == 1
    assert len(picks.select(min_confidence=0.1)) == 2
    assert len(picks.select(min_confidence=0.4)) == 1
    assert len(picks.select(trace_id=r"CX\.PB0[12]\.")) == 2


def test_detection_list_select():
    detections = seisbench.util.DetectionList(
        [
            seisbench.util.Detection("CX.PB01.", None, None, peak_value=0.5),
            seisbench.util.Detection("CX.PB02.", None, None, peak_value=0.3),
            seisbench.util.Detection("CX.PB03.", None, None, peak_value=None),
        ]
    )

    assert len(detections.select(min_confidence=0.1)) == 2
    assert len(detections.select(min_confidence=0.4)) == 1
    assert len(detections.select(trace_id=r"CX\.PB0[12]\.")) == 2


def test_classify_output_interface_error():
    output = seisbench.util.ClassifyOutput("model")
    with pytest.raises(NotImplementedError):
        iter(output)
    with pytest.raises(NotImplementedError):
        output[0]


def test_fdsn_get_bulk_safe():
    bulk = [
        ("SB", "ABC1", "", "HHZ", None, None),
        ("SB", "ABC2", "", "HHZ", None, None),
        ("SB", "ABC3", "", "HHZ", None, None),
        ("SB", "ABC4", "", "HHZ", None, None),
    ]

    class MockClient:
        @staticmethod
        def get_waveforms_bulk(bulk):
            stream = obspy.Stream()
            for net, sta, loc, cha, _, _ in bulk:
                if sta in ["ABC1", "ABC3"]:
                    raise FDSNException("")
                else:
                    stream.append(
                        obspy.Trace(
                            header={
                                "network": net,
                                "station": sta,
                                "location": loc,
                                "channel": cha,
                            }
                        )
                    )
            return stream

    stream = fdsn_get_bulk_safe(MockClient(), bulk)
    assert len(stream) == 2
    assert len(stream.select(station="ABC2")) == 1
    assert len(stream.select(station="ABC4")) == 1


def test_classify_output_pickle(tmp_path):
    output = seisbench.util.ClassifyOutput(creator="Test", bla="abc")
    path = tmp_path / "test.pkl"
    with open(path, "wb") as f:
        pickle.dump(output, f)

    with open(path, "rb") as f:
        reloaded = pickle.load(f)

    assert reloaded.creator == output.creator
    assert reloaded.bla == output.bla


def test_torch_detrend():
    x = np.random.rand(5, 3, 1000)

    y1 = seisbench.util.torch_detrend(torch.tensor(x)).numpy()
    y2 = scipy.signal.detrend(x)

    np.testing.assert_allclose(y1, y2)


def test_pad_packed_sequence():
    seq = [np.ones((5, 1)), np.ones((6, 3)), np.ones((1, 2)), np.ones((7, 2))]

    packed = seisbench.util.pad_packed_sequence(seq)

    assert packed.shape == (4, 7, 3)
    assert np.sum(packed == 1) == sum(x.size for x in seq)
    assert np.sum(packed == 0) == packed.size - sum(x.size for x in seq)


def test_pick_list_to_dataframe():
    picks = seisbench.util.PickList()
    assert len(picks.to_dataframe()) == 0

    t0 = obspy.UTCDateTime()
    t1 = t0 + 10

    picks = seisbench.util.PickList(
        [
            seisbench.util.Pick("XX.YY.Z1", t0, t0 + 10, t0 + 5, 0.9, "P"),
            seisbench.util.Pick("XX.YY.Z2", t1, t1 + 10, t1 + 5, 0.6, "S"),
        ]
    )
    pick_df = picks.to_dataframe()

    assert len(pick_df) == 2
    np.testing.assert_allclose(
        (pick_df["end_time"] - pick_df["time"]) / pd.Timedelta(1, "s"), 5
    )
    np.testing.assert_allclose(
        (pick_df["time"] - pick_df["start_time"]) / pd.Timedelta(1, "s"), 5
    )
    assert all(pick_df["station"] == ["XX.YY.Z1", "XX.YY.Z2"])
    assert all(pick_df["phase"] == ["P", "S"])
    assert all(pick_df["probability"] == [0.9, 0.6])


def test_detection_list_to_dataframe():
    detections = seisbench.util.DetectionList()
    assert len(detections.to_dataframe()) == 0

    t0 = obspy.UTCDateTime()
    t1 = t0 + 10

    detections = seisbench.util.DetectionList(
        [
            seisbench.util.Detection("XX.YY.Z1", t0, t0 + 5, 0.9),
            seisbench.util.Detection("XX.YY.Z2", t1, t1 + 5, 0.6),
        ]
    )
    detection_df = detections.to_dataframe()

    assert len(detection_df) == 2
    np.testing.assert_allclose(
        (detection_df["end_time"] - detection_df["start_time"]) / pd.Timedelta(1, "s"),
        5,
    )
    assert all(detection_df["station"] == ["XX.YY.Z1", "XX.YY.Z2"])
    assert all(detection_df["probability"] == [0.9, 0.6])


def test_round_away():
    from obspy.core.compatibility import round_away

    arr = np.random.uniform(-1, 1, size=100_000)

    for val in arr:
        assert _round_py2(val) == round_away(val)

    assert round_away(0.5) == _round_py2(0.5)
    assert round_away(-0.5) == _round_py2(-0.5)


@pytest.mark.parametrize("implementation", ["obspy", "seisbench"])
@pytest.mark.parametrize("sampling_rate", [100, 200])
def test_slice(
    benchmark,
    sampling_rate: int,
    implementation: Literal["obspy", "seisbench"],
):
    stream = obspy.Stream()
    n_traces = 100
    length = 60.0 * 5.0  # seconds
    n_samples = int(length * sampling_rate)
    slice_length = 30.0  # seconds

    for i in range(n_traces):
        random_begin = np.round(np.random.uniform(0, 60.0), decimals=3)  # seconds
        trace = obspy.Trace(
            np.random.uniform(-100, 100, size=n_samples).astype(np.float32),
            header={
                "network": "XX",
                "station": f"STA{i:03d}",
                "location": "",
                "channel": "BHZ",
                "starttime": obspy.UTCDateTime(2025, 1, 1) + random_begin,
                "sampling_rate": sampling_rate,
            },
        )
        stream.append(trace)

    st_starttime = min(tr.stats.starttime for tr in stream)
    st_endtime = max(tr.stats.endtime for tr in stream)

    def slice_obspy():
        groups = []
        begin = st_starttime

        while begin < st_endtime:
            cut_start = begin
            cut_end = cut_start + slice_length
            sliced = stream.slice(cut_start, cut_end)
            groups.append(sliced)
            begin += slice_length
        return groups

    def slice_sb():
        groups = []
        begin = st_starttime

        while begin < st_endtime:
            cut_start = begin
            cut_end = cut_start + slice_length
            sliced = stream_slice(stream, cut_start, cut_end)
            groups.append(sliced)
            begin += slice_length
        return groups

    benchmark.group = f"slicing_{sampling_rate}hz"

    sb_slices = benchmark(slice_sb) if implementation == "seisbench" else slice_sb()
    obspy_slices = (
        benchmark(slice_obspy) if implementation == "obspy" else slice_obspy()
    )

    assert len(obspy_slices) == len(sb_slices)

    for traces_obspy, traces_sb in zip(obspy_slices, sb_slices, strict=True):
        traces_obspy.sort()
        traces_sb.sort()
        for tr_obspy, tr_sb in zip(traces_obspy, traces_sb, strict=True):
            assert tr_obspy.id == tr_sb.id
            obspy_stats = tr_obspy.stats
            sb_stats = tr_sb.stats

            # We use Python3 rounding, so npts may differ by 1
            # assert abs(tr_obspy.stats.npts - tr_sb.stats.npts) in [0, 1]
            assert obspy_stats.npts == sb_stats.npts
            assert obspy_stats.starttime == sb_stats.starttime
            assert obspy_stats.endtime == sb_stats.endtime
