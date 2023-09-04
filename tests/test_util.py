import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import obspy
import pytest
import requests
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException

import seisbench.util
from seisbench.util.trace_ops import (
    fdsn_get_bulk_safe,
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
