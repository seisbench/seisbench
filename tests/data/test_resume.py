import sys

import numpy as np
import pandas as pd
import pytest

import seisbench.data as sbd
from seisbench.util.file import _try_lock, callback_if_uncached


def _add(writer, names):
    for i, name in enumerate(names):
        writer.add_trace(
            {"trace_name": name, "station_location_code": "00", "split": "train"},
            np.full((3, 100 + i), float(i)),
        )


def _writer(path, resume=False):
    writer = sbd.WaveformDataWriter(
        path / "metadata.csv", path / "waveforms.hdf5", resume=resume
    )
    writer.data_format = {"dimension_order": "CW", "component_order": "ZNE"}
    return writer


def _kill(writer):
    # A hard kill: no finalize, the files are just left behind
    writer._waveform_file.close()


@pytest.mark.parametrize("bucket_size", [1024, 1])
def test_writer_resume(tmp_path, bucket_size):
    writer = _writer(tmp_path)
    writer.bucket_size = bucket_size
    _add(writer, ["a", "b", "c"])
    writer.checkpoint()
    _add(writer, ["d"])  # Not checkpointed: lost
    _kill(writer)

    with _writer(tmp_path, resume=True) as writer:
        writer.bucket_size = bucket_size
        assert writer.resumed_trace_names == {"a", "b", "c"}
        _add(writer, ["d", "e"])

    metadata = pd.read_csv(tmp_path / "metadata.csv", dtype=str, keep_default_na=False)
    names = metadata["trace_name"]
    if "trace_name_original" in metadata:
        names = metadata["trace_name_original"].where(
            metadata["trace_name_original"] != "", names
        )
    assert sorted(names) == ["a", "b", "c", "d", "e"]
    # Values are written back unchanged, e.g., location codes are not read as numbers
    assert set(metadata["station_location_code"]) == {"00"}

    dataset = sbd.WaveformDataset(tmp_path)
    assert len(dataset) == 5
    for i in range(len(dataset)):
        assert dataset.get_waveforms(i).shape[0] == 3


def test_writer_resume_after_error(tmp_path):
    # An exception finalizes the files, which can then be resumed, too
    with pytest.raises(RuntimeError):
        with _writer(tmp_path) as writer:
            _add(writer, ["a", "b"])
            raise RuntimeError("download failed")

    with _writer(tmp_path, resume=True) as writer:
        assert writer.resumed_trace_names == {"a", "b"}
        _add(writer, ["c"])

    assert len(sbd.WaveformDataset(tmp_path)) == 3


def test_callback_resume(tmp_path):
    file = tmp_path / "metadata.csv"
    partial = tmp_path / "metadata.csv.partial"
    partial.write_text("progress")
    calls = []

    def callback(path):
        calls.append(path.read_text())
        path.write_text(path.read_text() + " done")

    # Without resume, partial files mean another download is running or failed
    with pytest.raises(ValueError, match="partial"):
        callback_if_uncached(file, callback)

    if sys.platform != "win32":  # Without fcntl, nothing is locked
        # Another process holds the lock: still running
        lock = _try_lock(file)
        with pytest.raises(ValueError, match="partial"):
            callback_if_uncached(file, callback, resume=True)
        lock.close()

    # The lock is free: the earlier download stopped, so resume from its files
    callback_if_uncached(file, callback, resume=True)
    assert calls == ["progress"]
    assert file.read_text() == "progress done"
    assert not (tmp_path / "metadata.csv.lock").exists()
