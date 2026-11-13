import numpy as np
import pytest

import seisbench.data as sbd

pytest.importorskip("pyrocko")

from pyrocko import io  # noqa: E402
from pyrocko.gui.marker import load_markers  # noqa: E402
from pyrocko.model import load_events  # noqa: E402

from seisbench.data.inspection import DatasetInspection  # noqa: E402


def _dataset(path):
    rng = np.random.default_rng(0)
    with sbd.WaveformDataWriter(
        path / "metadata.csv", path / "waveforms.hdf5"
    ) as writer:
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": 100,
        }
        for i_event in range(2):
            for station, location in (("STA1", "00"), ("STA2", None)):
                metadata = {
                    "trace_name": f"ev{i_event}_{station}",
                    "split": "train",
                    "source_id": f"ev{i_event}",
                    "source_origin_time": f"2020-01-0{i_event + 1}T12:00:00.000000Z",
                    "source_latitude_deg": 50.0,
                    "source_longitude_deg": 12.0,
                    "source_depth_km": 10.0,
                    "source_magnitude": 2.0,
                    "source_magnitude_type": "ML",
                    "station_network_code": "XX",
                    "station_code": station,
                    "station_latitude_deg": 50.1,
                    "station_longitude_deg": 12.1,
                    "station_elevation_m": 500.0,
                    "trace_channel": "HH",
                    "trace_start_time": f"2020-01-0{i_event + 1}T11:59:50.000000Z",
                    "trace_sampling_rate_hz": 100.0,
                    "trace_P_arrival_sample": 1200,
                    "trace_P_status": "manual",
                    "trace_S_arrival_sample": 1500,
                    "trace_S_status": "manual",
                }
                if location is not None:
                    metadata["station_location_code"] = location
                writer.add_trace(metadata, rng.normal(size=(3, 3000)))
    return sbd.WaveformDataset(path, component_order="ZNE")


def _load(directory):
    markers = load_markers(str(directory / "all_picks.picks"))
    traces = [
        trace
        for file in sorted((directory / "mseed").glob("*.mseed"))
        for trace in io.load(str(file))
    ]
    return markers, traces


def test_export_location_codes(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    DatasetInspection(dataset).export(tmp_path / "export")

    markers, traces = _load(tmp_path / "export")
    pick_nsl = {marker.nslc_ids[0][:3] for marker in markers}
    # A location code such as 00, and none
    assert pick_nsl == {("XX", "STA1", "00"), ("XX", "STA2", "")}
    assert pick_nsl == {trace.nslc_id[:3] for trace in traces}
    assert {marker.get_phasename() for marker in markers} == {"P", "S"}
    assert len(markers) == 8
    assert len(load_events(str(tmp_path / "export" / "events.yaml"))) == 2


def test_export_filtered(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    dataset.filter(dataset.metadata["source_id"] == "ev1")
    # Indices as used by get_sample, not the labels of the filtered metadata
    assert dataset.get_event_sample_indices("ev1") == [0, 1]

    DatasetInspection(dataset).export(tmp_path / "export")
    markers, traces = _load(tmp_path / "export")
    assert len(markers) == 4
    assert len(traces) == 6
    events = load_events(str(tmp_path / "export" / "events.yaml"))
    assert [event.extras["id"] for event in events] == ["ev1"]
