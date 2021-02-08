import seisbench.models

import numpy as np
import obspy
from obspy import UTCDateTime


def test_weights_docstring():
    model = seisbench.models.GPD()
    assert model.weights_docstring is None

    model = seisbench.models.GPD.from_pretrained("dummy")
    assert isinstance(model.weights_docstring, str)


def test_has_mismatching_records():
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

    # Overlapping matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 10), trace0.slice(t0 + 5, t0 + 15)])
    assert not seisbench.models.WaveformModel.has_mismatching_records(stream)

    # Overlapping not matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 10), trace1.slice(t0 + 5, t0 + 15)])
    assert seisbench.models.WaveformModel.has_mismatching_records(stream)

    # Full intersection matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 15), trace0.slice(t0 + 5, t0 + 10)])
    assert not seisbench.models.WaveformModel.has_mismatching_records(stream)

    # Full intersection not matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 15), trace1.slice(t0 + 5, t0 + 10)])
    assert seisbench.models.WaveformModel.has_mismatching_records(stream)

    # No intersection matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 10), trace0.slice(t0 + 15, t0 + 20)])
    assert not seisbench.models.WaveformModel.has_mismatching_records(stream)

    # No intersection not matching
    stream = obspy.Stream([trace0.slice(t0, t0 + 10), trace1.slice(t0 + 15, t0 + 20)])
    assert not seisbench.models.WaveformModel.has_mismatching_records(stream)

    # Three traces matching
    stream = obspy.Stream(
        [
            trace0.slice(t0, t0 + 10),
            trace0.slice(t0 + 5, t0 + 15),
            trace0.slice(t0 + 7, t0 + 11),
        ]
    )
    assert not seisbench.models.WaveformModel.has_mismatching_records(stream)

    # Three traces not matching
    stream = obspy.Stream(
        [
            trace0.slice(t0, t0 + 10),
            trace0.slice(t0 + 5, t0 + 15),
            trace1.slice(t0 + 7, t0 + 11),
        ]
    )
    assert seisbench.models.WaveformModel.has_mismatching_records(stream)


class DummyWaveformModel(seisbench.models.WaveformModel):
    def annotate(self, stream, *args, **kwargs):
        pass

    def classify(self, stream, *args, **kwargs):
        pass


def test_stream_to_arrays():
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
