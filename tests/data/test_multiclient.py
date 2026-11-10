import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn.header import FDSNException

from seisbench.data.multiclient import (
    FDSNSource,
    HinetRequest,
    HinetSource,
    WaveformRequest,
    _TransientError,
    _UnauthorizedError,
    antialias_filter,
    hinet_counts,
    resample,
)

START = UTCDateTime(2010, 5, 1, 3, 0, 20)
# Fake Hi-net counts are the number of samples since this time
DAY = UTCDateTime(2010, 5, 1)


def _request(station: str, offset: float, length: float) -> WaveformRequest:
    return WaveformRequest(
        "N", station, "", "EH?", START + offset, START + offset + length
    )


def _source(fetched: list, cache_dir, no_data=()) -> HinetSource:
    source = HinetSource("user", "password", cache_dir=cache_dir, retries=1)

    def fetch(request: HinetRequest) -> Stream:
        fetched.append(request)
        npts = int((request.endtime - request.starttime) * 100)
        # Counts are the sample times, to check that blocks are joined correctly
        first = round((request.starttime - DAY.timestamp) * 100)
        data = (first + np.arange(npts)).astype(np.int32)
        return Stream(
            [
                Trace(
                    data.copy(),
                    header=dict(
                        network="N",
                        station=station,
                        channel=f"EH{component}",
                        starttime=UTCDateTime(request.starttime),
                        sampling_rate=100.0,
                    ),
                )
                for station in request.stations
                if station not in no_data
                for component in "ZNE"
            ]
        )

    source._fetch_once = fetch
    return source


def _get_all(source, requests):
    async def get_all():
        return [await source.get_waveforms(None, request) for request in requests]

    return asyncio.run(get_all())


def _check(stream, request):
    assert {trace.stats.station for trace in stream} == {request.station}
    assert len(stream) == 3
    for trace in stream:
        assert trace.stats.starttime == request.starttime
        assert trace.stats.endtime == request.endtime
        assert trace.data.dtype == np.int32
        assert trace.data[0] == round((request.starttime - DAY) * 100)
        assert np.all(np.diff(trace.data) == 1)


def test_hinet_plan_combines_requests(tmp_path):
    fetched = []
    source = _source(fetched, tmp_path)
    # An event at two stations, the noise before it at one, and an event 1 h later
    requests = [
        _request("AYEH", 400, 120),  # 03:07 - 03:09
        _request("BHRH", 410, 120),
        _request("AYEH", 100, 60),  # 03:02 - 03:03, noise
        _request("RKBH", 3600, 120),
    ]
    source.plan(requests)
    streams = _get_all(source, requests)

    # The event and its noise are blocks 03:00 to 03:10: one request
    assert len(fetched) == 2
    first, second = fetched
    assert first.stations == {"AYEH", "BHRH"}
    assert first.starttime == UTCDateTime(2010, 5, 1, 3, 0).timestamp
    assert first.endtime == UTCDateTime(2010, 5, 1, 3, 10).timestamp
    assert second.stations == {"RKBH"}
    for stream, request in zip(streams, requests):
        _check(stream, request)

    # Cached: another source with the same cache downloads nothing
    fetched.clear()
    for stream, request in zip(
        _get_all(_source(fetched, tmp_path), requests), requests
    ):
        _check(stream, request)
    assert fetched == []


def test_hinet_longer_window_reuses_cache(tmp_path):
    fetched = []
    source = _source(fetched, tmp_path)
    _get_all(source, [_request("AYEH", 400, 120)])  # 03:07 - 03:09, block 03:05
    # +-1 min more: only the blocks before and after are downloaded
    request = _request("AYEH", 340, 240)  # 03:06 - 03:10
    (stream,) = _get_all(source, [request])
    _check(stream, request)
    assert fetched[1:] == [
        HinetRequest(
            UTCDateTime(2010, 5, 1, 3, 5).timestamp,
            UTCDateTime(2010, 5, 1, 3, 15).timestamp,
            frozenset({"AYEH"}),
        )
    ]


def test_hinet_no_data_is_cached(tmp_path):
    fetched = []
    source = _source(fetched, tmp_path, no_data={"AYEH"})
    request = _request("AYEH", 0, 60)
    assert _get_all(source, [request, request]) == [None, None]
    assert len(fetched) == 1


def test_hinet_failure_is_not_cached(tmp_path):
    source = HinetSource("user", "password", cache_dir=tmp_path, retries=1)

    def fail(request):
        raise ConnectionError("Hi-net returned no data.")

    source._fetch_once = fail
    with pytest.raises(ConnectionError):
        _get_all(source, [_request("AYEH", 0, 60)])
    assert not list(tmp_path.rglob("*.mseed"))


def test_hinet_counts():
    # win2sac_32 writes counts / sensitivity * 1e9 as float32
    sensitivity = 1.671e9  # counts per m/s, e.g., N.BHRH
    counts = np.array([-1721, 0, 1, 1576, 2**22], dtype=np.int32)
    velocity = (counts / sensitivity * 1e9).astype(np.float32)
    np.testing.assert_array_equal(hinet_counts(velocity, sensitivity), counts)

    with pytest.raises(ValueError):
        hinet_counts(velocity, sensitivity * 1.5)


def test_hinet_get_stations():
    source = HinetSource("user", "password")
    source._stations = [
        SimpleNamespace(
            code="0101", name="N.AYEH", latitude=35.1, longitude=138.2, elevation=100.0
        ),
        SimpleNamespace(
            code="0101", name="N.BHRH", latitude=36.0, longitude=139.0, elevation=None
        ),
        SimpleNamespace(
            code="0101", name="N.XXXH", latitude=None, longitude=None, elevation=None
        ),
    ]
    inv = asyncio.run(
        source.get_stations(
            None, {"AYEH", "BHRH", "XXXH", "ABC"}, START, START, "channel"
        )
    )
    assert [station.code for station in inv[0]] == ["AYEH", "BHRH"]
    coordinates = inv.get_coordinates("N.AYEH..EHZ", START)
    assert coordinates["latitude"] == 35.1

    # Other networks are not served
    assert (
        len(
            asyncio.run(
                source.get_stations(None, {"AYEH"}, START, START, "channel", {"AV"})
            )
        )
        == 0
    )


TOKEN = "-----BEGIN PGP SIGNED MESSAGE-----\ntoken"


def test_fdsn_eida_token(tmp_path, monkeypatch):
    assert FDSNSource("https://x", eida_token=TOKEN).eida_token == TOKEN

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    (tmp_path / "token.pgp").write_text(TOKEN)
    assert FDSNSource("https://x", eida_token="~/token.pgp").eida_token == TOKEN

    with pytest.raises(FileNotFoundError):
        FDSNSource("https://x", eida_token="~/missing.pgp")


def test_fdsn_reauthenticate():
    # The server forgot the credentials: authenticate once again for all requests
    source = FDSNSource("https://x", eida_token=TOKEN)
    source._auth = "forgotten"
    calls = []

    async def authenticate(session):
        calls.append(session)
        await asyncio.sleep(0.01)
        source._auth = f"new{len(calls)}"

    async def get(session, url, params, middlewares):
        await asyncio.sleep(0)
        if middlewares == ("forgotten",):
            raise _UnauthorizedError("401")
        return middlewares[0].encode()

    source.authenticate = authenticate
    source._get = get

    async def run():
        return await asyncio.gather(
            *(source._get_authenticated(None, {}) for _ in range(4))
        )

    assert asyncio.run(run()) == [b"new1"] * 4
    assert len(calls) == 1


def test_resample_independent_of_length():
    # ObsPy's FFT resampling filters odd and even lengths differently
    data = np.random.default_rng(0).standard_normal(24001)
    streams = []
    for npts in (24000, 24001):
        stream = Stream([Trace(data[:npts].copy(), header=dict(sampling_rate=200.0))])
        resample(stream, 100.0)
        streams.append(stream)
    even, odd = streams[0][0], streams[1][0]
    assert even.stats.sampling_rate == odd.stats.sampling_rate == 100.0
    assert even.stats.starttime == odd.stats.starttime
    # Identical except for the filter length at the end, and for the slightly
    # different mean at the edges
    np.testing.assert_allclose(
        even.data[:-50], odd.data[: even.stats.npts - 50], atol=1e-4
    )


@pytest.mark.parametrize("rate", [200.0, 250.0])
def test_resample_sine(rate):
    t = np.arange(int(60 * rate)) / rate
    trace = Trace(1000.0 + np.sin(2 * np.pi * 5.0 * t), header=dict(sampling_rate=rate))
    stream = Stream([trace])
    resample(stream, 100.0)
    trace = stream[0]
    assert trace.stats.sampling_rate == 100.0
    assert trace.stats.starttime == UTCDateTime(0)
    assert trace.stats.npts == 6000
    expected = 1000.0 + np.sin(2 * np.pi * 5.0 * trace.times())
    # Amplitude, phase and offset are kept. The edges settle within 0.2 s.
    np.testing.assert_allclose(trace.data[20:-20], expected[20:-20], atol=1e-3)
    np.testing.assert_allclose(trace.data, expected, atol=0.1)


@pytest.mark.parametrize("rate", [200.0, 250.0])
def test_resample_antialias(rate):
    # Flat up to 0.7 x Nyquist, e.g., no 50 Hz hum left after resampling to 100 Hz
    t = np.arange(int(60 * rate)) / rate
    amplitudes = {}
    for frequency in (30.0, 50.0, 60.0):
        stream = Stream(
            [Trace(np.sin(2 * np.pi * frequency * t), header=dict(sampling_rate=rate))]
        )
        resample(stream, 100.0)
        amplitudes[frequency] = np.sqrt(2) * stream[0].data[100:-100].std()
    assert amplitudes[30.0] == pytest.approx(1.0, abs=1e-3)
    assert amplitudes[50.0] < 1e-3
    assert amplitudes[60.0] < 1e-3
    assert antialias_filter.cache_info().hits > 0


class _FakeResponse:
    def __init__(self, status):
        self.status = status
        self.url = "https://x/fdsnws/dataselect/1/query"

    async def read(self):
        return b"data"

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        pass


class _FakeSession:
    """Answers with the given status codes, or raises the given exceptions."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def get(self, url, params, **kwargs):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _FakeResponse(answer)


def test_fdsn_retries():
    aiohttp = pytest.importorskip("aiohttp")

    source = FDSNSource("https://x", retries=2, retry_wait=0.0)

    def get(answers):
        session = _FakeSession(answers)
        result = asyncio.run(source._get(session, "https://x", {}))
        return result, session.calls

    assert get([503, 200]) == (b"data", 2)
    assert get([aiohttp.ServerDisconnectedError(), 200]) == (b"data", 2)
    assert get([asyncio.TimeoutError(), 429, 200]) == (b"data", 3)
    # No data and permanent errors are not retried
    assert get([204]) == (None, 1)
    with pytest.raises(FDSNException):
        get([400])
    # Gives up after the retries
    with pytest.raises(_TransientError):
        get([503, 503, 503])
