import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from seisbench.data.multiclient import (
    FDSNSource,
    HinetRequest,
    HinetSource,
    WaveformRequest,
    _UnauthorizedError,
)

START = UTCDateTime(2010, 5, 1, 3, 0, 20)


def _request(station: str, offset: float, length: float) -> WaveformRequest:
    return WaveformRequest(
        "N", station, "", "EH?", START + offset, START + offset + length
    )


def _source(fetched: list, cache_dir, no_data=()) -> HinetSource:
    source = HinetSource("user", "password", cache_dir=cache_dir, retries=1)

    def fetch(request: HinetRequest) -> Stream:
        fetched.append(request)
        npts = int((request.endtime - request.starttime) * 100)
        # Values are the sample times, to check that blocks are joined correctly
        data = request.starttime + np.arange(npts) / 100
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
        np.testing.assert_allclose(trace.data[0], request.starttime.timestamp)
        np.testing.assert_allclose(np.diff(trace.data), 0.01, atol=1e-6)


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
